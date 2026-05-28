import json
import logging
import os
from multiprocessing.dummy import Pool as ThreadPool

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils import data as torch_data
from torch.utils.data import Dataset
from tqdm import tqdm

from gfmrag import utils
from gfmrag.datasets import QADataset
from gfmrag.prompt_builder import QAPromptBuilder
from gfmrag.ultra import query_utils

# A logger for this file
logger = logging.getLogger(__name__)


@torch.no_grad()
def doc_retrieval(
    cfg: DictConfig,
    model: nn.Module,
    qa_data: Dataset,
    device: torch.device,
) -> list[dict]:
    world_size = utils.get_world_size()
    rank = utils.get_rank()

    _, test_data = qa_data._data
    graph = qa_data.kg
    ent2docs = qa_data.ent2docs

    # Retrieve the supporting documents for each query
    sampler = torch_data.DistributedSampler(test_data, world_size, rank, shuffle=False)
    test_loader = torch_data.DataLoader(
        test_data, cfg.test.retrieval_batch_size, sampler=sampler
    )

    # Create doc retriever
    doc_ranker = instantiate(cfg.doc_ranker, ent2doc=ent2docs)

    if cfg.test.init_entities_weight:
        entities_weight = utils.get_entities_weight(ent2docs)
    else:
        entities_weight = None

    model.eval()
    all_predictions: list[dict] = []
    for batch in tqdm(test_loader):
        batch = query_utils.cuda(batch, device=device)
        ent_pred = model(graph, batch, entities_weight=entities_weight)
        doc_pred = doc_ranker(ent_pred)  # Ent2docs mapping
        idx = batch["sample_id"]
        all_predictions.extend(
            {"id": int(i.item()), "ent_pred": e, "doc_pred": d}
            for i, e, d in zip(idx.cpu(), ent_pred.cpu(), doc_pred.cpu())
        )

    # Gather the predictions across all processes
    if utils.get_world_size() > 1:
        gathered_predictions = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered_predictions, all_predictions)
    else:
        gathered_predictions = [all_predictions]  # type: ignore

    sorted_predictions = sorted(
        [item for sublist in gathered_predictions for item in sublist],  # type: ignore
        key=lambda x: x["id"],
    )
    deduped_predictions = []
    seen_prediction_ids = set()
    duplicate_count = 0
    for item in sorted_predictions:
        sample_idx = int(item["id"])
        if sample_idx in seen_prediction_ids:
            duplicate_count += 1
            continue
        seen_prediction_ids.add(sample_idx)
        deduped_predictions.append(item)

    if duplicate_count and rank == 0:
        logger.warning(
            "Removed %s duplicate retrieval predictions introduced by distributed "
            "sampling padding.",
            duplicate_count,
        )
    if len(deduped_predictions) != len(test_data) and rank == 0:
        logger.warning(
            "Retrieval prediction count (%s) does not match test sample count (%s).",
            len(deduped_predictions),
            len(test_data),
        )
    utils.synchronize()
    return deduped_predictions


def ans_prediction(
    cfg: DictConfig, output_dir: str, qa_data: Dataset, retrieval_result: list[dict]
) -> str:
    if not retrieval_result:
        raise ValueError(
            "No retrieval results were produced. This usually means the dataset has "
            "zero test samples; check processed/stage1/test.json for "
            f"{cfg.dataset.data_name}."
        )

    llm = instantiate(cfg.llm)
    doc_retriever = utils.DocumentRetriever(qa_data.doc, qa_data.id2doc)
    raw_test_data = qa_data.raw_test_data
    raw_test_data_by_idx = {idx: item for idx, item in enumerate(raw_test_data)}
    id2ent = {v: k for k, v in qa_data.ent2id.items()}

    prompt_builder = QAPromptBuilder(cfg.qa_prompt)

    def predict(retrieval_doc: dict) -> dict | Exception:
        sample_idx = retrieval_doc["id"]
        if isinstance(sample_idx, torch.Tensor):
            sample_idx = int(sample_idx.item())
        else:
            sample_idx = int(sample_idx)
        data = raw_test_data_by_idx[sample_idx]
        save_top_k_entity = min(
            int(cfg.test.save_top_k_entity),
            retrieval_doc["ent_pred"].numel(),
        )
        retrieved_ent_idx = torch.topk(
            retrieval_doc["ent_pred"], save_top_k_entity, dim=-1
        ).indices
        retrieved_ent = [id2ent[i.item()] for i in retrieved_ent_idx]
        retrieved_docs = doc_retriever(retrieval_doc["doc_pred"], top_k=cfg.test.top_k)

        message = prompt_builder.build_input_prompt(data["question"], retrieved_docs)

        response = llm.generate_sentence(message)
        if isinstance(response, Exception):
            return response
        else:
            return {
                "id": data["id"],
                "question": data["question"],
                "answer": data["answer"],
                "answer_aliases": data.get(
                    "answer_aliases", []
                ),  # Some datasets have answer aliases
                "response": response,
                "retrieved_ent": retrieved_ent,
                "retrieved_docs": retrieved_docs,
            }

    error_path = os.path.join(output_dir, "prediction_errors.jsonl")
    num_success = 0
    num_errors = 0
    with open(os.path.join(output_dir, "prediction.jsonl"), "w") as f, open(
        error_path, "w"
    ) as error_f:
        with ThreadPool(cfg.test.n_threads) as pool:
            for results in tqdm(
                pool.imap(predict, retrieval_result),
                total=len(retrieval_result),
            ):
                if isinstance(results, Exception):
                    logger.error(f"Error: {results}")
                    error_f.write(json.dumps({"error": repr(results)}) + "\n")
                    error_f.flush()
                    num_errors += 1
                    continue

                f.write(json.dumps(results) + "\n")
                f.flush()
                num_success += 1

    if num_success == 0:
        raise RuntimeError(
            "All answer generation calls failed. "
            f"See {error_path} for the captured errors. "
            f"Total failed calls: {num_errors}."
        )

    return os.path.join(output_dir, "prediction.jsonl")


@hydra.main(config_path="config", config_name="stage3_qa_inference", version_base=None)
def main(cfg: DictConfig) -> None:
    output_dir = HydraConfig.get().runtime.output_dir
    utils.init_distributed_mode()
    torch.manual_seed(cfg.seed + utils.get_rank())
    if utils.get_rank() == 0:
        logger.info(f"Config:\n {OmegaConf.to_yaml(cfg)}")
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Output directory: {output_dir}")

    model, model_config = utils.load_model_from_pretrained(
        cfg.graph_retriever.model_path
    )
    qa_data = QADataset(
        **cfg.dataset,
        text_emb_model_cfgs=OmegaConf.create(model_config["text_emb_model_config"]),
    )
    device = utils.get_device()
    model = model.to(device)

    qa_data.kg = qa_data.kg.to(device)
    qa_data.ent2docs = qa_data.ent2docs.to(device)

    if cfg.test.retrieved_result_path:
        retrieval_result = torch.load(cfg.test.retrieved_result_path, weights_only=True)
    else:
        if cfg.test.prediction_result_path:
            retrieval_result = None
        else:
            retrieval_result = doc_retrieval(cfg, model, qa_data, device=device)
    if utils.is_main_process():
        if cfg.test.save_retrieval and retrieval_result is not None:
            logger.info(
                f"Ranking saved to disk: {os.path.join(output_dir, 'retrieval_result.pt')}"
            )
            torch.save(
                retrieval_result, os.path.join(output_dir, "retrieval_result.pt")
            )
        if cfg.test.prediction_result_path:
            output_path = cfg.test.prediction_result_path
        else:
            output_path = ans_prediction(cfg, output_dir, qa_data, retrieval_result)

        # Evaluation
        evaluator = instantiate(
            cfg.qa_evaluator,
            prediction_file=output_path,
            output_dir=output_dir,
            seed=cfg.seed,
            total_query_count=len(qa_data.raw_test_data),
        )
        metrics = evaluator.evaluate()
        query_utils.print_metrics(metrics, logger)
        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)
        return metrics


if __name__ == "__main__":
    main()
