import argparse
import json
import logging
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2" 

from src.hipporag.HippoRAG import HippoRAG
from src.hipporag.gnn_hidden_triplet_miner import GNNHiddenTripletMiner
from src.hipporag.utils.config_utils import BaseConfig
from src.hipporag.utils.misc_utils import string_to_bool


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OPENAI_API_KEY"] = "123"


def configure_logging():
    logging.basicConfig(level=logging.INFO, force=True)
    # Keep tqdm progress bars stable by suppressing very chatty HTTP client logs.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


def build_save_dir(dataset: str, save_dir: str) -> str:
    if save_dir == "outputs":
        return os.path.join(save_dir, dataset)
    return f"{save_dir}_{dataset}"


def validate_existing_outputs(save_dir: str, llm_name: str, embedding_name: str) -> str:
    llm_label = llm_name.replace("/", "_")
    embedding_label = embedding_name.replace("/", "_")
    working_dir = os.path.join(save_dir, f"{llm_label}_{embedding_label}")
    graph_path = os.path.join(working_dir, "graph.pickle")
    openie_path = os.path.join(save_dir, f"openie_results_ner_{llm_label}.json")

    missing_paths = [path for path in [working_dir, graph_path, openie_path] if not os.path.exists(path)]
    if missing_paths:
        missing_text = "\n".join(missing_paths)
        raise FileNotFoundError(
            "GNN-only training needs an existing indexed HippoRAG workspace. "
            f"The following paths are missing:\n{missing_text}"
        )

    return working_dir


def save_gnn_training_summary(save_dir: str, llm_name: str, summary: dict):
    llm_label = llm_name.replace("/", "_")
    summary_path = os.path.join(save_dir, f"hidden_triplet_gnn_training_{llm_label}.json")

    os.makedirs(save_dir, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logging.info("GNN-only training summary saved to %s", summary_path)


def maybe_override(hidden_triplet_overrides: dict, key: str, value):
    if value is not None:
        hidden_triplet_overrides[key] = value


def parse_args():
    parser = argparse.ArgumentParser(description="Train only the hidden-triplet GNN on an existing HippoRAG index.")
    parser.add_argument("--dataset", type=str, default="musique", help="Dataset name used to build the existing index.")
    parser.add_argument("--save_dir", type=str, default="outputs", help="Base save directory or the dataset output prefix.")
    parser.add_argument("--llm_name", type=str, default="gpt-4o-mini", help="LLM name used by the existing index.")
    parser.add_argument(
        "--llm_base_url",
        type=str,
        default="https://api.openai.com/v1",
        help="LLM base URL. This script does not call the LLM, but the HippoRAG object still expects the setting.",
    )
    parser.add_argument(
        "--embedding_name",
        type=str,
        default="nvidia/NV-Embed-v2",
        help="Embedding model name used by the existing index.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Global random seed for standalone GNN training.")
    parser.add_argument(
        "--embedding_base_url",
        type=str,
        default="http://0.0.0.0:8000/v1/embeddings",
        help="Embedding base URL used to load embedding stores.",
    )
    parser.add_argument(
        "--load_hidden_triplet_augmented_content",
        type=str,
        default=None,
        help="Whether to load previously saved hidden-triplet augmented graph/facts before training.",
    )
    parser.add_argument(
        "--hidden_triplet_gnn_hidden_dim",
        type=int,
        default=None,
        help="Hidden size of the PyG encoder.",
    )
    parser.add_argument(
        "--hidden_triplet_gnn_encoder_type",
        type=str,
        choices=["gcn", "graphsage", "gatv2", "rgcn"],
        default=None,
        help="GNN encoder architecture used for hidden-triplet link prediction.",
    )
    parser.add_argument(
        "--hidden_triplet_gnn_epochs",
        type=int,
        default=None,
        help="Training epochs for the PyG encoder.",
    )
    parser.add_argument(
        "--hidden_triplet_learning_rate",
        type=float,
        default=None,
        help="Learning rate for the PyG encoder.",
    )
    parser.add_argument(
        "--hidden_triplet_graph_refresh_interval",
        type=int,
        default=None,
        help="Refresh the dynamically remasked fact-edge split and support graph once every k epochs.",
    )
    parser.add_argument(
        "--hidden_triplet_early_stopping_patience",
        type=int,
        default=None,
        help="Stop standalone GNN tuning after this many evaluation points without eval_loss improvement. Set <= 0 to disable.",
    )
    parser.add_argument(
        "--hidden_triplet_early_stopping_min_delta",
        type=float,
        default=None,
        help="Minimum eval_loss decrease required to reset early stopping patience.",
    )
    parser.add_argument("--hidden_triplet_subgraph_walk_length", type=int, default=None)
    parser.add_argument("--hidden_triplet_subgraph_walks_per_root", type=int, default=None)
    parser.add_argument("--hidden_triplet_subgraph_negative_walks_per_root", type=int, default=None)
    parser.add_argument("--hidden_triplet_subgraph_fact_walk_scale", type=float, default=None)
    parser.add_argument("--hidden_triplet_subgraph_synonym_walk_scale", type=float, default=None)
    parser.add_argument("--hidden_triplet_subgraph_walk_weight_clip", type=float, default=None)
    parser.add_argument(
        "--hidden_triplet_subgraph_train_roots_per_epoch",
        type=int,
        default=None,
        help="Fixed number of root entities to sample each training epoch.",
    )
    parser.add_argument(
        "--hidden_triplet_subgraph_eval_roots",
        type=int,
        default=None,
        help="Fixed number of root entities to sample for evaluation. Set 0 to disable.",
    )
    parser.add_argument("--hidden_triplet_subgraph_fact_mask_ratio", type=float, default=None)
    parser.add_argument("--hidden_triplet_subgraph_entity_delete_ratio", type=float, default=None)
    parser.add_argument("--hidden_triplet_subgraph_min_nodes", type=int, default=None)
    parser.add_argument("--hidden_triplet_subgraph_min_fact_edges", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    configure_logging()

    dataset_save_dir = build_save_dir(args.dataset, args.save_dir)
    working_dir = validate_existing_outputs(
        save_dir=dataset_save_dir,
        llm_name=args.llm_name,
        embedding_name=args.embedding_name,
    )

    hidden_triplet_overrides = {}
    if args.load_hidden_triplet_augmented_content is not None:
        hidden_triplet_overrides["load_hidden_triplet_augmented_content"] = string_to_bool(
            args.load_hidden_triplet_augmented_content
        )
    maybe_override(hidden_triplet_overrides, "hidden_triplet_gnn_hidden_dim", args.hidden_triplet_gnn_hidden_dim)
    maybe_override(hidden_triplet_overrides, "hidden_triplet_gnn_encoder_type", args.hidden_triplet_gnn_encoder_type)
    maybe_override(hidden_triplet_overrides, "hidden_triplet_gnn_epochs", args.hidden_triplet_gnn_epochs)
    maybe_override(hidden_triplet_overrides, "hidden_triplet_learning_rate", args.hidden_triplet_learning_rate)
    maybe_override(
        hidden_triplet_overrides,
        "hidden_triplet_graph_refresh_interval",
        args.hidden_triplet_graph_refresh_interval,
    )
    maybe_override(
        hidden_triplet_overrides,
        "hidden_triplet_early_stopping_patience",
        args.hidden_triplet_early_stopping_patience,
    )
    maybe_override(
        hidden_triplet_overrides,
        "hidden_triplet_early_stopping_min_delta",
        args.hidden_triplet_early_stopping_min_delta,
    )
    for key in [
        "hidden_triplet_subgraph_walk_length",
        "hidden_triplet_subgraph_walks_per_root",
        "hidden_triplet_subgraph_negative_walks_per_root",
        "hidden_triplet_subgraph_fact_walk_scale",
        "hidden_triplet_subgraph_synonym_walk_scale",
        "hidden_triplet_subgraph_walk_weight_clip",
        "hidden_triplet_subgraph_train_roots_per_epoch",
        "hidden_triplet_subgraph_eval_roots",
        "hidden_triplet_subgraph_fact_mask_ratio",
        "hidden_triplet_subgraph_entity_delete_ratio",
        "hidden_triplet_subgraph_min_nodes",
        "hidden_triplet_subgraph_min_fact_edges",
    ]:
        maybe_override(hidden_triplet_overrides, key, getattr(args, key))
    hidden_triplet_overrides.setdefault("load_hidden_triplet_augmented_content", False)

    config = BaseConfig(
        save_dir=dataset_save_dir,
        dataset=args.dataset,
        llm_name=args.llm_name,
        llm_base_url=args.llm_base_url,
        seed=args.seed,
        embedding_model_name=args.embedding_name,
        enable_hidden_triplet_mining=False,
        **hidden_triplet_overrides,
    )

    logging.info("Loading HippoRAG workspace from %s", working_dir)
    hipporag = HippoRAG(
        global_config=config,
        embedding_base_url=args.embedding_base_url,
    )

    miner = GNNHiddenTripletMiner(hipporag)
    if not hipporag.ready_to_retrieve:
        hipporag.prepare_retrieval_objects()

    graph_data = miner._extract_full_graph_for_subgraph_mining()
    train_output = miner._train_subgraph_missingness_detector(graph_data)

    training_summary = {
        "hidden_triplet_mining_strategy": "subgraph_missingness",
        "num_graph_nodes": int(graph_data["num_nodes"]),
        "num_graph_edges": int(graph_data["num_edges"]),
        "num_fact_edges": int(graph_data["num_fact_edges"]),
        "num_synonym_edges": int(graph_data["num_synonym_edges"]),
        "num_entity_chunk_edges": int(graph_data["num_entity_chunk_edges"]),
        "gnn_training_time_sec": round(float(train_output["training_time_sec"]), 4),
        "gnn_train_loss": round(float(train_output["train_loss"]), 6),
        "gnn_eval_loss": None if train_output["eval_loss"] is None else round(float(train_output["eval_loss"]), 6),
        "gnn_best_eval_loss": (
            None if train_output["best_eval_loss"] is None else round(float(train_output["best_eval_loss"]), 6)
        ),
        "gnn_eval_accuracy": None if train_output["eval_accuracy"] is None else round(float(train_output["eval_accuracy"]), 6),
        "gnn_eval_f1": None if train_output["eval_f1"] is None else round(float(train_output["eval_f1"]), 6),
        "gnn_eval_ranking_accuracy": (
            None
            if train_output["eval_ranking_accuracy"] is None
            else round(float(train_output["eval_ranking_accuracy"]), 6)
        ),
        "hidden_triplet_gnn_hidden_dim": config.hidden_triplet_gnn_hidden_dim,
        "hidden_triplet_gnn_encoder_type": config.hidden_triplet_gnn_encoder_type,
        "hidden_triplet_gnn_epochs": config.hidden_triplet_gnn_epochs,
        "hidden_triplet_learning_rate": config.hidden_triplet_learning_rate,
        "hidden_triplet_graph_refresh_interval": config.hidden_triplet_graph_refresh_interval,
        "hidden_triplet_early_stopping_patience": config.hidden_triplet_early_stopping_patience,
        "hidden_triplet_early_stopping_min_delta": config.hidden_triplet_early_stopping_min_delta,
        "gnn_early_stopped": bool(train_output["early_stopped"]),
        "gnn_completed_epochs": int(train_output["completed_epochs"]),
        "gnn_best_epoch": train_output["best_epoch"],
        "gnn_best_eval_ranking_accuracy": train_output["best_eval_ranking_accuracy"],
        "gnn_training_logs": train_output["training_logs"],
        "hidden_triplet_subgraph_walk_length": config.hidden_triplet_subgraph_walk_length,
        "hidden_triplet_subgraph_walks_per_root": config.hidden_triplet_subgraph_walks_per_root,
        "hidden_triplet_subgraph_negative_walks_per_root": config.hidden_triplet_subgraph_negative_walks_per_root,
        "hidden_triplet_subgraph_fact_mask_ratio": config.hidden_triplet_subgraph_fact_mask_ratio,
        "hidden_triplet_subgraph_entity_delete_ratio": config.hidden_triplet_subgraph_entity_delete_ratio,
        "hidden_triplet_subgraph_train_roots_per_epoch": config.hidden_triplet_subgraph_train_roots_per_epoch,
        "hidden_triplet_subgraph_eval_roots": config.hidden_triplet_subgraph_eval_roots,
        "resolved_train_roots_per_epoch": train_output["resolved_train_roots_per_epoch"],
        "resolved_eval_roots": train_output["resolved_eval_roots"],
    }

    save_gnn_training_summary(dataset_save_dir, args.llm_name, training_summary)
    logging.info("GNN-only training finished: %s", training_summary)
    print(json.dumps(training_summary, indent=2))


if __name__ == "__main__":
    main()
