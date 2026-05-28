"""
Run LightRAG on HippoRAG-style multi-hop QA datasets with optional GNN subgraph
completion.

Supported datasets:
    musique, 2wikimultihopqa, hotpotqa

Example:
    python examples/lightrag_multihopqa_gnn_eval.py \
        --env_file GDM.env \
        --datasets musique 2wikimultihopqa hotpotqa \
        --query_mode mix \
        --output_dir outputs/lightrag_multihopqa_gnn
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3" 
import random
import re
import shutil
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lightrag.gnn_augment import (
    SubgraphCompletionConfig,
    augment_lightrag_with_subgraph_completion,
)
from lightrag.gnn_augment.evaluation import evaluate_literaryqa_predictions
from lightrag.gnn_augment.literaryqa import insert_literaryqa_book
from lightrag.utils import is_timeout_exception, logger


SUPPORTED_DATASETS = ("musique", "2wikimultihopqa", "hotpotqa")
DEFAULT_DATASET_ROOT = "/home/GDM/HippoRAG-main_v1/reproduce/dataset"
DEFAULT_RANDOM_SEED = 42
HIPPORAG_QA_RAG_PROMPT = """---Role---

You are an expert AI assistant specializing in synthesizing information from a provided knowledge base. Your primary function is to answer user queries accurately by ONLY using the information within the provided **Context**.

---Goal---

Generate a grounded answer to the user query.
The answer must integrate relevant facts from the Knowledge Graph and Document Chunks found in the **Context**.
Use Knowledge Graph Data to connect entities and relationships, and use Document Chunks to verify the textual evidence.
Consider the conversation history if provided to maintain conversational flow and avoid repeating information.

---Instructions---

1. Evidence Use:
  - Carefully determine the user's query intent in the context of the conversation history to fully understand the user's information need.
  - Scrutinize both `Knowledge Graph Data` and `Document Chunks` in the **Context**. Identify and extract all pieces of information that are directly relevant to answering the user query.
  - Weave the extracted facts into a coherent and logical response. Your own knowledge must ONLY be used to formulate fluent sentences and connect ideas, NOT to introduce any external information.
  - Track the reference_id of document chunks that directly support the facts used in the answer, but do not generate a References section.

2. Content & Grounding:
  - Strictly adhere to the provided context from the **Context**; DO NOT invent, assume, or infer any information not explicitly stated.
  - If the answer cannot be found in the **Context**, do not attempt to guess.

3. Output Format:
  - The response MUST be in the same language as the user query.
  - Start the response with `Thought: ` and briefly explain how the relevant Knowledge Graph Data and Document Chunks support the answer.
  - Conclude with exactly one final line beginning with `Answer: `.
  - The `Answer: ` line must contain only the concise, definitive answer string, with no markdown, bullets, citations, references, or extra explanation.
  - Do not generate anything after the `Answer: ` line.
  - If the answer cannot be found in the **Context**, end with exactly: `Answer: insufficient information`

---Context---

{context_data}
"""

HIPPORAG_QA_NAIVE_PROMPT = """---Role---

You are an expert AI assistant specializing in synthesizing information from a provided knowledge base. Your primary function is to answer user queries accurately by ONLY using the information within the provided **Context**.

---Goal---

Generate a grounded answer to the user query.
The answer must integrate relevant facts from the Document Chunks found in the **Context**.
Consider the conversation history if provided to maintain conversational flow and avoid repeating information.

---Instructions---

1. Evidence Use:
  - Carefully determine the user's query intent in the context of the conversation history to fully understand the user's information need.
  - Scrutinize `Document Chunks` in the **Context**. Identify and extract all pieces of information that are directly relevant to answering the user query.
  - Weave the extracted facts into a coherent and logical response. Your own knowledge must ONLY be used to formulate fluent sentences and connect ideas, NOT to introduce any external information.
  - Track the reference_id of document chunks that directly support the facts used in the answer, but do not generate a References section.

2. Content & Grounding:
  - Strictly adhere to the provided context from the **Context**; DO NOT invent, assume, or infer any information not explicitly stated.
  - If the answer cannot be found in the **Context**, do not attempt to guess.

3. Output Format:
  - The response MUST be in the same language as the user query.
  - Start the response with `Thought: ` and briefly explain how the relevant Document Chunks support the answer.
  - Conclude with exactly one final line beginning with `Answer: `.
  - The `Answer: ` line must contain only the concise, definitive answer string, with no markdown, bullets, citations, references, or extra explanation.
  - Do not generate anything after the `Answer: ` line.
  - If the answer cannot be found in the **Context**, end with exactly: `Answer: insufficient information`

---Context---

{content_data}
"""

DEFAULT_QA_RESPONSE_TYPE = "Multiple Paragraphs"
DEFAULT_QA_USER_PROMPT ="""
In addition to the requested response, include an Answer section that can be parsed automatically.

Answer Section Format:
- The Answer section heading must be exactly: `### Answer`
- Under `### Answer`, output exactly one line beginning with `Answer: `
- The text after `Answer: ` must be the concise final answer string only.
- Do not include markdown formatting, bullet points, citations, references, explanations, or reasoning after `Answer: `.
- If the answer cannot be found in the provided Context, output exactly: `Answer: insufficient information`

Answer Section Example:
```
### Answer
Answer:1862
```
"""


def hipporag_qa_system_prompt(query_mode: str) -> str:
    return HIPPORAG_QA_NAIVE_PROMPT if query_mode == "naive" else HIPPORAG_QA_RAG_PROMPT


def extract_answer_after_answer_prefix(text: str) -> str:
    if "Answer:" in text:
        return text.split("Answer:", 1)[1].strip()
    return text.strip()


def normalize_qa_prediction(raw_answer: str, args: argparse.Namespace) -> str:
    if args.extract_answer_after_answer_prefix:
        return extract_answer_after_answer_prefix(raw_answer)
    return raw_answer


@dataclass(slots=True)
class MultiHopQARun:
    dataset: str
    title: str
    document_id: str
    corpus_path: Path
    samples_path: Path
    book_output_subdir: str
    metadata: dict[str, Any]


def set_random_seed(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("GNN_SUBGRAPH_RANDOM_SEED", str(seed))
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    except ImportError:
        pass


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def result_filename(stem: str, variant: str | None = None) -> str:
    suffix = "_gnn" if variant == "gnn" else ""
    return f"{stem}{suffix}.json"


def resolve_datasets(raw: list[str]) -> list[str]:
    if any(item.lower() == "all" for item in raw):
        return list(SUPPORTED_DATASETS)
    datasets = []
    for item in raw:
        name = item.strip()
        if name not in SUPPORTED_DATASETS:
            raise ValueError(
                f"Unsupported dataset `{name}`. Choose from {SUPPORTED_DATASETS} or all."
            )
        if name not in datasets:
            datasets.append(name)
    return datasets


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dataset_run(dataset: str, dataset_root: Path) -> MultiHopQARun:
    corpus_path = dataset_root / f"{dataset}_corpus.json"
    samples_path = dataset_root / f"{dataset}.json"
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found for {dataset}: {corpus_path}")
    if not samples_path.exists():
        raise FileNotFoundError(f"Samples not found for {dataset}: {samples_path}")
    return MultiHopQARun(
        dataset=dataset,
        title=dataset,
        document_id=dataset,
        corpus_path=corpus_path,
        samples_path=samples_path,
        book_output_subdir=dataset,
        metadata={
            "dataset": dataset,
            "title": dataset,
            "document_id": dataset,
            "dataset_family": "multihopqa",
            "corpus_path": str(corpus_path),
            "samples_path": str(samples_path),
        },
    )


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def normalize_corpus(dataset: str, corpus: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for index, row in enumerate(corpus):
        title = _clean_text(row.get("title")) or f"{dataset}_{index:06d}"
        text = _clean_text(row.get("text") or row.get("paragraph_text"))
        content = f"{title}\n{text}".strip()
        source_idx = row.get("idx", index)
        try:
            source_idx_int = int(source_idx)
        except (TypeError, ValueError):
            source_idx_int = index
        rows.append(
            {
                "chunk_id": f"{dataset}__chunk_{source_idx_int:06d}",
                "chunk_index": index,
                "title": title,
                "text": content,
                "source_title": title,
                "source_idx": source_idx,
            }
        )
    return rows


def _join_context_sentences(dataset: str, sentences: list[Any]) -> str:
    parts = [str(sentence) for sentence in sentences]
    if dataset.startswith("hotpotqa"):
        return "".join(parts)
    return " ".join(parts)


def gold_support_titles(dataset: str, sample: dict[str, Any]) -> list[str]:
    titles: list[str] = []
    if sample.get("supporting_facts"):
        titles = [str(item[0]) for item in sample.get("supporting_facts") or [] if item]
    elif sample.get("paragraphs"):
        titles = [
            str(item.get("title"))
            for item in sample.get("paragraphs") or []
            if isinstance(item, dict) and item.get("is_supporting") is not False
        ]
    elif sample.get("contexts"):
        titles = [
            str(item.get("title"))
            for item in sample.get("contexts") or []
            if isinstance(item, dict) and item.get("is_supporting")
        ]
    return sorted({title.strip() for title in titles if title and title.strip()})


def gold_support_docs(dataset: str, sample: dict[str, Any]) -> list[str]:
    docs: list[str] = []
    if sample.get("supporting_facts") and sample.get("context"):
        gold_titles = set(gold_support_titles(dataset, sample))
        for title, sentences in sample.get("context") or []:
            if str(title) not in gold_titles:
                continue
            if isinstance(sentences, list):
                text = _join_context_sentences(dataset, sentences)
            else:
                text = str(sentences)
            docs.append(f"{title}\n{text}")
    elif sample.get("paragraphs"):
        for item in sample.get("paragraphs") or []:
            if not isinstance(item, dict) or item.get("is_supporting") is False:
                continue
            title = str(item.get("title") or "")
            text = str(item.get("text") or item.get("paragraph_text") or "")
            docs.append(f"{title}\n{text}".strip())
    elif sample.get("contexts"):
        for item in sample.get("contexts") or []:
            if not isinstance(item, dict) or not item.get("is_supporting"):
                continue
            title = str(item.get("title") or "")
            text = str(item.get("text") or "")
            docs.append(f"{title}\n{text}".strip())
    return list(dict.fromkeys(doc for doc in docs if doc))


def normalize_sample(dataset: str, sample: dict[str, Any]) -> dict[str, Any]:
    row = dict(sample)
    if row.get("id") is None and row.get("_id") is not None:
        row["id"] = row["_id"]
    if row.get("answer_aliases") is None:
        row["answer_aliases"] = []
    support_titles = gold_support_titles(dataset, row)
    support_docs = gold_support_docs(dataset, row)
    row["dataset"] = dataset
    row["gold_support_titles"] = support_titles
    row["gold_support_docs"] = support_docs
    row["gold_support_doc_count"] = len(support_docs)
    return row


def load_multihopqa_corpus(run: MultiHopQARun) -> list[dict[str, Any]]:
    return normalize_corpus(run.dataset, load_json(run.corpus_path))


def load_multihopqa_samples(run: MultiHopQARun) -> list[dict[str, Any]]:
    samples = load_json(run.samples_path)
    return [normalize_sample(run.dataset, sample) for sample in samples]


def build_embedding_func():
    from lightrag.llm.openai import openai_embed
    from lightrag.utils import EmbeddingFunc

    model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    return EmbeddingFunc(
        model_name=model,
        send_dimensions=False,
        embedding_dim=int(os.getenv("EMBEDDING_DIM", 1024)),
        max_token_size=int(os.getenv("EMBEDDING_TOKEN_LIMIT", 4096)),
        func=partial(
            openai_embed.func,
            model=model,
            base_url=os.getenv("EMBEDDING_BINDING_HOST", "http://0.0.0.0:8000/v1"),
            api_key=os.getenv("EMBEDDING_BINDING_API_KEY", "not_needed"),
        ),
    )


def build_rerank_func():
    if os.getenv("DISABLE_RERANK", "false").lower() in {"1", "true", "yes"}:
        return None
    from lightrag.rerank import jina_rerank

    return partial(
        jina_rerank,
        model=os.getenv(
            "RERANK_MODEL",
            "BAAI/bge-reranker-v2-m3",
        ),
        api_key=os.getenv("RERANK_BINDING_API_KEY"),
        base_url=os.getenv("RERANK_BINDING_HOST", "http://0.0.0.0:8300/v1/rerank"),
    )


def build_llm_func(timeout: int, temperature: float):
    from lightrag.llm.openai import openai_complete_if_cache

    async def llm_model_func(
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> str:
        kwargs["temperature"] = temperature
        return await openai_complete_if_cache(
            model=os.getenv("LLM_MODEL", "Models/Qwen3-32B"),
            prompt=prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            base_url=os.getenv("LLM_BINDING_HOST", "http://0.0.0.0:9500/v1"),
            api_key=os.getenv("LLM_BINDING_API_KEY", "not_needed"),
            timeout=timeout,
            **kwargs,
        )

    return llm_model_func


def dataset_workspace(run: MultiHopQARun, variant: str) -> str:
    return (Path(run.book_output_subdir) / variant).as_posix()


def active_output_dir(args: argparse.Namespace) -> Path:
    return Path(getattr(args, "_active_output_dir", args.output_dir))


def safe_cache_label(value: str) -> str:
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "")).strip("_")
    return label or "judge"


def answer_judge_cache_path(args: argparse.Namespace) -> str | None:
    if not args.answer_judge_cache:
        return None
    if args.answer_judge_cache_path:
        return args.answer_judge_cache_path
    model_label = safe_cache_label(args.answer_judge_model)
    return str(
        active_output_dir(args)
        / "llm_cache"
        / "answer_judge"
        / f"{model_label}_answer_judge_cache.json"
    )


def workspace_dir(args: argparse.Namespace, run: MultiHopQARun, variant: str) -> Path:
    return active_output_dir(args) / "books" / run.book_output_subdir / variant


def result_path_for_variant(
    args: argparse.Namespace,
    run: MultiHopQARun,
    variant: str,
) -> Path:
    working_dir = workspace_dir(args, run, variant)
    candidates = [working_dir / result_filename("multihopqa_results", variant)]
    if variant == "base":
        candidates.extend(
            [
                working_dir / "multihopqa_results_base.json",
                working_dir / "literaryqa_results.json",
                working_dir / "literaryqa_results_base.json",
            ]
        )
    else:
        candidates.extend(
            [
                working_dir / "multihopqa_results.json",
                working_dir / "literaryqa_results_gnn.json",
                working_dir / "literaryqa_results.json",
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def samples_and_predictions_from_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    samples = []
    predictions = []
    for idx, row in enumerate(rows):
        gold_answers = [
            str(answer)
            for answer in (row.get("gold_answers") or [])
            if answer is not None and str(answer).strip()
        ]
        samples.append(
            {
                "id": row.get("id") or row.get("_id") or row.get("qid") or idx,
                "dataset": row.get("dataset"),
                "question": row.get("question"),
                "answer": gold_answers[0] if gold_answers else "",
                "answer_aliases": gold_answers[1:] if len(gold_answers) > 1 else [],
            }
        )
        predictions.append(str(row.get("prediction") or ""))
    return samples, predictions


async def rerun_existing_evaluation(
    run: MultiHopQARun,
    variant: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    result_path = result_path_for_variant(args, run, variant)
    if not result_path.exists():
        raise FileNotFoundError(
            f"Existing {variant} result file not found for {run.dataset}: {result_path}"
        )

    report = load_json(result_path)
    prediction_rows = report.get("predictions") or []
    if not isinstance(prediction_rows, list):
        raise ValueError(f"{result_path} has no list `predictions` field.")

    samples, predictions = samples_and_predictions_from_rows(prediction_rows)
    eval_report = await evaluate_literaryqa_predictions(
        samples,
        predictions,
        enable_judge=args.enable_judge,
        judge_model=args.answer_judge_model,
        judge_base_url=args.answer_judge_base_url,
        judge_api_key_env=args.answer_judge_api_key_env,
        judge_max_tokens=args.answer_judge_max_tokens,
        judge_concurrency=args.answer_judge_concurrency,
        judge_max_retries=args.answer_judge_max_retries,
        judge_retry_delay=args.answer_judge_retry_delay,
        judge_cache_path=answer_judge_cache_path(args),
        enable_judge_cache=args.answer_judge_cache,
    )

    if args.eval_only_backup:
        backup_path = result_path.with_suffix(result_path.suffix + ".before_eval_only.bak")
        if not backup_path.exists():
            shutil.copy2(result_path, backup_path)

    report["evaluation"] = eval_report["aggregate"]
    report["per_example_eval"] = eval_report["examples"]
    report["evaluation_rerun"] = {
        "mode": "eval_only",
        "enable_judge": bool(args.enable_judge),
        "answer_judge_model": args.answer_judge_model,
        "answer_judge_base_url": args.answer_judge_base_url,
        "answer_judge_cache_path": answer_judge_cache_path(args),
        "num_examples": len(samples),
    }
    write_json(result_path, report)
    logger.info(
        "Re-evaluated %s %s from existing predictions: %s",
        run.dataset,
        variant,
        result_path,
    )
    return report


def _ignore_non_workspace_outputs(_dir: str, names: list[str]) -> set[str]:
    ignored = set()
    for name in names:
        if (
            name.startswith("multihopqa_results")
            or name.startswith("literaryqa_results")
        ) and name.endswith(".json"):
            ignored.add(name)
        elif name == "gnn_subgraph_completion":
            ignored.add(name)
    return ignored


def clone_base_workspace_for_gnn(
    run: MultiHopQARun,
    args: argparse.Namespace,
) -> dict[str, Any]:
    base_dir = workspace_dir(args, run, "base")
    gnn_dir = workspace_dir(args, run, "gnn")

    if not base_dir.exists():
        raise FileNotFoundError(
            f"Base workspace not found for {run.dataset}: {base_dir}. "
            "Run the base variant first, or run without --no-run_base."
        )

    if args.refresh_gnn_from_base and gnn_dir.exists():
        shutil.rmtree(gnn_dir)

    if not gnn_dir.exists():
        shutil.copytree(
            base_dir,
            gnn_dir,
            ignore=_ignore_non_workspace_outputs,
        )
        status = "copied"
    else:
        status = "reused_existing_gnn_workspace"

    return {
        "status": status,
        "dataset": run.dataset,
        "source_workspace": str(base_dir),
        "target_workspace": str(gnn_dir),
        "refresh_gnn_from_base": bool(args.refresh_gnn_from_base),
    }


async def initialize_rag(
    storage_root: Path,
    workspace: str,
    args: argparse.Namespace,
    *,
    allow_legacy_query_cache_fallback: bool,
):
    from lightrag import LightRAG

    rag = LightRAG(
        working_dir=str(storage_root),
        workspace=workspace,
        llm_model_func=build_llm_func(args.llm_timeout, args.llm_temperature),
        embedding_func=build_embedding_func(),
        rerank_model_func=build_rerank_func(),
        default_llm_timeout=args.llm_timeout,
        llm_model_max_async=args.lightrag_llm_concurrency,
        embedding_func_max_async=args.embedding_func_max_async,
        embedding_batch_num=args.embedding_batch_num,
        max_parallel_insert=args.max_parallel_insert,
        skip_failed_llm_calls=args.skip_failed_llm_calls,
        allow_legacy_query_cache_fallback=allow_legacy_query_cache_fallback,
        kv_storage=os.getenv("KV_STORAGE", "JsonKVStorage"),
        doc_status_storage=os.getenv("DOC_STATUS_STORAGE", "JsonDocStatusStorage"),
        vector_storage=os.getenv("VECTOR_STORAGE", "NanoVectorDBStorage"),
        graph_storage=os.getenv("GRAPH_STORAGE", "NetworkXStorage"),
    )
    await rag.initialize_storages()
    return rag


async def answer_samples(
    rag,
    samples: list[dict[str, Any]],
    args: argparse.Namespace,
    variant: str,
) -> tuple[list[str], list[dict[str, Any]]]:
    from lightrag import QueryParam

    semaphore = asyncio.Semaphore(max(1, args.query_concurrency))
    gnn_context_gate = "off"
    qa_system_prompt = (
        hipporag_qa_system_prompt(args.query_mode) if args.hipporag_qa_format else None
    )
    qa_response_type = None if args.hipporag_qa_format else args.qa_response_type
    qa_user_prompt = None if args.hipporag_qa_format else args.qa_user_prompt

    async def _answer_one(idx: int, sample: dict[str, Any]) -> tuple[int, str, dict[str, Any]]:
        async with semaphore:
            try:
                result = await rag.aquery_llm(
                    str(sample["question"]),
                    param=QueryParam(
                        mode=args.query_mode,
                        stream=False,
                        enable_rerank=args.enable_rerank,
                        top_k=args.top_k,
                        chunk_top_k=args.chunk_top_k,
                        max_total_tokens=args.max_total_tokens,
                        gnn_context_gate=gnn_context_gate,
                        response_type=qa_response_type,
                        user_prompt=qa_user_prompt,
                    ),
                    system_prompt=qa_system_prompt,
                )
                llm_response = result.get("llm_response", {})
                raw_answer = str(llm_response.get("content") or "")
                answer = normalize_qa_prediction(raw_answer, args)
                return idx, answer, {
                    "id": sample.get("id"),
                    "dataset": sample.get("dataset"),
                    "question": sample.get("question"),
                    "gold_answers": [sample.get("answer")]
                    + list(sample.get("answer_aliases") or []),
                    "gold_support_titles": sample.get("gold_support_titles") or [],
                    "gold_support_doc_count": sample.get("gold_support_doc_count"),
                    "prediction": answer,
                    "raw_prediction": raw_answer,
                    "qa_prompt_format": "hipporag" if args.hipporag_qa_format else "lightrag",
                    "retrieval_data": result.get("data", {}),
                    "metadata": result.get("metadata", {}),
                    "index": idx,
                }
            except Exception as exc:
                if args.skip_failed_llm_calls and is_timeout_exception(exc):
                    logger.warning(
                        "Skipping timed-out QA query %s `%s`: %s",
                        idx,
                        sample.get("question"),
                        exc,
                    )
                    return idx, "", {
                        "id": sample.get("id"),
                        "dataset": sample.get("dataset"),
                        "question": sample.get("question"),
                        "gold_answers": [sample.get("answer")]
                        + list(sample.get("answer_aliases") or []),
                        "gold_support_titles": sample.get("gold_support_titles") or [],
                        "gold_support_doc_count": sample.get("gold_support_doc_count"),
                        "prediction": "",
                        "raw_prediction": "",
                        "qa_prompt_format": "hipporag" if args.hipporag_qa_format else "lightrag",
                        "retrieval_data": {},
                        "metadata": {
                            "query_mode": args.query_mode,
                            "skipped": True,
                            "skip_reason": "timeout",
                        },
                        "index": idx,
                        "error": {
                            "type": type(exc).__name__,
                            "message": str(exc),
                        },
                    }
                raise

    results = await asyncio.gather(
        *[_answer_one(idx, sample) for idx, sample in enumerate(samples, start=1)]
    )
    results.sort(key=lambda item: item[0])
    predictions = [answer for _idx, answer, _row in results]
    rows = [row for _idx, _answer, row in results]
    return predictions, rows


async def run_variant(
    run: MultiHopQARun,
    samples: list[dict[str, Any]],
    corpus: list[dict[str, Any]],
    variant: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    working_dir = workspace_dir(args, run, variant)
    storage_root = active_output_dir(args) / "books"
    workspace_clone_report = None
    if variant == "gnn" and args.clone_base_to_gnn:
        workspace_clone_report = clone_base_workspace_for_gnn(run, args)
        logger.info(
            "GNN workspace for %s prepared from base: %s",
            run.dataset,
            workspace_clone_report["status"],
        )

    rag = await initialize_rag(
        storage_root,
        dataset_workspace(run, variant),
        args,
        allow_legacy_query_cache_fallback=variant == "base"
        and not args.hipporag_qa_format,
    )
    try:
        insert_report = await insert_literaryqa_book(rag, run, corpus)
        gnn_report = None
        if variant == "gnn":
            config = SubgraphCompletionConfig.from_reproduction_profile(
                run.dataset,
                artifact_dir=str(working_dir / "gnn_subgraph_completion")
            )
            config.random_seed = args.random_seed
            config.subgraph_missing_score_threshold = (
                args.gnn_subgraph_missing_score_threshold
            )
            config.subgraph_fact_mask_ratio = args.gnn_subgraph_fact_mask_ratio
            config.subgraph_entity_delete_ratio = args.gnn_subgraph_entity_delete_ratio
            config.subgraph_llm_budget = args.gnn_llm_budget
            config.reproduction_profile["parameters"].update(
                {
                    "subgraph_llm_budget": config.subgraph_llm_budget,
                    "subgraph_fact_mask_ratio": config.subgraph_fact_mask_ratio,
                    "subgraph_entity_delete_ratio": config.subgraph_entity_delete_ratio,
                    "subgraph_missing_score_threshold": config.subgraph_missing_score_threshold,
                }
            )
            gnn_report = await augment_lightrag_with_subgraph_completion(rag, config)

        predictions, prediction_rows = await answer_samples(rag, samples, args, variant)
        eval_report = await evaluate_literaryqa_predictions(
            samples,
            predictions,
            enable_judge=args.enable_judge,
            judge_model=args.answer_judge_model,
            judge_base_url=args.answer_judge_base_url,
            judge_api_key_env=args.answer_judge_api_key_env,
            judge_max_tokens=args.answer_judge_max_tokens,
            judge_concurrency=args.answer_judge_concurrency,
            judge_max_retries=args.answer_judge_max_retries,
            judge_retry_delay=args.answer_judge_retry_delay,
            judge_cache_path=answer_judge_cache_path(args),
            enable_judge_cache=args.answer_judge_cache,
        )
        report = {
            "dataset": run.metadata,
            "book": run.metadata,
            "variant": variant,
            "workspace_clone_report": workspace_clone_report,
            "insert_report": insert_report,
            "gnn_report": gnn_report,
            "evaluation": eval_report["aggregate"],
            "qa_error_count": sum(1 for row in prediction_rows if row.get("error")),
            "predictions": prediction_rows,
            "per_example_eval": eval_report["examples"],
        }
        write_json(working_dir / result_filename("multihopqa_results", variant), report)
        return report
    finally:
        await rag.finalize_storages()
        from lightrag.kg.shared_storage import finalize_share_data

        finalize_share_data()


async def main_async(args: argparse.Namespace) -> None:
    if args.env_file:
        load_dotenv(args.env_file, override=False)
    else:
        load_dotenv(override=False)

    if not args.run_base and not args.run_gnn:
        raise ValueError("At least one of --run_base or --run_gnn must be enabled.")

    set_random_seed(args.random_seed)
    datasets = resolve_datasets(args.datasets)
    if args.dataset_limit is not None:
        datasets = datasets[: max(0, args.dataset_limit)]

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_dir = Path(args.output_dir)
    separate_dataset_outputs = args.separate_dataset_outputs and len(datasets) > 1
    all_reports: dict[str, list[dict[str, Any]]] = {"base": [], "gnn": []}
    dataset_summaries: dict[str, Any] = {}
    dataset_output_dirs: dict[str, str] = {}

    for dataset in datasets:
        dataset_output_dir = output_dir / dataset if separate_dataset_outputs else output_dir
        setattr(args, "_active_output_dir", str(dataset_output_dir))
        dataset_reports: dict[str, list[dict[str, Any]]] = {"base": [], "gnn": []}
        dataset_output_dirs[dataset] = str(dataset_output_dir)
        run = dataset_run(dataset, dataset_root)

        if args.eval_only:
            logger.info("Re-evaluating existing predictions for %s", dataset)
            if args.run_base:
                base_report = await rerun_existing_evaluation(run, "base", args)
                dataset_reports["base"].append(base_report)
                all_reports["base"].append(base_report)
                write_json(
                    dataset_output_dir / "base_partial_results.json",
                    dataset_reports["base"],
                )
            if args.run_gnn:
                gnn_report = await rerun_existing_evaluation(run, "gnn", args)
                dataset_reports["gnn"].append(gnn_report)
                all_reports["gnn"].append(gnn_report)
                write_json(
                    dataset_output_dir / "gnn_partial_results_gnn.json",
                    dataset_reports["gnn"],
                )

            dataset_summary = summarize_reports(dataset_reports)
            dataset_summaries[dataset] = dataset_summary
            write_json(
                dataset_output_dir / summary_filename(args),
                {
                    "datasets": [dataset],
                    "summary": dataset_summary,
                    "reports": dataset_reports,
                    "eval_only": True,
                },
            )
            continue

        corpus = load_multihopqa_corpus(run)
        samples = load_multihopqa_samples(run)
        run.metadata.update(
            {
                "num_chunks": len(corpus),
                "num_questions": len(samples),
                "text_chars": sum(len(row.get("text", "")) for row in corpus),
                "text_words": sum(len(str(row.get("text", "")).split()) for row in corpus),
            }
        )
        if args.question_limit is not None:
            samples = samples[: args.question_limit]

        logger.info(
            "Running %s with %s chunks and %s questions",
            dataset,
            len(corpus),
            len(samples),
        )

        if args.run_base:
            base_report = await run_variant(run, samples, corpus, "base", args)
            dataset_reports["base"].append(base_report)
            all_reports["base"].append(base_report)
            write_json(
                dataset_output_dir / "base_partial_results.json",
                dataset_reports["base"],
            )

        if args.run_gnn:
            gnn_report = await run_variant(run, samples, corpus, "gnn", args)
            dataset_reports["gnn"].append(gnn_report)
            all_reports["gnn"].append(gnn_report)
            write_json(
                dataset_output_dir / "gnn_partial_results_gnn.json",
                dataset_reports["gnn"],
            )

        dataset_summary = summarize_reports(dataset_reports)
        dataset_summaries[dataset] = dataset_summary
        write_json(
            dataset_output_dir / summary_filename(args),
            {
                "datasets": [dataset],
                "summary": dataset_summary,
                "reports": dataset_reports,
            },
        )

    summary = summarize_reports(all_reports)
    final_payload = {
        "datasets": datasets,
        "eval_only": bool(args.eval_only),
        "separate_dataset_outputs": separate_dataset_outputs,
        "dataset_output_dirs": dataset_output_dirs,
        "summary": summary,
        "dataset_summaries": dataset_summaries,
    }
    if not separate_dataset_outputs:
        final_payload["reports"] = all_reports
    write_json(output_dir / summary_filename(args), final_payload)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def summarize_reports(reports_by_variant: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    summary = {}
    for variant, reports in reports_by_variant.items():
        if not reports:
            continue
        total = sum(len(report["per_example_eval"]) for report in reports)
        metrics = {}
        for metric in ("ExactMatch", "F1", "LLMAnswerConsistency"):
            values = []
            weights = []
            for report in reports:
                if metric not in report["evaluation"]:
                    continue
                count = len(report["per_example_eval"])
                values.append(report["evaluation"][metric])
                weights.append(count)
            if values:
                metrics[metric] = sum(v * w for v, w in zip(values, weights)) / max(
                    1, sum(weights)
                )
        summary[variant] = {
            "datasets": len(reports),
            "examples": total,
            "metrics": metrics,
        }
    return summary


def summary_filename(args: argparse.Namespace) -> str:
    return (
        "multihopqa_lightrag_base_gnn_summary.json"
        if args.run_base and args.run_gnn
        else "multihopqa_lightrag_base_summary.json"
        if args.run_base
        else "multihopqa_lightrag_gnn_summary.json"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(SUPPORTED_DATASETS),
        help="Datasets to run: musique, 2wikimultihopqa, hotpotqa, or all.",
    )
    parser.add_argument("--dataset_root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output_dir", default="outputs/lightrag_multihopqa_gnn")
    parser.add_argument("--env_file", default=os.getenv("LIGHTRAG_ENV_FILE"))
    parser.add_argument("--dataset_limit", type=int, default=None)
    parser.add_argument("--question_limit", type=int, default=None)
    parser.add_argument(
        "--separate_dataset_outputs",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("SEPARATE_DATASET_OUTPUTS", "true").lower()
        in {"1", "true", "yes", "y"},
        help=(
            "When multiple datasets are selected, write each dataset under "
            "<output_dir>/<dataset>/ so caches, workspaces, and summaries are separate."
        ),
    )
    parser.add_argument("--run_base", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run_gnn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--eval_only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Recompute evaluation from existing multihopqa_results*.json predictions. "
            "This skips LightRAG initialization, corpus insertion, QA, and GNN completion. "
            "--run_base/--run_gnn select which variants to re-evaluate."
        ),
    )
    parser.add_argument(
        "--eval_only_backup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create a .before_eval_only.bak copy before overwriting result JSON files.",
    )
    parser.add_argument(
        "--clone_base_to_gnn",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("CLONE_BASE_TO_GNN", "true").lower()
        in {"1", "true", "yes", "y"},
    )
    parser.add_argument(
        "--refresh_gnn_from_base",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("REFRESH_GNN_FROM_BASE", "true").lower()
        in {"1", "true", "yes", "y"},
    )
    parser.add_argument(
        "--query_mode",
        default="mix",
        choices=["local", "global", "hybrid", "mix", "naive"],
    )
    parser.add_argument(
        "--hipporag_qa_format",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("HIPPORAG_QA_FORMAT", "true").lower()
        in {"1", "true", "yes", "y"},
        help=(
            "Use LightRAG's KG/chunk QA instructions with HippoRAG-style parseable "
            "output: reason as Thought and end with `Answer: `."
        ),
    )
    parser.add_argument(
        "--extract_answer_after_answer_prefix",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("EXTRACT_ANSWER_AFTER_ANSWER_PREFIX", "true").lower()
        in {"1", "true", "yes", "y"},
        help=(
            "Store the prediction as the text after the first `Answer:` prefix, "
            "matching HippoRAG's QA post-processing."
        ),
    )
    parser.add_argument(
        "--qa_response_type",
        default=os.getenv("QA_RESPONSE_TYPE", DEFAULT_QA_RESPONSE_TYPE),
        help="Value passed to LightRAG QueryParam.response_type for QA generation.",
    )
    parser.add_argument(
        "--qa_user_prompt",
        default=os.getenv("QA_USER_PROMPT", DEFAULT_QA_USER_PROMPT),
        help=(
            "Additional LightRAG QA instruction passed through QueryParam.user_prompt. "
            "The default asks the model to output only a short answer for EM/F1."
        ),
    )
    parser.add_argument("--enable_rerank", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--random_seed",
        type=int,
        default=int(os.getenv("RANDOM_SEED", DEFAULT_RANDOM_SEED)),
    )
    parser.add_argument("--top_k", type=int, default=int(os.getenv("TOP_K", 1)))
    parser.add_argument("--chunk_top_k", type=int, default=int(os.getenv("CHUNK_TOP_K", 5)))
    parser.add_argument(
        "--max_total_tokens",
        type=int,
        default=int(os.getenv("MAX_TOTAL_TOKENS", 30000)),
    )
    parser.add_argument("--llm_timeout", type=int, default=int(os.getenv("LLM_TIMEOUT", 600)))
    parser.add_argument(
        "--llm_temperature",
        type=float,
        default=float(os.getenv("LLM_TEMPERATURE", 0)),
    )
    parser.add_argument(
        "--skip_failed_llm_calls",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("SKIP_FAILED_LLM_CALLS", "true").lower()
        in {"1", "true", "yes", "y"},
    )
    parser.add_argument(
        "--query_concurrency",
        type=int,
        default=int(os.getenv("QUERY_CONCURRENCY", 64)),
    )
    parser.add_argument(
        "--lightrag_llm_concurrency",
        type=int,
        default=int(os.getenv("MAX_ASYNC", 32)),
    )
    parser.add_argument(
        "--embedding_func_max_async",
        type=int,
        default=int(os.getenv("EMBEDDING_FUNC_MAX_ASYNC", 64)),
    )
    parser.add_argument(
        "--embedding_batch_num",
        type=int,
        default=int(os.getenv("EMBEDDING_BATCH_NUM", 64)),
    )
    parser.add_argument(
        "--max_parallel_insert",
        type=int,
        default=int(os.getenv("MAX_PARALLEL_INSERT", 64)),
    )

    parser.add_argument(
        "--gnn_subgraph_missing_score_threshold",
        type=float,
        default=float(os.getenv("GNN_SUBGRAPH_MISSING_SCORE_THRESHOLD", 0.5)),
        help="Minimum GNN missingness score required to send a subgraph to LLM completion.",
    )
    parser.add_argument(
        "--gnn_subgraph_fact_mask_ratio",
        type=float,
        default=float(os.getenv("GNN_SUBGRAPH_FACT_MASK_RATIO", 0.2)),
        help="Ratio of fact edges removed when constructing relation-missing training views.",
    )
    parser.add_argument(
        "--gnn_subgraph_entity_delete_ratio",
        type=float,
        default=float(os.getenv("GNN_SUBGRAPH_ENTITY_DELETE_RATIO", 0.08)),
        help="Ratio of eligible entity nodes removed when constructing entity-missing training views.",
    )
    parser.add_argument(
        "--gnn_llm_budget",
        type=float,
        default=float(os.getenv("GNN_SUBGRAPH_LLM_BUDGET", 0.1)),
    )

    parser.add_argument("--enable_judge", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--answer_judge_model",
        default=os.getenv("ANSWER_JUDGE_MODEL", "deepseek-v4-pro"),
    )
    parser.add_argument(
        "--answer_judge_base_url",
        default=os.getenv("ANSWER_JUDGE_BASE_URL", "https://api.deepseek.com"),
    )
    parser.add_argument(
        "--answer_judge_api_key_env",
        default=os.getenv("ANSWER_JUDGE_API_KEY_ENV", "DEEPSEEK_API_KEY"),
    )
    parser.add_argument(
        "--answer_judge_max_tokens",
        type=int,
        default=int(os.getenv("ANSWER_JUDGE_MAX_TOKENS", 5128)),
    )
    parser.add_argument(
        "--answer_judge_concurrency",
        type=int,
        default=int(os.getenv("ANSWER_JUDGE_CONCURRENCY", 32)),
    )
    parser.add_argument(
        "--answer_judge_max_retries",
        type=int,
        default=int(os.getenv("ANSWER_JUDGE_MAX_RETRIES", 3)),
    )
    parser.add_argument(
        "--answer_judge_retry_delay",
        type=float,
        default=float(os.getenv("ANSWER_JUDGE_RETRY_DELAY", 1.0)),
    )
    parser.add_argument(
        "--answer_judge_cache",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("ANSWER_JUDGE_CACHE", "true").lower()
        in {"1", "true", "yes", "y"},
    )
    parser.add_argument(
        "--answer_judge_cache_path",
        default=os.getenv("ANSWER_JUDGE_CACHE_PATH"),
    )
    return parser


if __name__ == "__main__":
    asyncio.run(main_async(build_arg_parser().parse_args()))
