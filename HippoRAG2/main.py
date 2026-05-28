import gc
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

from typing import Any, Dict, List
import json
import time

from src.hipporag.HippoRAG import HippoRAG
from src.hipporag.utils.misc_utils import compute_mdhash_id, string_to_bool
from src.hipporag.utils.config_utils import (
    BaseConfig,
    get_hipporag_hidden_triplet_profile,
    normalize_hipporag_hidden_triplet_profile_dataset_name,
)

import argparse

# os.environ["LOG_LEVEL"] = "DEBUG"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import logging

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.join(REPO_ROOT, "reproduce", "dataset")
DEFAULT_EMBEDDING_BASE_URL = "http://127.0.0.1:8000/v1/embeddings"

def get_gold_docs(samples: List, dataset_name: str = None) -> List:
    gold_docs = []
    for sample in samples:
        if 'supporting_facts' in sample:  # hotpotqa, 2wikimultihopqa
            gold_title = set([item[0] for item in sample['supporting_facts']])
            gold_title_and_content_list = [item for item in sample['context'] if item[0] in gold_title]
            if dataset_name.startswith('hotpotqa'):
                gold_doc = [item[0] + '\n' + ''.join(item[1]) for item in gold_title_and_content_list]
            else:
                gold_doc = [item[0] + '\n' + ' '.join(item[1]) for item in gold_title_and_content_list]
        elif 'contexts' in sample:
            gold_doc = [item['title'] + '\n' + item['text'] for item in sample['contexts'] if item['is_supporting']]
        else:
            assert 'paragraphs' in sample, "`paragraphs` should be in sample, or consider the setting not to evaluate retrieval"
            gold_paragraphs = []
            for item in sample['paragraphs']:
                if 'is_supporting' in item and item['is_supporting'] is False:
                    continue
                gold_paragraphs.append(item)
            gold_doc = [item['title'] + '\n' + (item['text'] if 'text' in item else item['paragraph_text']) for item in gold_paragraphs]

        gold_doc = list(set(gold_doc))
        gold_docs.append(gold_doc)
    return gold_docs


def get_gold_answers(samples):
    gold_answers = []
    for sample_idx in range(len(samples)):
        gold_ans = None
        sample = samples[sample_idx]

        if 'answer' in sample or 'gold_ans' in sample:
            gold_ans = sample['answer'] if 'answer' in sample else sample['gold_ans']
        elif 'reference' in sample:
            gold_ans = sample['reference']
        elif 'obj' in sample:
            gold_ans = set(
                [sample['obj']] + [sample['possible_answers']] + [sample['o_wiki_title']] + [sample['o_aliases']])
            gold_ans = list(gold_ans)
        assert gold_ans is not None
        if isinstance(gold_ans, str):
            gold_ans = [gold_ans]
        assert isinstance(gold_ans, list)
        gold_ans = set(gold_ans)
        if 'answer_aliases' in sample:
            gold_ans.update(sample['answer_aliases'])

        gold_answers.append(gold_ans)

    return gold_answers


def save_eval_results(
    save_dir: str,
    llm_name: str,
    retrieval_results: dict,
    qa_results: dict,
    retrieval_timing: dict,
    qa_timing: dict,
    use_gnn_suffix: bool = False,
    index_timing: dict = None,
):
    llm_label = llm_name.replace("/", "_")
    file_suffix = "_gnn" if use_gnn_suffix else ""
    retrieval_results_path = os.path.join(save_dir, f"retrieval_results_{llm_label}{file_suffix}.json")
    qa_results_path = os.path.join(save_dir, f"qa_results_{llm_label}{file_suffix}.json")

    os.makedirs(save_dir, exist_ok=True)

    retrieval_payload = dict(retrieval_results or {})
    retrieval_payload["_timing"] = retrieval_timing
    if index_timing is not None:
        retrieval_payload["_index_timing"] = index_timing

    qa_payload = dict(qa_results or {})
    qa_payload["_timing"] = qa_timing
    if index_timing is not None:
        qa_payload["_index_timing"] = index_timing

    with open(retrieval_results_path, "w") as f:
        json.dump(retrieval_payload, f, indent=2, default=str)

    with open(qa_results_path, "w") as f:
        json.dump(qa_payload, f, indent=2, default=str)

    logging.info(f"Retrieval results saved to {retrieval_results_path}")
    logging.info(f"QA results saved to {qa_results_path}")


def numeric_metrics_only(metrics: Dict[str, Any]) -> Dict[str, float]:
    return {
        key: round(float(value), 6)
        for key, value in (metrics or {}).items()
        if not str(key).startswith("_")
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    }


def print_metrics_summary(title: str, qa_results: Dict[str, Any], retrieval_results: Dict[str, Any] = None) -> None:
    summary = {"qa": numeric_metrics_only(qa_results)}
    retrieval_metrics = numeric_metrics_only(retrieval_results or {})
    if retrieval_metrics:
        summary["retrieval"] = retrieval_metrics
    print(f"\n{title}")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def build_dataset_save_dir(save_dir: str, dataset_name: str) -> str:
    if save_dir == 'outputs':
        return os.path.join(save_dir, dataset_name)
    return save_dir + '_' + dataset_name


def resolve_repo_path(path: str, base_dir: str = None) -> str:
    if os.path.isabs(path):
        return path

    candidates = []
    if base_dir is not None:
        candidates.append(os.path.join(base_dir, path))
    candidates.append(os.path.join(REPO_ROOT, path))
    candidates.append(os.path.join(DATASET_ROOT, path))

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


def make_json_safe(value):
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [make_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return make_json_safe(value.tolist())
    if hasattr(value, "item"):
        return make_json_safe(value.item())
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def summarize_llm_token_metadata(metadata_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary = {
        "num_calls": 0,
        "num_calls_with_token_metadata": 0,
        "num_error_calls": 0,
        "num_cache_hits": 0,
        "num_cache_misses_with_token_metadata": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "uncached_prompt_tokens": 0,
        "uncached_completion_tokens": 0,
        "uncached_total_tokens": 0,
        "max_prompt_tokens": 0,
        "max_completion_tokens": 0,
        "max_total_tokens": 0,
        "avg_total_tokens_per_call_with_metadata": 0.0,
    }
    for metadata in metadata_items or []:
        if not isinstance(metadata, dict):
            continue
        summary["num_calls"] += 1
        if metadata.get("error"):
            summary["num_error_calls"] += 1
        if metadata.get("cache_hit") is True:
            summary["num_cache_hits"] += 1

        has_token_metadata = (
            metadata.get("prompt_tokens") is not None
            or metadata.get("completion_tokens") is not None
        )
        if not has_token_metadata:
            continue

        prompt_tokens = int(metadata.get("prompt_tokens") or 0)
        completion_tokens = int(metadata.get("completion_tokens") or 0)
        total_tokens = prompt_tokens + completion_tokens
        summary["num_calls_with_token_metadata"] += 1
        summary["prompt_tokens"] += prompt_tokens
        summary["completion_tokens"] += completion_tokens
        summary["total_tokens"] += total_tokens
        if metadata.get("cache_hit") is not True:
            summary["num_cache_misses_with_token_metadata"] += 1
            summary["uncached_prompt_tokens"] += prompt_tokens
            summary["uncached_completion_tokens"] += completion_tokens
            summary["uncached_total_tokens"] += total_tokens
        summary["max_prompt_tokens"] = max(summary["max_prompt_tokens"], prompt_tokens)
        summary["max_completion_tokens"] = max(summary["max_completion_tokens"], completion_tokens)
        summary["max_total_tokens"] = max(summary["max_total_tokens"], total_tokens)

    denominator = summary["num_calls_with_token_metadata"]
    if denominator > 0:
        summary["avg_total_tokens_per_call_with_metadata"] = round(
            summary["total_tokens"] / denominator,
            4,
        )
    return summary


def get_index_cache_state(hipporag: HippoRAG, docs: List[str]) -> Dict[str, Any]:
    graph_cache_path = getattr(hipporag, "_graph_pickle_filename", None)
    base_graph_cache_path = getattr(hipporag, "_base_graph_pickle_filename", None)
    augmented_graph_cache_path = getattr(hipporag, "_augmented_graph_pickle_filename", None)
    graph_cache_type = "unknown"
    if graph_cache_path == base_graph_cache_path:
        graph_cache_type = "base"
    elif graph_cache_path == augmented_graph_cache_path:
        graph_cache_type = "hidden_triplet_augmented"

    missing_chunk_rows = hipporag.chunk_embedding_store.get_missing_string_hash_ids(docs)
    existing_chunk_ids = set(hipporag.chunk_embedding_store.get_all_id_to_rows().keys())
    input_chunk_ids = {compute_mdhash_id(doc, prefix="chunk-") for doc in docs}

    openie_cache_path = getattr(hipporag, "openie_results_path", None)
    openie_state: Dict[str, Any] = {
        "openie_cache_path": openie_cache_path,
        "openie_cache_existed_before_index": bool(openie_cache_path and os.path.isfile(openie_cache_path)),
    }
    try:
        _, openie_chunks_to_process = hipporag.load_existing_openie(input_chunk_ids)
        openie_state["num_openie_chunks_to_process_before_index"] = len(openie_chunks_to_process)
        openie_state["num_openie_chunks_cached_before_index"] = max(
            0,
            len(input_chunk_ids) - len(openie_chunks_to_process),
        )
    except Exception as exc:
        openie_state["openie_cache_probe_error"] = f"{type(exc).__name__}: {exc}"

    graph_cache_existed = bool(graph_cache_path and os.path.exists(graph_cache_path))
    graph_cache_will_load = graph_cache_existed and not hipporag.global_config.force_index_from_scratch
    state = {
        "graph_cache_path": graph_cache_path,
        "graph_cache_type": graph_cache_type,
        "graph_cache_existed_before_index": graph_cache_existed,
        "graph_cache_will_load_before_index": graph_cache_will_load,
        "force_index_from_scratch": bool(hipporag.global_config.force_index_from_scratch),
        "force_openie_from_scratch": bool(hipporag.global_config.force_openie_from_scratch),
        "num_docs_to_index": len(docs),
        "num_input_chunks_to_index": len(input_chunk_ids),
        "num_chunk_embeddings_existing_before_index": len(existing_chunk_ids),
        "num_input_chunk_embeddings_cached_before_index": max(0, len(input_chunk_ids) - len(missing_chunk_rows)),
        "num_new_chunk_embeddings_before_index": len(missing_chunk_rows),
    }
    state.update(openie_state)
    state["cache_likely_affected_timing"] = (
        graph_cache_will_load
        or state["num_new_chunk_embeddings_before_index"] < state["num_input_chunks_to_index"]
        or state.get("num_openie_chunks_cached_before_index", 0) > 0
    )
    state["fully_cached_before_index"] = (
        graph_cache_will_load
        and state["num_new_chunk_embeddings_before_index"] == 0
        and state.get("num_openie_chunks_to_process_before_index", 1) == 0
    )
    return state


def get_usable_gold_docs(samples: List, dataset_name: str):
    try:
        gold_docs = get_gold_docs(samples, dataset_name)
        if len(gold_docs) == 0 or not any(len(docs) > 0 for docs in gold_docs):
            return None
        return gold_docs
    except Exception:
        return None


def build_hipporag_config(
    args,
    dataset_name: str,
    save_dir: str,
    llm_base_url: str,
    llm_name: str,
    corpus_len: int,
    force_index_from_scratch: bool,
    force_openie_from_scratch: bool,
    enable_hidden_triplet_mining: bool,
    load_hidden_triplet_augmented_content: bool,
) -> BaseConfig:
    config_kwargs = dict(
        save_dir=save_dir,
        llm_base_url=llm_base_url,
        llm_name=llm_name,
        seed=args.seed,
        llm_concurrency=args.llm_concurrency,
        llm_cache_dir=args.llm_cache_dir,
        dataset=dataset_name,
        embedding_model_name=args.embedding_name,
        embedding_base_url=args.embedding_base_url,
        force_index_from_scratch=force_index_from_scratch,
        force_openie_from_scratch=force_openie_from_scratch,
        rerank_dspy_file_path="src/hipporag/prompts/dspy_prompts/filter_llama3.3-70B-Instruct.json",
        retrieval_top_k=200,
        linking_top_k=5,
        max_qa_steps=3,
        qa_top_k=5,
        graph_type="facts_and_sim_passage_node_unidirectional",
        embedding_batch_size=8,
        max_new_tokens=None,
        corpus_len=corpus_len,
        openie_mode=args.openie_mode,
        enable_hidden_triplet_mining=enable_hidden_triplet_mining,
        force_hidden_triplet_mining_from_scratch=string_to_bool(args.force_hidden_triplet_mining_from_scratch),
        load_hidden_triplet_augmented_content=load_hidden_triplet_augmented_content,
        enable_answer_judge=string_to_bool(args.enable_answer_judge),
        answer_judge_model=args.answer_judge_model,
        answer_judge_base_url=args.answer_judge_base_url,
        answer_judge_api_key_env=args.answer_judge_api_key_env,
        answer_judge_max_workers=args.answer_judge_max_workers,
        answer_judge_max_tokens=args.answer_judge_max_tokens,
        answer_judge_cache_dir=args.answer_judge_cache_dir,
    )
    profile_dataset = normalize_hipporag_hidden_triplet_profile_dataset_name(dataset_name)
    hidden_triplet_profile = get_hipporag_hidden_triplet_profile(dataset_name)
    config_kwargs.update(hidden_triplet_profile)
    if args.hidden_triplet_subgraph_llm_budget is not None:
        config_kwargs["hidden_triplet_subgraph_llm_budget"] = args.hidden_triplet_subgraph_llm_budget
    if args.hidden_triplet_subgraph_fact_mask_ratio is not None:
        config_kwargs["hidden_triplet_subgraph_fact_mask_ratio"] = args.hidden_triplet_subgraph_fact_mask_ratio
    if args.hidden_triplet_subgraph_entity_delete_ratio is not None:
        config_kwargs["hidden_triplet_subgraph_entity_delete_ratio"] = args.hidden_triplet_subgraph_entity_delete_ratio
    if args.hidden_triplet_subgraph_missing_score_threshold is not None:
        config_kwargs["hidden_triplet_subgraph_missing_score_threshold"] = args.hidden_triplet_subgraph_missing_score_threshold
    config_kwargs["hidden_triplet_reproduction_profile_name"] = f"hipporag:{profile_dataset}"
    config_kwargs["hidden_triplet_reproduction_profile"] = {
        "framework": "HippoRAG",
        "dataset": profile_dataset,
        "parameters": {
            key: config_kwargs[key]
            for key in sorted(hidden_triplet_profile)
            if key in config_kwargs
        },
    }
    return BaseConfig(**config_kwargs)


def run_hipporag_dataset(
    args,
    dataset_name: str,
    corpus_path: str,
    samples_path: str,
    save_dir: str,
    llm_base_url: str,
    llm_name: str,
    force_index_from_scratch: bool,
    force_openie_from_scratch: bool,
    enable_hidden_triplet_mining: bool,
    load_hidden_triplet_augmented_content: bool,
    book_metadata: Dict[str, Any] = None,
) -> Dict[str, Any]:
    with open(corpus_path, "r") as f:
        corpus = json.load(f)
    docs = [f"{doc['title']}\n{doc['text']}" for doc in corpus]

    with open(samples_path, "r") as f:
        samples = json.load(f)
    all_queries = [s['question'] for s in samples]
    gold_answers = get_gold_answers(samples)
    gold_docs = get_usable_gold_docs(samples, dataset_name)
    if gold_docs is not None:
        assert len(all_queries) == len(gold_docs) == len(gold_answers), "Length of queries, gold_docs, and gold_answers should be the same."
    else:
        assert len(all_queries) == len(gold_answers), "Length of queries and gold_answers should be the same."

    config = build_hipporag_config(
        args=args,
        dataset_name=dataset_name,
        save_dir=save_dir,
        llm_base_url=llm_base_url,
        llm_name=llm_name,
        corpus_len=len(corpus),
        force_index_from_scratch=force_index_from_scratch,
        force_openie_from_scratch=force_openie_from_scratch,
        enable_hidden_triplet_mining=enable_hidden_triplet_mining,
        load_hidden_triplet_augmented_content=load_hidden_triplet_augmented_content,
    )

    hipporag = HippoRAG(global_config=config, embedding_base_url=args.embedding_base_url)
    index_cache_state = get_index_cache_state(hipporag, docs)
    index_start_time = time.time()
    hipporag.index(docs)
    index_elapsed_sec = round(float(time.time() - index_start_time), 4)
    index_timing = {
        "index_total_time_sec": index_elapsed_sec,
        "index_wall_time_sec": index_elapsed_sec,
        "cache_state_before_index": index_cache_state,
        "timing_note": (
            "index_total_time_sec is the wall-clock time for this run. "
            "If cache_state_before_index.cache_likely_affected_timing is true, existing graph/OpenIE/embedding cache made this shorter than a fresh rebuild. "
            "If cache_state_before_index.fully_cached_before_index is true, treat it as cached-index loading time rather than full indexing time. "
            "Use --force_index_from_scratch true --force_openie_from_scratch true plus a new save_dir, or clear the old save_dir, when measuring fully fresh indexing cost."
        ),
    }
    logging.info("Total Index Time %.2fs", index_timing["index_total_time_sec"])

    query_solutions, _, qa_metadata, overall_retrieval_result, overall_qa_results = hipporag.rag_qa(
        queries=all_queries,
        gold_docs=gold_docs,
        gold_answers=gold_answers
    )
    retrieval_timing = {
        "retrieval_total_time_sec": round(float(hipporag.all_retrieval_time), 4),
        "recognition_memory_time_sec": round(float(hipporag.rerank_time), 4),
        "ppr_time_sec": round(float(hipporag.ppr_time), 4),
        "retrieval_misc_time_sec": round(float(hipporag.all_retrieval_time - (hipporag.rerank_time + hipporag.ppr_time)), 4),
    }
    qa_timing = {
        "qa_total_time_sec": round(float(hipporag.all_qa_time), 4),
    }
    book_metadata = book_metadata or {}
    query_results = [make_json_safe(solution.to_dict()) for solution in query_solutions]
    for query_result, metadata in zip(query_results, qa_metadata):
        query_result["qa_metadata"] = make_json_safe(metadata)

    result = {
        "dataset": dataset_name,
        "book_dataset": book_metadata.get("dataset", dataset_name),
        "save_dir": save_dir,
        "corpus_path": corpus_path,
        "samples_path": samples_path,
        "num_chunks": len(corpus),
        "num_queries": len(all_queries),
        "has_gold_docs": gold_docs is not None,
        "retrieval_results": overall_retrieval_result,
        "qa_results": overall_qa_results,
        "index_timing": index_timing,
        "retrieval_timing": retrieval_timing,
        "qa_timing": qa_timing,
        "qa_token_usage": summarize_llm_token_metadata(qa_metadata),
        "qa_metadata": make_json_safe(qa_metadata),
        "hidden_triplet_reproduction_profile": make_json_safe(
            getattr(config, "hidden_triplet_reproduction_profile", {})
        ),
        "book_metadata": book_metadata,
        "query_results": query_results,
    }

    del hipporag
    gc.collect()
    return result


def load_literaryqa_manifest(dataset_name: str):
    manifest_candidates = [
        os.path.join(DATASET_ROOT, dataset_name, "manifest.json"),
        os.path.join(DATASET_ROOT, f"{dataset_name}_manifest.json"),
    ]
    for manifest_path in manifest_candidates:
        if os.path.exists(manifest_path):
            with open(manifest_path, "r") as f:
                payload = json.load(f)
            if isinstance(payload, list):
                return manifest_path, {"format": "legacy_list_manifest", "dataset": dataset_name, "books": payload}
            return manifest_path, payload
    return None, None


def aggregate_numeric_metrics(book_runs: List[Dict[str, Any]], result_key: str) -> Dict[str, float]:
    weighted_totals: Dict[str, float] = {}
    total_weights: Dict[str, int] = {}

    for book_run in book_runs:
        result = book_run.get(result_key) or {}
        weight = int(book_run.get("num_queries", 0))
        if weight <= 0:
            continue

        for key, value in result.items():
            if key.startswith("_") or not isinstance(value, (int, float)):
                continue
            weighted_totals[key] = weighted_totals.get(key, 0.0) + float(value) * weight
            total_weights[key] = total_weights.get(key, 0) + weight

    return {
        key: round(weighted_totals[key] / total_weights[key], 4)
        for key in sorted(weighted_totals)
        if total_weights.get(key, 0) > 0
    }


def sum_timings(book_runs: List[Dict[str, Any]], timing_key: str) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    for book_run in book_runs:
        for key, value in (book_run.get(timing_key) or {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                totals[key] = totals.get(key, 0.0) + float(value)
    return {key: round(value, 4) for key, value in totals.items()}


def summarize_index_cache_states(book_runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    states = [
        (book_run.get("index_timing") or {}).get("cache_state_before_index") or {}
        for book_run in book_runs
    ]
    return {
        "num_books": len(states),
        "num_books_cache_likely_affected_timing": sum(
            1 for state in states if state.get("cache_likely_affected_timing")
        ),
        "num_books_fully_cached_before_index": sum(
            1 for state in states if state.get("fully_cached_before_index")
        ),
        "num_books_graph_cache_will_load_before_index": sum(
            1 for state in states if state.get("graph_cache_will_load_before_index")
        ),
        "num_new_chunk_embeddings_before_index": sum(
            int(state.get("num_new_chunk_embeddings_before_index", 0) or 0)
            for state in states
        ),
        "num_openie_chunks_to_process_before_index": sum(
            int(state.get("num_openie_chunks_to_process_before_index", 0) or 0)
            for state in states
        ),
    }


def summarize_book_runs(book_runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summaries = []
    for book_run in book_runs:
        summaries.append(
            {
                "dataset": book_run["dataset"],
                "book_dataset": book_run["book_dataset"],
                "save_dir": book_run["save_dir"],
                "num_chunks": book_run["num_chunks"],
                "num_queries": book_run["num_queries"],
                "has_gold_docs": book_run["has_gold_docs"],
                "qa_results": book_run.get("qa_results"),
                "retrieval_results": book_run.get("retrieval_results"),
                "index_timing": book_run.get("index_timing"),
                "retrieval_timing": book_run.get("retrieval_timing"),
                "qa_timing": book_run.get("qa_timing"),
                "book_metadata": book_run.get("book_metadata", {}),
            }
        )
    return summaries


def run_literaryqa_multibook(
    args,
    dataset_name: str,
    save_dir: str,
    llm_base_url: str,
    llm_name: str,
    force_index_from_scratch: bool,
    force_openie_from_scratch: bool,
    enable_hidden_triplet_mining: bool,
    load_hidden_triplet_augmented_content: bool,
) -> bool:
    manifest_path, manifest = load_literaryqa_manifest(dataset_name)
    if manifest is None:
        return False

    books = manifest.get("books", [])
    if not books:
        raise ValueError(f"No LiteraryQA books found in manifest: {manifest_path}")
    num_books_in_manifest = len(books)

    book_limit = getattr(args, "literaryqa_book_limit", None)
    if book_limit is not None:
        if book_limit <= 0:
            raise ValueError("--literaryqa_book_limit must be a positive integer when set.")
        books = books[:book_limit]
    if not books:
        raise ValueError(f"No LiteraryQA books selected from manifest: {manifest_path}")

    os.makedirs(save_dir, exist_ok=True)
    book_cache_root = os.path.join(save_dir, "books")
    os.makedirs(book_cache_root, exist_ok=True)

    logging.info("Running LiteraryQA multi-book dataset from %s", manifest_path)
    logging.info("Book cache root: %s", book_cache_root)
    if book_limit is not None:
        logging.info(
            "Processing the first %s LiteraryQA books from %s manifest books.",
            len(books),
            num_books_in_manifest,
        )

    book_runs = []
    manifest_dir = os.path.dirname(manifest_path)
    llm_label = llm_name.replace("/", "_")
    file_suffix = "_gnn" if enable_hidden_triplet_mining or load_hidden_triplet_augmented_content else ""
    for book_idx, book in enumerate(books):
        book_dataset = book.get("dataset") or f"{dataset_name}_book_{book_idx:04d}"
        book_output_subdir = book.get("book_output_subdir") or book_dataset
        book_output_subdir = book_output_subdir.replace("/", "_").replace("\\", "_")
        book_save_dir = os.path.join(book_cache_root, book_output_subdir)
        corpus_path = resolve_repo_path(book["corpus_path"], base_dir=manifest_dir)
        samples_path = resolve_repo_path(book["samples_path"], base_dir=manifest_dir)

        logging.info(
            "Running LiteraryQA book %s/%s: %s",
            book_idx + 1,
            len(books),
            book.get("title") or book_dataset,
        )
        book_run = run_hipporag_dataset(
            args=args,
            dataset_name=dataset_name,
            corpus_path=corpus_path,
            samples_path=samples_path,
            save_dir=book_save_dir,
            llm_base_url=llm_base_url,
            llm_name=llm_name,
            force_index_from_scratch=force_index_from_scratch,
            force_openie_from_scratch=force_openie_from_scratch,
            enable_hidden_triplet_mining=enable_hidden_triplet_mining,
            load_hidden_triplet_augmented_content=load_hidden_triplet_augmented_content,
            book_metadata=book,
        )
        book_runs.append(book_run)

        save_eval_results(
            book_save_dir,
            llm_name,
            book_run["retrieval_results"],
            book_run["qa_results"],
            book_run["retrieval_timing"],
            book_run["qa_timing"],
            index_timing=book_run["index_timing"],
            use_gnn_suffix=enable_hidden_triplet_mining | load_hidden_triplet_augmented_content,
        )
        book_query_results_path = os.path.join(book_save_dir, f"qa_query_results_{llm_label}{file_suffix}.json")
        with open(book_query_results_path, "w") as f:
            json.dump(
                {
                    "dataset": dataset_name,
                    "book_dataset": book_run["book_dataset"],
                    "book_metadata": book_run.get("book_metadata", {}),
                    "num_chunks": book_run["num_chunks"],
                    "num_queries": book_run["num_queries"],
                    "has_gold_docs": book_run["has_gold_docs"],
                    "index_timing": book_run.get("index_timing"),
                    "retrieval_timing": book_run.get("retrieval_timing"),
                    "qa_timing": book_run.get("qa_timing"),
                    "qa_token_usage": book_run.get("qa_token_usage"),
                    "retrieval_results": book_run.get("retrieval_results"),
                    "qa_results": book_run.get("qa_results"),
                    "query_results": book_run["query_results"],
                },
                f,
                indent=2,
                default=str,
            )
        logging.info("LiteraryQA book query-level QA results saved to %s", book_query_results_path)

    book_summaries = summarize_book_runs(book_runs)
    aggregate_index_timing = sum_timings(book_runs, "index_timing")
    aggregate_index_timing["_cache_summary"] = summarize_index_cache_states(book_runs)
    literaryqa_summary = {
        "dataset": dataset_name,
        "manifest_path": manifest_path,
        "manifest_format": manifest.get("format"),
        "book_cache_root": book_cache_root,
        "literaryqa_book_limit": book_limit,
        "num_books_in_manifest": num_books_in_manifest,
        "num_books": len(book_runs),
        "num_total_chunks": sum(book_run["num_chunks"] for book_run in book_runs),
        "num_total_queries": sum(book_run["num_queries"] for book_run in book_runs),
        "num_books_with_gold_docs": sum(1 for book_run in book_runs if book_run["has_gold_docs"]),
        "index_timing": aggregate_index_timing,
        "books": book_summaries,
    }

    aggregate_retrieval_results = aggregate_numeric_metrics(book_runs, "retrieval_results")
    aggregate_retrieval_results["_literaryqa"] = literaryqa_summary
    aggregate_qa_results = aggregate_numeric_metrics(book_runs, "qa_results")
    aggregate_qa_results["_literaryqa"] = literaryqa_summary

    save_eval_results(
        save_dir,
        llm_name,
        aggregate_retrieval_results,
        aggregate_qa_results,
        sum_timings(book_runs, "retrieval_timing"),
        sum_timings(book_runs, "qa_timing"),
        index_timing=aggregate_index_timing,
        use_gnn_suffix=enable_hidden_triplet_mining | load_hidden_triplet_augmented_content,
    )

    query_results_path = os.path.join(save_dir, f"qa_query_results_{llm_label}{file_suffix}.json")
    with open(query_results_path, "w") as f:
        json.dump(
            {
                "dataset": dataset_name,
                "manifest_path": manifest_path,
                "num_books": len(book_runs),
                "num_total_queries": literaryqa_summary["num_total_queries"],
                "index_timing": aggregate_index_timing,
                "books": [
                    {
                        "dataset": book_run["dataset"],
                        "book_dataset": book_run["book_dataset"],
                        "book_metadata": book_run.get("book_metadata", {}),
                        "index_timing": book_run.get("index_timing"),
                        "qa_token_usage": book_run.get("qa_token_usage"),
                        "query_results": book_run["query_results"],
                    }
                    for book_run in book_runs
                ],
            },
            f,
            indent=2,
            default=str,
        )
    logging.info("LiteraryQA query-level QA results saved to %s", query_results_path)
    print_metrics_summary(
        "LiteraryQA aggregate metrics",
        qa_results=aggregate_qa_results,
        retrieval_results=aggregate_retrieval_results,
    )
    return True


def main():
    parser = argparse.ArgumentParser(description="HippoRAG retrieval and QA")
    parser.add_argument('--dataset', type=str, default='musique', help='Dataset name')
    parser.add_argument('--llm_base_url', type=str, default='https://api.openai.com/v1', help='LLM base URL')
    parser.add_argument('--llm_name', type=str, default='gpt-4o-mini', help='LLM name')
    parser.add_argument('--embedding_name', type=str, default='nvidia/NV-Embed-v2', help='embedding model name')
    parser.add_argument('--embedding_base_url', type=str, default=DEFAULT_EMBEDDING_BASE_URL,
                        help='OpenAI-compatible embedding endpoint. This URL is ignored for cache matching; the embedding model name controls embedding-cache identity.')
    parser.add_argument('--seed', type=int, default=42, help='Global random seed used for GNN training, subgraph mining, QA generation, and answer judging.')
    parser.add_argument('--llm_concurrency', type=int, default=64,
                        help='Unified concurrency limit for all non-answer-judge LLM calls in the main HippoRAG flow, including OpenIE, QA generation, and hidden-triplet completion.')
    parser.add_argument('--llm_cache_dir', type=str, default=None,
                        help='Optional shared directory for non-answer-judge LLM cache. Defaults to <save_dir>/llm_cache.')
    parser.add_argument('--force_index_from_scratch', type=str, default='false',
                        help='If set to True, will ignore all existing storage files and graph data and will rebuild from scratch.')
    parser.add_argument('--force_openie_from_scratch', type=str, default='false', help='If set to False, will try to first reuse openie results for the corpus if they exist.')
    parser.add_argument('--openie_mode', choices=['online', 'offline'], default='online',
                        help="OpenIE mode, offline denotes using VLLM offline batch mode for indexing, while online denotes")
    parser.add_argument('--enable_hidden_triplet_mining', type=str, default='false',
                        help='If set to True, run GNN-based hidden triplet mining after indexing.')
    parser.add_argument('--force_hidden_triplet_mining_from_scratch', type=str, default='false',
                        help='If set to True, ignore existing hidden-triplet mining artifacts and rebuild them from the base graph.')
    parser.add_argument('--load_hidden_triplet_augmented_content', type=str, default='false',
                        help='If set to True, load hidden-triplet augmented graph and fact/entity stores for retrieval and QA.')
    parser.add_argument(
        '--hidden_triplet_subgraph_llm_budget',
        type=float,
        default=None,
        help='If set to an integer >= 1, use it directly as the LLM subgraph budget. If set to a fraction in (0, 1), resolve it as int(value * num_chunks * 2).',
    )
    parser.add_argument(
        '--hidden_triplet_subgraph_fact_mask_ratio',
        type=float,
        default=None,
        help='Ratio of fact edges removed when constructing relation-missing subgraph views.',
    )
    parser.add_argument(
        '--hidden_triplet_subgraph_entity_delete_ratio',
        type=float,
        default=None,
        help='Ratio of eligible entity nodes removed when constructing entity-missing subgraph views.',
    )
    parser.add_argument(
        '--hidden_triplet_subgraph_missing_score_threshold',
        type=float,
        default=None,
        help='Minimum GNN missingness score required before a sampled subgraph can be sent to the LLM.',
    )
    parser.add_argument('--enable_answer_judge', type=str, default='false',
                        help='If set to True, evaluate answer consistency with an LLM judge.')
    parser.add_argument('--answer_judge_model', type=str, default='deepseek-v4-pro',
                        help='OpenAI-compatible model name for the LLM answer judge.')
    parser.add_argument('--answer_judge_base_url', type=str, default='https://api.deepseek.com',
                        help='OpenAI-compatible base URL for the LLM answer judge.')
    parser.add_argument('--answer_judge_api_key_env', type=str, default='DEEPSEEK_API_KEY',
                        help='Environment variable name containing the LLM answer judge API key.')
    parser.add_argument('--answer_judge_max_workers', type=int, default=100,
                        help='Maximum concurrent answer-judge API requests.')
    parser.add_argument('--answer_judge_max_tokens', type=int, default=5128,
                        help='Maximum output tokens for each answer-judge request.')
    parser.add_argument('--answer_judge_cache_dir', type=str, default=None,
                        help='Optional shared directory for answer-judge LLM cache. Defaults to <save_dir>/llm_cache/answer_judge.')
    parser.add_argument('--save_dir', type=str, default='outputs', help='Save directory')
    parser.add_argument(
        '--literaryqa_book_limit',
        type=int,
        default=None,
        help='For --dataset literaryqa, process only the first N books in the manifest. For example, 50 processes the first 50 books.',
    )
    args = parser.parse_args()

    dataset_name = args.dataset
    save_dir = build_dataset_save_dir(args.save_dir, dataset_name)
    llm_base_url = args.llm_base_url
    llm_name = args.llm_name

    force_index_from_scratch = string_to_bool(args.force_index_from_scratch)
    force_openie_from_scratch = string_to_bool(args.force_openie_from_scratch)
    enable_hidden_triplet_mining = string_to_bool(args.enable_hidden_triplet_mining)
    load_hidden_triplet_augmented_content = string_to_bool(args.load_hidden_triplet_augmented_content)
    user_requested_augmented_load = load_hidden_triplet_augmented_content

    if enable_hidden_triplet_mining:
        load_hidden_triplet_augmented_content = True

    logging.basicConfig(level=logging.INFO)
    if enable_hidden_triplet_mining and not user_requested_augmented_load:
        logging.info(
            "Enabled hidden-triplet augmented content loading automatically because hidden triplet mining is enabled."
        )

    if run_literaryqa_multibook(
        args=args,
        dataset_name=dataset_name,
        save_dir=save_dir,
        llm_base_url=llm_base_url,
        llm_name=llm_name,
        force_index_from_scratch=force_index_from_scratch,
        force_openie_from_scratch=force_openie_from_scratch,
        enable_hidden_triplet_mining=enable_hidden_triplet_mining,
        load_hidden_triplet_augmented_content=load_hidden_triplet_augmented_content,
    ):
        return

    corpus_path = os.path.join(DATASET_ROOT, f"{dataset_name}_corpus.json")
    samples_path = os.path.join(DATASET_ROOT, f"{dataset_name}.json")
    run_result = run_hipporag_dataset(
        args=args,
        dataset_name=dataset_name,
        corpus_path=corpus_path,
        samples_path=samples_path,
        save_dir=save_dir,
        llm_base_url=llm_base_url,
        llm_name=llm_name,
        force_index_from_scratch=force_index_from_scratch,
        force_openie_from_scratch=force_openie_from_scratch,
        enable_hidden_triplet_mining=enable_hidden_triplet_mining,
        load_hidden_triplet_augmented_content=load_hidden_triplet_augmented_content,
    )
    save_eval_results(
        save_dir,
        llm_name,
        run_result["retrieval_results"],
        run_result["qa_results"],
        run_result["retrieval_timing"],
        run_result["qa_timing"],
        index_timing=run_result["index_timing"],
        use_gnn_suffix=enable_hidden_triplet_mining | load_hidden_triplet_augmented_content,
    )
    llm_label = llm_name.replace("/", "_")
    file_suffix = "_gnn" if (enable_hidden_triplet_mining | load_hidden_triplet_augmented_content) else ""
    query_results_path = os.path.join(save_dir, f"qa_query_results_{llm_label}{file_suffix}.json")
    with open(query_results_path, "w") as f:
        json.dump(
            {
                "dataset": dataset_name,
                "num_chunks": run_result["num_chunks"],
                "num_queries": run_result["num_queries"],
                "has_gold_docs": run_result["has_gold_docs"],
                "index_timing": run_result.get("index_timing"),
                "retrieval_timing": run_result.get("retrieval_timing"),
                "qa_timing": run_result.get("qa_timing"),
                "qa_token_usage": run_result.get("qa_token_usage"),
                "retrieval_results": run_result.get("retrieval_results"),
                "qa_results": run_result.get("qa_results"),
                "query_results": run_result["query_results"],
            },
            f,
            indent=2,
            default=str,
        )
    logging.info("Query-level QA results saved to %s", query_results_path)
    print_metrics_summary(
        f"{dataset_name} aggregate metrics",
        qa_results=run_result.get("qa_results"),
        retrieval_results=run_result.get("retrieval_results"),
    )

if __name__ == "__main__":
    main()
