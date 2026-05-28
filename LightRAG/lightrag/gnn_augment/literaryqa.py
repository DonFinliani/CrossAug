from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lightrag.base import DocStatus
from lightrag.utils import logger, sanitize_text_for_encoding


@dataclass(slots=True)
class LiteraryQABook:
    dataset: str
    title: str
    document_id: str
    corpus_path: Path
    samples_path: Path
    book_output_subdir: str
    metadata: dict[str, Any]


def _resolve_relative_path(manifest_path: Path, value: str) -> Path:
    raw = Path(value)
    if raw.is_absolute() and raw.exists():
        return raw
    candidates = [
        Path.cwd() / raw,
        manifest_path.parent / raw,
        manifest_path.parent.parent / raw,
        manifest_path.parent.parent.parent / raw,
        manifest_path.parent.parent.parent.parent / raw,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def load_literaryqa_books(
    manifest_path: str | Path,
    book_limit: int | None = None,
) -> list[LiteraryQABook]:
    manifest = Path(manifest_path).expanduser().resolve()
    with open(manifest, "r", encoding="utf-8") as f:
        payload = json.load(f)

    books = []
    for row in payload.get("books", []):
        books.append(
            LiteraryQABook(
                dataset=row["dataset"],
                title=row.get("title", row["dataset"]),
                document_id=row.get("document_id", row["dataset"]),
                corpus_path=_resolve_relative_path(manifest, row["corpus_path"]),
                samples_path=_resolve_relative_path(manifest, row["samples_path"]),
                book_output_subdir=row.get("book_output_subdir", row["dataset"]),
                metadata=dict(row),
            )
        )
        if book_limit is not None and len(books) >= book_limit:
            break
    return books


def load_literaryqa_corpus(book: LiteraryQABook) -> list[dict[str, Any]]:
    with open(book.corpus_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_literaryqa_samples(book: LiteraryQABook) -> list[dict[str, Any]]:
    with open(book.samples_path, "r", encoding="utf-8") as f:
        return json.load(f)


def gold_answers_for_sample(sample: dict[str, Any]) -> list[str]:
    answers = []
    if sample.get("answer"):
        answers.append(str(sample["answer"]))
    for alias in sample.get("answer_aliases") or []:
        if alias:
            answers.append(str(alias))
    return list(dict.fromkeys(answers))


def _new_pipeline_status() -> tuple[dict[str, Any], asyncio.Lock]:
    return (
        {
            "busy": True,
            "cancellation_requested": False,
            "latest_message": "",
            "history_messages": [],
        },
        asyncio.Lock(),
    )


async def _graph_counts(rag: Any) -> tuple[int, int]:
    try:
        nodes = await rag.chunk_entity_relation_graph.get_all_nodes()
        edges = await rag.chunk_entity_relation_graph.get_all_edges()
        return len(nodes or []), len(edges or [])
    except Exception as exc:
        logger.warning("Failed to inspect current LightRAG graph counts: %s", exc)
        return 0, 0


async def _run_lightrag_entity_relation_pipeline(
    rag: Any,
    book: LiteraryQABook,
    doc_key: str,
    full_text: str,
    chunks: dict[str, Any],
    new_docs: dict[str, Any],
    persist_chunks: bool = True,
) -> dict[str, Any]:
    """Run the same entity/relation extraction and merge path as LightRAG.

    LiteraryQA is already chunked, so this helper only replaces the chunking
    stage. The graph construction itself uses LightRAG's native
    extract_entities + merge_nodes_and_edges implementation.
    """

    pipeline_status, pipeline_status_lock = _new_pipeline_status()
    processing_start_time = int(time.time())
    created_at = datetime.now(timezone.utc).isoformat()

    if rag.doc_status is not None:
        await rag.doc_status.upsert(
            {
                doc_key: {
                    "status": DocStatus.PROCESSING,
                    "chunks_count": len(chunks),
                    "chunks_list": list(chunks.keys()),
                    "content_summary": full_text[:100],
                    "content_length": len(full_text),
                    "created_at": created_at,
                    "updated_at": created_at,
                    "file_path": book.title,
                    "metadata": {"processing_start_time": processing_start_time},
                }
            }
        )

    try:
        from lightrag.operate import merge_nodes_and_edges

        first_stage_tasks = []
        if chunks and persist_chunks:
            first_stage_tasks.extend(
                [
                    rag.chunks_vdb.upsert(chunks),
                    rag.text_chunks.upsert(chunks),
                ]
            )
        if new_docs:
            first_stage_tasks.append(rag.full_docs.upsert(new_docs))
        if first_stage_tasks:
            await asyncio.gather(*first_stage_tasks)

        chunk_results = []
        if chunks:
            chunk_results = await rag._process_extract_entities(
                chunks,
                pipeline_status=pipeline_status,
                pipeline_status_lock=pipeline_status_lock,
            )
            await merge_nodes_and_edges(
                chunk_results=chunk_results,
                knowledge_graph_inst=rag.chunk_entity_relation_graph,
                entity_vdb=rag.entities_vdb,
                relationships_vdb=rag.relationships_vdb,
                global_config=asdict(rag),
                full_entities_storage=rag.full_entities,
                full_relations_storage=rag.full_relations,
                doc_id=doc_key,
                pipeline_status=pipeline_status,
                pipeline_status_lock=pipeline_status_lock,
                llm_response_cache=rag.llm_response_cache,
                entity_chunks_storage=rag.entity_chunks,
                relation_chunks_storage=rag.relation_chunks,
                current_file_number=1,
                total_files=1,
                file_path=book.title,
            )

        processing_end_time = int(time.time())
        if rag.doc_status is not None:
            await rag.doc_status.upsert(
                {
                    doc_key: {
                        "status": DocStatus.PROCESSED,
                        "chunks_count": len(chunks),
                        "chunks_list": list(chunks.keys()),
                        "content_summary": full_text[:100],
                        "content_length": len(full_text),
                        "created_at": created_at,
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                        "file_path": book.title,
                        "metadata": {
                            "processing_start_time": processing_start_time,
                            "processing_end_time": processing_end_time,
                        },
                    }
                }
            )

        await rag._insert_done(pipeline_status, pipeline_status_lock)
        graph_nodes, graph_edges = await _graph_counts(rag)
        return {
            "extraction_chunks": len(chunks),
            "chunk_result_count": len(chunk_results),
            "skipped_extraction_chunks": max(0, len(chunks) - len(chunk_results)),
            "graph_nodes": graph_nodes,
            "graph_edges": graph_edges,
        }
    except Exception as exc:
        if rag.doc_status is not None:
            await rag.doc_status.upsert(
                {
                    doc_key: {
                        "status": DocStatus.FAILED,
                        "error_msg": str(exc),
                        "chunks_count": len(chunks),
                        "chunks_list": list(chunks.keys()),
                        "content_summary": full_text[:100],
                        "content_length": len(full_text),
                        "created_at": created_at,
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                        "file_path": book.title,
                        "metadata": {
                            "processing_start_time": processing_start_time,
                            "processing_end_time": int(time.time()),
                        },
                    }
                }
            )
        if rag.llm_response_cache is not None:
            try:
                await rag.llm_response_cache.index_done_callback()
            except Exception as persist_exc:
                logger.error("Failed to persist LLM cache after failure: %s", persist_exc)
        raise
    finally:
        pipeline_status["busy"] = False


async def insert_literaryqa_book(
    rag: Any,
    book: LiteraryQABook,
    corpus: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Insert LiteraryQA pre-chunked corpus into LightRAG without re-chunking.

    This preserves the dataset chunk ids as LightRAG chunk ids, which is useful
    for GNN evidence provenance and citation audits.
    """

    corpus = corpus if corpus is not None else load_literaryqa_corpus(book)
    if not corpus:
        return {"status": "skipped", "reason": "empty_corpus", "chunks": 0}

    sorted_corpus = sorted(corpus, key=lambda row: int(row.get("chunk_index", 0)))
    full_text = "\n\n".join(str(row.get("text", "")) for row in sorted_corpus)
    full_text = sanitize_text_for_encoding(full_text)
    doc_key = book.document_id

    new_docs = {
        doc_key: {
            "content": full_text,
            "file_path": book.title,
        }
    }
    add_doc_keys = await rag.full_docs.filter_keys({doc_key})
    new_docs = {key: value for key, value in new_docs.items() if key in add_doc_keys}

    all_chunks: dict[str, Any] = {}
    for index, row in enumerate(sorted_corpus):
        chunk_text = sanitize_text_for_encoding(str(row.get("text", "")))
        if not chunk_text:
            continue
        chunk_key = str(row.get("chunk_id") or f"{book.dataset}__chunk_{index:04d}")
        title = row.get("title") or f"{book.title} [chunk {index + 1:04d}]"
        file_path = f"{book.title} / {title}"
        all_chunks[chunk_key] = {
            "content": chunk_text,
            "full_doc_id": doc_key,
            "tokens": len(rag.tokenizer.encode(chunk_text)),
            "chunk_order_index": index,
            "file_path": file_path,
            "title": title,
            "book_title": book.title,
            "document_id": book.document_id,
            "literaryqa_chunk_id": chunk_key,
        }

    add_chunk_keys = await rag.text_chunks.filter_keys(set(all_chunks.keys()))
    inserting_chunks = {
        key: value for key, value in all_chunks.items() if key in add_chunk_keys
    }
    if not inserting_chunks and not new_docs:
        graph_nodes, graph_edges = await _graph_counts(rag)
        if graph_nodes == 0 and graph_edges == 0 and all_chunks:
            logger.warning(
                "LiteraryQA book %s has chunks but no graph; rebuilding LightRAG graph",
                book.dataset,
            )
            graph_report = await _run_lightrag_entity_relation_pipeline(
                rag,
                book,
                doc_key,
                full_text,
                all_chunks,
                {},
                persist_chunks=False,
            )
            return {
                "status": "rebuilt_graph",
                "book": book.dataset,
                "chunks": len(all_chunks),
                "doc_inserted": False,
                **graph_report,
            }

        logger.warning("LiteraryQA book already indexed: %s", book.dataset)
        return {
            "status": "skipped",
            "reason": "already_indexed",
            "chunks": 0,
            "graph_nodes": graph_nodes,
            "graph_edges": graph_edges,
        }

    graph_report = await _run_lightrag_entity_relation_pipeline(
        rag,
        book,
        doc_key,
        full_text,
        inserting_chunks,
        new_docs,
    )
    return {
        "status": "inserted",
        "book": book.dataset,
        "chunks": len(inserting_chunks),
        "doc_inserted": bool(new_docs),
        **graph_report,
    }
