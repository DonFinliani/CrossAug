from __future__ import annotations

import json
import os
from collections import Counter
from typing import Any

import numpy as np

from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.utils import logger

from .config import SubgraphCompletionConfig
from .schemas import (
    CHUNK_NODE,
    ENTITY_NODE,
    LightRAGGraphSnapshot,
    RelationEdge,
    chunk_node_id,
)


def _split_source_ids(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item).strip()]
    text = str(value)
    if not text:
        return []
    return [part for part in text.split(GRAPH_FIELD_SEP) if part]


def _compact_text(text: str, max_chars: int) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


class LightRAGGraphAdapter:
    """Convert LightRAG storages into a temporary heterogeneous graph."""

    @staticmethod
    async def from_rag(
        rag: Any,
        config: SubgraphCompletionConfig,
    ) -> LightRAGGraphSnapshot:
        graph = rag.chunk_entity_relation_graph
        raw_nodes = await graph.get_all_nodes()
        raw_edges = await graph.get_all_edges()

        entity_data: dict[str, dict[str, Any]] = {}
        entity_source_ids: dict[str, list[str]] = {}
        node_ids: list[str] = []
        node_types: dict[str, str] = {}
        node_texts: dict[str, str] = {}

        for node in raw_nodes:
            entity_id = str(node.get("id") or node.get("entity_id") or "").strip()
            if not entity_id:
                continue
            entity_data[entity_id] = dict(node)
            entity_source_ids[entity_id] = _split_source_ids(node.get("source_id"))
            node_ids.append(entity_id)
            node_types[entity_id] = ENTITY_NODE
            node_texts[entity_id] = _compact_text(
                entity_id,
                config.max_embedding_text_chars,
            )

        relations: list[RelationEdge] = []
        all_chunk_ids: list[str] = []
        for source_ids in entity_source_ids.values():
            all_chunk_ids.extend(source_ids)

        for edge in raw_edges:
            src = str(edge.get("source") or edge.get("src_id") or "").strip()
            tgt = str(edge.get("target") or edge.get("tgt_id") or "").strip()
            if not src or not tgt or src == tgt:
                continue
            source_ids = _split_source_ids(edge.get("source_id"))
            all_chunk_ids.extend(source_ids)
            try:
                weight = float(edge.get("weight", 1.0) or 1.0)
            except (TypeError, ValueError):
                weight = 1.0
            relations.append(
                RelationEdge(
                    src=src,
                    tgt=tgt,
                    description=str(edge.get("description", "") or ""),
                    keywords=str(edge.get("keywords", "") or ""),
                    source_ids=source_ids,
                    file_path=str(edge.get("file_path", "unknown_source") or "unknown_source"),
                    weight=weight,
                    metadata=dict(edge),
                )
            )

        unique_chunk_ids = list(dict.fromkeys(all_chunk_ids))
        chunk_rows = await rag.text_chunks.get_by_ids(unique_chunk_ids) if unique_chunk_ids else []
        chunks: dict[str, dict[str, Any]] = {}
        for chunk_id, chunk_data in zip(unique_chunk_ids, chunk_rows):
            if not chunk_data or "content" not in chunk_data:
                continue
            chunks[chunk_id] = dict(chunk_data)
            node_id = chunk_node_id(chunk_id)
            node_ids.append(node_id)
            node_types[node_id] = CHUNK_NODE
            node_texts[node_id] = _compact_text(
                str(chunk_data.get("content", "")),
                config.max_embedding_text_chars,
            )

        logger.info(
            "GNN graph snapshot: %d entities, %d relations, %d chunks",
            len(entity_data),
            len(relations),
            len(chunks),
        )
        return LightRAGGraphSnapshot(
            node_ids=node_ids,
            node_types=node_types,
            node_texts=node_texts,
            entity_data=entity_data,
            chunks=chunks,
            relations=relations,
            entity_source_ids=entity_source_ids,
        )

    @staticmethod
    async def embed_snapshot_nodes(
        rag: Any,
        snapshot: LightRAGGraphSnapshot,
        config: SubgraphCompletionConfig,
    ) -> dict[str, np.ndarray]:
        if rag.embedding_func is None:
            raise ValueError("LightRAG embedding_func is required for GNN augmentation.")

        artifact_dir = config.artifact_dir or os.path.join(
            rag.working_dir, "gnn_subgraph_completion"
        )
        os.makedirs(artifact_dir, exist_ok=True)
        names_path = os.path.join(artifact_dir, "node_embedding_names.json")
        texts_path = os.path.join(artifact_dir, "node_embedding_texts.json")
        vectors_path = os.path.join(artifact_dir, "node_embeddings.npz")

        node_ids = list(snapshot.node_ids)
        texts = [snapshot.node_texts[node_id] for node_id in node_ids]
        if (
            os.path.exists(names_path)
            and os.path.exists(texts_path)
            and os.path.exists(vectors_path)
        ):
            try:
                with open(names_path, "r", encoding="utf-8") as f:
                    cached_names = json.load(f)
                with open(texts_path, "r", encoding="utf-8") as f:
                    cached_texts = json.load(f)
                if cached_names == node_ids and cached_texts == texts:
                    vectors = np.load(vectors_path)["vectors"]
                    return {
                        node_id: vectors[idx].astype(np.float32, copy=False)
                        for idx, node_id in enumerate(node_ids)
                    }
            except Exception as exc:
                logger.warning("Could not load GNN node embedding cache: %s", exc)

        batches: list[np.ndarray] = []
        batch_size = max(1, config.embedding_batch_size)
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            embeddings = await rag.embedding_func(batch, context="document")
            batches.append(np.asarray(embeddings, dtype=np.float32))

        vectors = np.vstack(batches) if batches else np.empty((0, 0), dtype=np.float32)
        np.savez_compressed(vectors_path, vectors=vectors)
        with open(names_path, "w", encoding="utf-8") as f:
            json.dump(node_ids, f, ensure_ascii=False)
        with open(texts_path, "w", encoding="utf-8") as f:
            json.dump(texts, f, ensure_ascii=False)

        logger.info(
            "GNN node embeddings built for %d nodes using entity names and chunk text",
            len(node_ids),
        )
        return {
            node_id: vectors[idx].astype(np.float32, copy=False)
            for idx, node_id in enumerate(node_ids)
        }


def summarize_snapshot(snapshot: LightRAGGraphSnapshot) -> dict[str, Any]:
    chunk_counts = Counter()
    for relation in snapshot.relations:
        chunk_counts.update(relation.source_ids)
    return {
        "entities": len(snapshot.entity_data),
        "relations": len(snapshot.relations),
        "chunks": len(snapshot.chunks),
        "relations_with_chunks": sum(1 for rel in snapshot.relations if rel.source_ids),
        "top_relation_chunks": chunk_counts.most_common(10),
    }
