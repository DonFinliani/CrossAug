from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from dataclasses import asdict
from typing import Any

from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.utils import logger

from .config import SubgraphCompletionConfig
from .schemas import CompletedRelation, LightRAGGraphSnapshot
from .subgraph_sampler import text_processing


class LightRAGRelationInjector:
    def __init__(self, config: SubgraphCompletionConfig) -> None:
        self.config = config

    @staticmethod
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

    @staticmethod
    def _dedupe_keep_order(values: list[str]) -> list[str]:
        seen = set()
        output = []
        for value in values:
            text = str(value or "").strip()
            if not text or text in seen:
                continue
            output.append(text)
            seen.add(text)
        return output

    def _file_path_for_chunk(
        self,
        snapshot: LightRAGGraphSnapshot,
        chunk_id: str,
    ) -> str:
        chunk = snapshot.chunks.get(chunk_id, {})
        path = (
            chunk.get("file_path")
            or chunk.get("title")
            or chunk.get("book_title")
            or "gnn_subgraph_completion"
        )
        return str(path)

    def _file_path_for_relation(
        self,
        snapshot: LightRAGGraphSnapshot,
        relation: CompletedRelation,
    ) -> str:
        file_paths = []
        seen = set()
        for chunk_id in relation.evidence_chunk_ids:
            chunk = snapshot.chunks.get(chunk_id, {})
            path = (
                chunk.get("file_path")
                or chunk.get("title")
                or chunk.get("book_title")
                or "gnn_subgraph_completion"
            )
            if path and path not in seen:
                file_paths.append(str(path))
                seen.add(path)
        return GRAPH_FIELD_SEP.join(file_paths) if file_paths else "gnn_subgraph_completion"

    def _valid_evidence_chunk_ids(
        self,
        snapshot: LightRAGGraphSnapshot,
        relation: CompletedRelation,
    ) -> list[str]:
        chunk_ids = self._dedupe_keep_order(
            [chunk_id for chunk_id in relation.evidence_chunk_ids if chunk_id]
        )
        known_chunk_ids = [chunk_id for chunk_id in chunk_ids if chunk_id in snapshot.chunks]
        return known_chunk_ids or chunk_ids

    def _entity_completion_fields(
        self,
        relation: CompletedRelation,
        entity_name: str,
        evidence_chunk_ids: list[str],
        existing_entities: set[str],
    ) -> tuple[str, str, list[str]] | None:
        if entity_name == relation.source_entity:
            entity_type = relation.source_entity_type or "UNKNOWN"
            description = relation.source_entity_description or ""
        elif entity_name == relation.target_entity:
            entity_type = relation.target_entity_type or "UNKNOWN"
            description = relation.target_entity_description or ""
        else:
            entity_type = "UNKNOWN"
            description = ""

        source_chunk_ids = relation.entity_source_chunk_ids.get(entity_name)
        if not source_chunk_ids:
            normalized_name = text_processing(entity_name)
            source_chunk_ids = relation.entity_source_chunk_ids.get(normalized_name)
        source_chunk_ids = self._dedupe_keep_order(source_chunk_ids or evidence_chunk_ids)

        if not description:
            if entity_name in existing_entities:
                return None
            description = (
                f"{entity_name} was introduced by GNN subgraph completion from "
                f"evidence for: {relation.description}"
            )

        return entity_type or "UNKNOWN", description, source_chunk_ids

    def _build_relation_report(
        self,
        relation: CompletedRelation,
        evidence_chunk_ids: list[str],
        existing_entities: set[str],
        existing_relation_keys: set[tuple[str, str]],
        status: str,
        error: str | None = None,
    ) -> dict[str, Any]:
        report: dict[str, Any] = {
            "status": status,
            "injection_mode": "merge_nodes_and_edges",
            "source_entity_report": {
                "status": "existing"
                if relation.source_entity in existing_entities
                else "created",
                "entity_name": relation.source_entity,
                "entity_type": relation.source_entity_type,
                "entity_description": relation.source_entity_description,
            },
            "target_entity_report": {
                "status": "existing"
                if relation.target_entity in existing_entities
                else "created",
                "entity_name": relation.target_entity,
                "entity_type": relation.target_entity_type,
                "entity_description": relation.target_entity_description,
            },
            "relation": {
                "src_id": relation.source_entity,
                "tgt_id": relation.target_entity,
                "keywords": relation.keywords,
                "description": relation.description,
                "source_id": GRAPH_FIELD_SEP.join(evidence_chunk_ids),
                "weight": max(1.0, float(relation.confidence or 1.0)),
                "preexisting_edge": relation.key in existing_relation_keys,
            },
            "num_evidence_chunks": len(evidence_chunk_ids),
            "completed_relation": asdict(relation),
        }
        if error is not None:
            report["error"] = error
        return report

    async def inject(
        self,
        rag: Any,
        snapshot: LightRAGGraphSnapshot,
        relations: list[CompletedRelation],
    ) -> list[dict[str, Any]]:
        inserted: list[dict[str, Any]] = []
        seen_triples: set[tuple[str, str, str]] = set()
        unique_relations: list[tuple[CompletedRelation, list[str]]] = []
        existing_entities = set(snapshot.entity_ids)
        existing_relation_keys = set(snapshot.relation_keys)

        for relation in relations:
            triple_key = (
                text_processing(relation.source_entity),
                text_processing(relation.keywords),
                text_processing(relation.target_entity),
            )
            if triple_key in seen_triples:
                continue
            seen_triples.add(triple_key)
            evidence_chunk_ids = self._valid_evidence_chunk_ids(snapshot, relation)
            if self.config.dry_run:
                inserted.append(
                    self._build_relation_report(
                        relation,
                        evidence_chunk_ids,
                        existing_entities,
                        existing_relation_keys,
                        "dry_run",
                    )
                )
                continue
            if not relation.source_entity or not relation.target_entity:
                inserted.append(
                    self._build_relation_report(
                        relation,
                        evidence_chunk_ids,
                        existing_entities,
                        existing_relation_keys,
                        "failed",
                        error="empty source or target entity",
                    )
                )
                continue
            if relation.source_entity == relation.target_entity:
                inserted.append(
                    self._build_relation_report(
                        relation,
                        evidence_chunk_ids,
                        existing_entities,
                        existing_relation_keys,
                        "failed",
                        error="source and target entity are identical",
                    )
                )
                continue
            if not evidence_chunk_ids:
                inserted.append(
                    self._build_relation_report(
                        relation,
                        evidence_chunk_ids,
                        existing_entities,
                        existing_relation_keys,
                        "failed",
                        error="no evidence chunk ids",
                    )
                )
                continue
            unique_relations.append((relation, evidence_chunk_ids))

        if self.config.dry_run or not unique_relations:
            logger.info("Injected %d GNN-completed relations", len(inserted))
            return inserted

        maybe_nodes: dict[str, list[dict[str, Any]]] = defaultdict(list)
        maybe_edges: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        node_record_keys: set[tuple[str, str, str]] = set()
        timestamp = int(time.time())

        for relation, evidence_chunk_ids in unique_relations:
            relation_weight = max(1.0, float(relation.confidence or 1.0))
            per_chunk_weight = relation_weight / max(1, len(evidence_chunk_ids))

            for entity_name in (relation.source_entity, relation.target_entity):
                entity_fields = self._entity_completion_fields(
                    relation,
                    entity_name,
                    evidence_chunk_ids,
                    existing_entities,
                )
                if entity_fields is None:
                    continue
                entity_type, entity_description, entity_chunk_ids = entity_fields
                for chunk_id in entity_chunk_ids:
                    node_key = (entity_name, chunk_id, entity_description)
                    if node_key in node_record_keys:
                        continue
                    node_record_keys.add(node_key)
                    maybe_nodes[entity_name].append(
                        {
                            "entity_name": entity_name,
                            "entity_type": entity_type,
                            "description": entity_description,
                            "source_id": chunk_id,
                            "file_path": self._file_path_for_chunk(snapshot, chunk_id),
                            "timestamp": timestamp,
                        }
                    )

            for chunk_id in evidence_chunk_ids:
                maybe_edges[(relation.source_entity, relation.target_entity)].append(
                    {
                        "src_id": relation.source_entity,
                        "tgt_id": relation.target_entity,
                        "weight": per_chunk_weight,
                        "description": relation.description,
                        "keywords": relation.keywords,
                        "source_id": chunk_id,
                        "file_path": self._file_path_for_chunk(snapshot, chunk_id),
                        "timestamp": timestamp,
                    }
                )

        pipeline_status, pipeline_status_lock = self._new_pipeline_status()
        try:
            from lightrag.operate import merge_nodes_and_edges

            await merge_nodes_and_edges(
                chunk_results=[(dict(maybe_nodes), dict(maybe_edges))],
                knowledge_graph_inst=rag.chunk_entity_relation_graph,
                entity_vdb=rag.entities_vdb,
                relationships_vdb=rag.relationships_vdb,
                global_config=asdict(rag),
                full_entities_storage=None,
                full_relations_storage=None,
                doc_id=None,
                pipeline_status=pipeline_status,
                pipeline_status_lock=pipeline_status_lock,
                llm_response_cache=rag.llm_response_cache,
                entity_chunks_storage=rag.entity_chunks,
                relation_chunks_storage=rag.relation_chunks,
                current_file_number=1,
                total_files=1,
                file_path="gnn_subgraph_completion",
            )
            await rag._insert_done(pipeline_status, pipeline_status_lock)
        except Exception as exc:
            logger.warning("Bulk GNN relation injection failed: %s", exc)
            for relation, evidence_chunk_ids in unique_relations:
                inserted.append(
                    self._build_relation_report(
                        relation,
                        evidence_chunk_ids,
                        existing_entities,
                        existing_relation_keys,
                        "failed",
                        error=str(exc),
                    )
                )
            logger.info("Injected %d GNN-completed relations", len(inserted))
            return inserted

        for relation, evidence_chunk_ids in unique_relations:
            status = (
                "merged_existing_edge"
                if relation.key in existing_relation_keys
                else "inserted_new_edge"
            )
            inserted.append(
                self._build_relation_report(
                    relation,
                    evidence_chunk_ids,
                    existing_entities,
                    existing_relation_keys,
                    status,
                )
            )
        logger.info(
            "Injected %d GNN-completed relations via LightRAG merge_nodes_and_edges",
            len(inserted),
        )
        return inserted
