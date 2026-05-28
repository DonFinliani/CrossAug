from __future__ import annotations

import random
import re
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from .config import SubgraphCompletionConfig
from .schemas import (
    CHUNK_NODE,
    CHUNK_NODE_TYPE,
    ENTITY_CHUNK_EDGE_TYPE,
    ENTITY_NODE,
    ENTITY_NODE_TYPE,
    FACT_EDGE_TYPE,
    LightRAGGraphSnapshot,
    OTHER_NODE_TYPE,
    SYNONYM_EDGE_TYPE,
    SampledSubgraphRecord,
    chunk_node_id,
)


def text_processing(text: Any) -> Any:
    if isinstance(text, list):
        return [text_processing(item) for item in text]
    if not isinstance(text, str):
        text = str(text)
    return re.sub("[^A-Za-z0-9 ]", " ", text.lower()).strip()


def _is_four_digit_year(entity_text: str) -> bool:
    stripped = entity_text.strip()
    return stripped.isdigit() and len(stripped) == 4 and 1000 <= int(stripped) <= 2099


def should_filter_candidate_entity_text(entity_text: str) -> bool:
    if not isinstance(entity_text, str) or entity_text.strip() == "":
        return True
    normalized = text_processing(entity_text)
    if normalized == "":
        return True
    if normalized.isdigit():
        return True
    if _is_four_digit_year(normalized):
        return True
    return len(normalized) <= 1


class SubgraphSampler:
    """HippoRAG-style subgraph sampler adapted to LightRAG storage."""

    def __init__(
        self,
        snapshot: LightRAGGraphSnapshot,
        node_features: dict[str, np.ndarray],
        config: SubgraphCompletionConfig,
    ) -> None:
        self.snapshot = snapshot
        self.node_features = node_features
        self.config = config
        self.graph_data = self._extract_full_graph_for_subgraph_mining()

    def _resolve_subgraph_llm_budget(self) -> int:
        raw_budget = float(self.config.subgraph_llm_budget)

        if raw_budget >= 1.0 and raw_budget.is_integer():
            return max(1, int(raw_budget))

        if 0.0 < raw_budget < 1.0:
            node_type_ids = self.graph_data.get("node_type_ids", [])
            num_chunks = sum(
                1 for node_type_id in node_type_ids if node_type_id == CHUNK_NODE_TYPE
            )
            resolved_budget = int(raw_budget * num_chunks * 2.0)
            return max(1, resolved_budget)

        return max(1, int(raw_budget))

    def _resolve_subgraph_inference_root_budget(self) -> int:
        raw_budget = float(self.config.subgraph_inference_root_budget)

        if raw_budget >= 1.0 and raw_budget.is_integer():
            return max(1, int(raw_budget))

        if 0.0 < raw_budget < 1.0:
            node_type_ids = self.graph_data.get("node_type_ids", [])
            num_entity_nodes = sum(
                1 for node_type_id in node_type_ids if node_type_id == ENTITY_NODE_TYPE
            )
            resolved_budget = int(raw_budget * num_entity_nodes)
            return max(1, resolved_budget)

        return max(1, int(raw_budget))

    def _node_type_id(self, node_name: str) -> int:
        node_type = self.snapshot.node_types.get(node_name)
        if node_type == ENTITY_NODE:
            return ENTITY_NODE_TYPE
        if node_type == CHUNK_NODE:
            return CHUNK_NODE_TYPE
        return OTHER_NODE_TYPE

    def _add_edge_record(
        self,
        edge_by_key: dict[tuple[int, int, int], dict[str, Any]],
        src_idx: int,
        dst_idx: int,
        edge_type: int,
        weight: float,
    ) -> None:
        if src_idx == dst_idx or weight <= 0:
            return
        left_idx, right_idx = (src_idx, dst_idx) if src_idx < dst_idx else (dst_idx, src_idx)
        edge_key = (left_idx, right_idx, edge_type)
        if edge_key not in edge_by_key:
            edge_by_key[edge_key] = {
                "src": left_idx,
                "dst": right_idx,
                "edge_type": edge_type,
                "weight": 0.0,
                "edge_key": edge_key,
            }
        edge_by_key[edge_key]["weight"] += float(weight)

    @staticmethod
    def _normalize_embedding_matrix(embeddings: np.ndarray) -> np.ndarray:
        embedding_matrix = np.asarray(embeddings, dtype=np.float32)
        if embedding_matrix.size == 0:
            return embedding_matrix
        embedding_norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
        embedding_norms = np.clip(embedding_norms, 1e-12, None)
        return embedding_matrix / embedding_norms

    @staticmethod
    def _topk_indices_and_scores(
        scores: np.ndarray,
        topk: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if scores.shape[1] == 0 or topk <= 0:
            return (
                np.empty((scores.shape[0], 0), dtype=np.int64),
                np.empty((scores.shape[0], 0), dtype=np.float32),
            )
        local_k = min(int(topk), scores.shape[1])
        if local_k < scores.shape[1]:
            indices = np.argpartition(-scores, local_k - 1, axis=1)[:, :local_k]
            top_scores = np.take_along_axis(scores, indices, axis=1)
            order = np.argsort(-top_scores, axis=1)
            indices = np.take_along_axis(indices, order, axis=1)
            top_scores = np.take_along_axis(top_scores, order, axis=1)
            return indices.astype(np.int64, copy=False), top_scores.astype(
                np.float32,
                copy=False,
            )
        order = np.argsort(-scores, axis=1)
        return order.astype(np.int64, copy=False), np.take_along_axis(
            scores,
            order,
            axis=1,
        ).astype(np.float32, copy=False)

    def _build_synonym_edges_from_entity_embeddings(
        self,
        node_names: list[str],
        node_type_ids: list[int],
        node_features: np.ndarray,
        entity_text_by_node: dict[str, str],
        edge_by_key: dict[tuple[int, int, int], dict[str, Any]],
    ) -> int:
        topk = max(0, int(self.config.synonymy_edge_topk))
        threshold = float(self.config.synonymy_edge_sim_threshold)
        if topk <= 0 or threshold >= 1.0:
            return 0

        entity_indices = [
            idx
            for idx, node_type_id in enumerate(node_type_ids)
            if node_type_id == ENTITY_NODE_TYPE
        ]
        if len(entity_indices) <= 1:
            return 0

        entity_matrix = np.asarray(node_features[entity_indices], dtype=np.float32)
        nonzero_mask = np.linalg.norm(entity_matrix, axis=1) > 1e-12
        if not np.any(nonzero_mask):
            return 0

        entity_indices = [
            entity_idx
            for entity_idx, keep in zip(entity_indices, nonzero_mask)
            if bool(keep)
        ]
        entity_matrix = entity_matrix[nonzero_mask]
        normalized_entity_matrix = self._normalize_embedding_matrix(entity_matrix)
        num_entities = len(entity_indices)
        topk = min(topk, num_entities)
        query_batch_size = max(1, int(self.config.synonymy_edge_query_batch_size))
        key_batch_size = max(1, int(self.config.synonymy_edge_key_batch_size))

        num_synonym_edges = 0
        for query_start in range(0, num_entities, query_batch_size):
            query_end = min(query_start + query_batch_size, num_entities)
            query_batch = normalized_entity_matrix[query_start:query_end]
            batch_score_parts: list[np.ndarray] = []
            batch_index_parts: list[np.ndarray] = []

            for key_start in range(0, num_entities, key_batch_size):
                key_end = min(key_start + key_batch_size, num_entities)
                key_batch = normalized_entity_matrix[key_start:key_end]
                scores = query_batch @ key_batch.T
                top_indices, top_scores = self._topk_indices_and_scores(scores, topk)
                if top_indices.shape[1] == 0:
                    continue
                batch_index_parts.append(top_indices + key_start)
                batch_score_parts.append(top_scores)

            if not batch_index_parts:
                continue

            candidate_indices = np.concatenate(batch_index_parts, axis=1)
            candidate_scores = np.concatenate(batch_score_parts, axis=1)
            final_indices, final_scores = self._topk_indices_and_scores(
                candidate_scores,
                topk,
            )
            candidate_indices = np.take_along_axis(
                candidate_indices,
                final_indices,
                axis=1,
            )

            for row_idx in range(candidate_indices.shape[0]):
                query_entity_pos = query_start + row_idx
                query_node_idx = entity_indices[query_entity_pos]
                query_node_name = node_names[query_node_idx]
                query_text = entity_text_by_node.get(query_node_name, query_node_name)
                if len(re.sub("[^A-Za-z0-9]", "", query_text)) <= 2:
                    continue

                num_neighbors = 0
                for neighbor_pos, score in zip(
                    candidate_indices[row_idx],
                    final_scores[row_idx],
                ):
                    score_value = float(score)
                    if score_value < threshold or num_neighbors > 100:
                        break
                    neighbor_pos = int(neighbor_pos)
                    if neighbor_pos == query_entity_pos:
                        continue
                    neighbor_node_idx = entity_indices[neighbor_pos]
                    neighbor_node_name = node_names[neighbor_node_idx]
                    neighbor_text = entity_text_by_node.get(
                        neighbor_node_name,
                        neighbor_node_name,
                    )
                    if not neighbor_text:
                        continue
                    before_count = len(edge_by_key)
                    self._add_edge_record(
                        edge_by_key,
                        query_node_idx,
                        neighbor_node_idx,
                        SYNONYM_EDGE_TYPE,
                        max(0.0, score_value),
                    )
                    if len(edge_by_key) > before_count:
                        num_synonym_edges += 1
                    num_neighbors += 1

        return num_synonym_edges

    def _build_fact_pair_evidence_lookup(
        self,
    ) -> dict[tuple[str, str], list[tuple[tuple[str, str, str], str]]]:
        lookup: dict[tuple[str, str], list[tuple[tuple[str, str, str], str]]] = defaultdict(list)
        seen: set[tuple[tuple[str, str, str], str]] = set()
        for relation in self.snapshot.relations:
            src_text = text_processing(relation.src)
            tgt_text = text_processing(relation.tgt)
            if not src_text or not tgt_text:
                continue
            predicate = text_processing(relation.keywords or relation.description or "related to")
            triple = (src_text, predicate, tgt_text)
            pair_key = tuple(sorted((src_text, tgt_text)))
            for chunk_id in relation.source_ids:
                if chunk_id not in self.snapshot.chunks:
                    continue
                item_key = (triple, chunk_id)
                if item_key in seen:
                    continue
                lookup[pair_key].append((triple, chunk_id))
                seen.add(item_key)
        return dict(lookup)

    def _extract_full_graph_for_subgraph_mining(self) -> dict[str, Any]:
        node_names = list(self.snapshot.node_ids)
        node_name_to_idx = {node_name: idx for idx, node_name in enumerate(node_names)}
        node_type_ids = [self._node_type_id(node_name) for node_name in node_names]
        embedding_dim = 0
        if self.node_features:
            embedding_dim = int(len(next(iter(self.node_features.values()))))
        if embedding_dim == 0:
            raise ValueError("Cannot build GNN graph because no node embeddings are available.")

        node_features = np.zeros((len(node_names), embedding_dim), dtype=np.float32)
        entity_text_by_node: dict[str, str] = {}
        for idx, node_name in enumerate(node_names):
            node_features[idx] = self.node_features.get(
                node_name,
                np.zeros(embedding_dim, dtype=np.float32),
            )
            if self.snapshot.node_types.get(node_name) == ENTITY_NODE:
                entity_text_by_node[node_name] = node_name

        edge_by_key: dict[tuple[int, int, int], dict[str, Any]] = {}
        for relation in self.snapshot.relations:
            if relation.src not in node_name_to_idx or relation.tgt not in node_name_to_idx:
                continue
            self._add_edge_record(
                edge_by_key,
                node_name_to_idx[relation.src],
                node_name_to_idx[relation.tgt],
                FACT_EDGE_TYPE,
                max(1.0, float(relation.weight)),
            )
            for chunk_id in relation.source_ids:
                chunk_node = chunk_node_id(chunk_id)
                if chunk_node not in node_name_to_idx:
                    continue
                self._add_edge_record(
                    edge_by_key,
                    node_name_to_idx[relation.src],
                    node_name_to_idx[chunk_node],
                    ENTITY_CHUNK_EDGE_TYPE,
                    1.0,
                )
                self._add_edge_record(
                    edge_by_key,
                    node_name_to_idx[relation.tgt],
                    node_name_to_idx[chunk_node],
                    ENTITY_CHUNK_EDGE_TYPE,
                    1.0,
                )

        for entity, chunk_ids in self.snapshot.entity_source_ids.items():
            if entity not in node_name_to_idx:
                continue
            for chunk_id in chunk_ids:
                chunk_node = chunk_node_id(chunk_id)
                if chunk_node not in node_name_to_idx:
                    continue
                self._add_edge_record(
                    edge_by_key,
                    node_name_to_idx[entity],
                    node_name_to_idx[chunk_node],
                    ENTITY_CHUNK_EDGE_TYPE,
                    1.0,
                )

        self._build_synonym_edges_from_entity_embeddings(
            node_names=node_names,
            node_type_ids=node_type_ids,
            node_features=node_features,
            entity_text_by_node=entity_text_by_node,
            edge_by_key=edge_by_key,
        )

        edge_records = list(edge_by_key.values())
        fact_edge_keys = {
            record["edge_key"] for record in edge_records if record["edge_type"] == FACT_EDGE_TYPE
        }
        synonym_edge_keys = {
            record["edge_key"] for record in edge_records if record["edge_type"] == SYNONYM_EDGE_TYPE
        }
        entity_chunk_edge_keys = {
            record["edge_key"]
            for record in edge_records
            if record["edge_type"] == ENTITY_CHUNK_EDGE_TYPE
        }

        weight_clip = max(0.0, self.config.subgraph_walk_weight_clip)
        edge_type_scale = {
            FACT_EDGE_TYPE: self.config.subgraph_fact_walk_scale,
            SYNONYM_EDGE_TYPE: self.config.subgraph_synonym_walk_scale,
        }
        adjacency_accumulator: list[dict[int, float]] = [defaultdict(float) for _ in node_names]
        for record in edge_records:
            if record["edge_type"] not in (FACT_EDGE_TYPE, SYNONYM_EDGE_TYPE):
                continue
            scale = edge_type_scale.get(record["edge_type"], 1.0)
            if scale <= 0:
                continue
            effective_weight = min(float(record["weight"]), weight_clip) * scale
            if effective_weight <= 0:
                continue
            adjacency_accumulator[record["src"]][record["dst"]] += effective_weight
            adjacency_accumulator[record["dst"]][record["src"]] += effective_weight
        weighted_adjacency = [
            {"neighbors": list(neighbor_map.keys()), "weights": list(neighbor_map.values())}
            for neighbor_map in adjacency_accumulator
        ]

        root_candidates = [
            idx
            for idx, node_name in enumerate(node_names)
            if self.snapshot.node_types.get(node_name) == ENTITY_NODE
            and not should_filter_candidate_entity_text(entity_text_by_node.get(node_name, ""))
            and len(weighted_adjacency[idx]["neighbors"]) > 0
        ]

        edge_records_by_node: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for record in edge_records:
            edge_records_by_node[record["src"]].append(record)
            edge_records_by_node[record["dst"]].append(record)

        return {
            "node_names": node_names,
            "node_name_to_idx": node_name_to_idx,
            "node_type_ids": node_type_ids,
            "node_features": node_features,
            "entity_text_by_node": entity_text_by_node,
            "edge_records": edge_records,
            "edge_records_by_node": dict(edge_records_by_node),
            "weighted_adjacency": weighted_adjacency,
            "root_candidates": root_candidates,
            "fact_pair_evidence_lookup": self._build_fact_pair_evidence_lookup(),
            "num_nodes": len(node_names),
            "num_edges": len(edge_records),
            "num_fact_edges": len(fact_edge_keys),
            "num_synonym_edges": len(synonym_edge_keys),
            "num_entity_chunk_edges": len(entity_chunk_edge_keys),
        }

    def _sample_weighted_random_walk_nodes(
        self,
        root_node_idx: int,
        rng: random.Random,
        walks_per_root: int | None = None,
    ) -> list[int]:
        walk_length = max(1, self.config.subgraph_walk_length)
        walks_per_root = max(
            1,
            int(walks_per_root)
            if walks_per_root is not None
            else self.config.subgraph_walks_per_root,
        )
        weighted_adjacency = self.graph_data["weighted_adjacency"]
        node_type_ids = self.graph_data["node_type_ids"]
        visited = {int(root_node_idx)}

        for _ in range(walks_per_root):
            current_idx = int(root_node_idx)
            entity_hops_taken = 0
            while entity_hops_taken < walk_length:
                neighbors = weighted_adjacency[current_idx]["neighbors"]
                weights = weighted_adjacency[current_idx]["weights"]
                if not neighbors:
                    break
                current_idx = int(rng.choices(neighbors, weights=weights, k=1)[0])
                visited.add(current_idx)
                if node_type_ids[current_idx] == ENTITY_NODE_TYPE:
                    entity_hops_taken += 1
        return sorted(visited)

    def _count_fact_edges_in_node_set(self, node_indices: set[int]) -> int:
        return sum(
            1
            for record in self.graph_data["edge_records"]
            if record["edge_type"] == FACT_EDGE_TYPE
            and record["src"] in node_indices
            and record["dst"] in node_indices
        )

    def _build_subgraph_record_from_node_indices(
        self,
        root_node_idx: int,
        node_indices: list[int],
        subgraph_id: str,
    ) -> SampledSubgraphRecord | None:
        node_set = set(node_indices)
        min_nodes = max(2, self.config.subgraph_min_nodes)
        min_fact_edges = max(1, self.config.subgraph_min_fact_edges)
        if len(node_indices) < min_nodes:
            return None
        unique_fact_edges = self._count_fact_edges_in_node_set(node_set)
        if unique_fact_edges < min_fact_edges:
            return None

        node_names = self.graph_data["node_names"]
        entity_node_names = sorted(
            node_names[idx]
            for idx in node_indices
            if self.graph_data["node_type_ids"][idx] == ENTITY_NODE_TYPE
        )
        entity_texts = sorted(
            self.graph_data["entity_text_by_node"].get(node_name, node_name)
            for node_name in entity_node_names
        )
        chunk_ids = sorted(
            node_names[idx].removeprefix("chunk::")
            for idx in node_indices
            if self.graph_data["node_type_ids"][idx] == CHUNK_NODE_TYPE
        )
        root_name = node_names[root_node_idx]
        return SampledSubgraphRecord(
            subgraph_id=subgraph_id,
            root_node_idx=int(root_node_idx),
            root_node_name=root_name,
            root_node_text=self.graph_data["entity_text_by_node"].get(root_name, root_name),
            node_indices=node_indices,
            entity_node_names=entity_node_names,
            entity_texts=entity_texts,
            chunk_ids=chunk_ids,
            unique_fact_edges=unique_fact_edges,
        )

    def sample_subgraph_record(
        self,
        root_node_idx: int,
        rng: random.Random,
        subgraph_id: str,
        walks_per_root: int | None = None,
    ) -> SampledSubgraphRecord | None:
        node_indices = self._sample_weighted_random_walk_nodes(
            root_node_idx,
            rng,
            walks_per_root=walks_per_root,
        )
        return self._build_subgraph_record_from_node_indices(
            root_node_idx=root_node_idx,
            node_indices=node_indices,
            subgraph_id=subgraph_id,
        )

    def fact_edge_keys_in_subgraph(self, node_indices: list[int]) -> list[tuple[int, int, int]]:
        node_set = set(node_indices)
        return [
            record["edge_key"]
            for record in self.graph_data["edge_records"]
            if record["edge_type"] == FACT_EDGE_TYPE
            and record["src"] in node_set
            and record["dst"] in node_set
        ]

    def entity_delete_candidates(self, node_indices: list[int], root_node_idx: int) -> list[int]:
        node_set = set(node_indices)
        candidates = []
        for node_idx in node_indices:
            if node_idx == root_node_idx:
                continue
            if self.graph_data["node_type_ids"][node_idx] != ENTITY_NODE_TYPE:
                continue
            has_fact = False
            for record in self.graph_data["edge_records_by_node"].get(node_idx, []):
                other_idx = record["dst"] if record["src"] == node_idx else record["src"]
                if other_idx in node_set and record["edge_type"] == FACT_EDGE_TYPE:
                    has_fact = True
                    break
            if has_fact:
                candidates.append(node_idx)
        return candidates

    def sample_training_groups(
        self,
        rng: random.Random,
        num_roots: int,
        prefix: str,
    ) -> list[dict[str, Any]]:
        root_candidates = self.graph_data["root_candidates"][:]
        if not root_candidates:
            raise ValueError("No valid entity root candidates are available for subgraph GNN training.")

        from .model import build_subgraph_data

        groups: list[dict[str, Any]] = []
        attempts = 0
        max_attempts = max(num_roots * 20, 100)
        inference_walks_per_root = max(1, self.config.subgraph_walks_per_root)
        negative_walks_per_root = max(
            inference_walks_per_root,
            self.config.subgraph_negative_walks_per_root,
        )
        while len(groups) < num_roots and attempts < max_attempts:
            attempts += 1
            root_idx = rng.choice(root_candidates)
            clean_record = self.sample_subgraph_record(
                root_node_idx=root_idx,
                rng=rng,
                subgraph_id=f"{prefix}-{attempts}-negative-sampled",
                walks_per_root=negative_walks_per_root,
            )
            if clean_record is None:
                continue
            corruption_record = self.sample_subgraph_record(
                root_node_idx=root_idx,
                rng=rng,
                subgraph_id=f"{prefix}-{attempts}-sampled",
                walks_per_root=inference_walks_per_root,
            )
            if corruption_record is None:
                continue

            full_data = build_subgraph_data(self.graph_data, clean_record.node_indices, 0.0)
            corrupted = []
            fact_edge_keys = self.fact_edge_keys_in_subgraph(corruption_record.node_indices)
            min_fact_edges = max(1, self.config.subgraph_min_fact_edges)
            max_removable_fact_edges = max(0, len(fact_edge_keys) - min_fact_edges)
            if max_removable_fact_edges > 0:
                mask_ratio = min(max(self.config.subgraph_fact_mask_ratio, 0.0), 1.0)
                mask_count = min(
                    max_removable_fact_edges,
                    max(1, int(round(len(fact_edge_keys) * mask_ratio))),
                )
                removed_fact_edges = set(rng.sample(fact_edge_keys, mask_count))
                corrupted.append(
                    build_subgraph_data(
                        self.graph_data,
                        corruption_record.node_indices,
                        1.0,
                        removed_edge_keys=removed_fact_edges,
                        corruption_type=1,
                    )
                )

            entity_candidates = self.entity_delete_candidates(
                corruption_record.node_indices,
                corruption_record.root_node_idx,
            )
            if entity_candidates:
                delete_ratio = min(max(self.config.subgraph_entity_delete_ratio, 0.0), 1.0)
                delete_count = max(1, int(round(len(entity_candidates) * delete_ratio)))
                delete_count = min(delete_count, len(entity_candidates))
                deleted_entities = set(rng.sample(entity_candidates, delete_count))
                remaining_nodes = set(corruption_record.node_indices) - deleted_entities
                if self._count_fact_edges_in_node_set(remaining_nodes) >= min_fact_edges:
                    corrupted.append(
                        build_subgraph_data(
                            self.graph_data,
                            corruption_record.node_indices,
                            1.0,
                            removed_node_indices=deleted_entities,
                            corruption_type=2,
                        )
                    )
            if not corrupted:
                continue
            groups.append(
                {
                    "clean_record": clean_record,
                    "corruption_record": corruption_record,
                    "full": full_data,
                    "corrupted": corrupted,
                }
            )

        if not groups:
            raise ValueError("Failed to construct any valid subgraph missingness training examples.")
        return groups

    @staticmethod
    def flatten_subgraph_groups(groups: list[dict[str, Any]]) -> tuple[list[Any], list[tuple[int, int]]]:
        data_list = []
        ranking_pairs = []
        for group in groups:
            full_idx = len(data_list)
            data_list.append(group["full"])
            for corrupted_data in group["corrupted"]:
                corrupted_idx = len(data_list)
                data_list.append(corrupted_data)
                ranking_pairs.append((full_idx, corrupted_idx))
        return data_list, ranking_pairs

    def resolve_root_sample_count(self, requested_count: int, allow_zero: bool = False) -> int:
        requested_count = max(0, int(requested_count))
        max_available_roots = len(self.graph_data.get("root_candidates", []))
        if max_available_roots == 0:
            return 0
        if requested_count == 0:
            return 0 if allow_zero else 1
        resolved_count = requested_count if allow_zero else max(1, requested_count)
        return min(resolved_count, max_available_roots)

    def entity_hop_distances_from_root(self, root_idx: int, hop_k: int) -> dict[int, int]:
        weighted_adjacency = self.graph_data["weighted_adjacency"]
        node_type_ids = self.graph_data["node_type_ids"]
        best_entity_hops = {int(root_idx): 0}
        frontier = [(int(root_idx), 0)]
        while frontier:
            next_frontier = []
            for node_idx, entity_hops in frontier:
                for neighbor_idx in weighted_adjacency[node_idx]["neighbors"]:
                    next_entity_hops = entity_hops + (
                        1 if node_type_ids[neighbor_idx] == ENTITY_NODE_TYPE else 0
                    )
                    if next_entity_hops > hop_k:
                        continue
                    previous_best = best_entity_hops.get(int(neighbor_idx))
                    if previous_best is not None and previous_best <= next_entity_hops:
                        continue
                    best_entity_hops[int(neighbor_idx)] = next_entity_hops
                    next_frontier.append((int(neighbor_idx), next_entity_hops))
            frontier = next_frontier
        return best_entity_hops

    @staticmethod
    def node_jaccard(left_nodes: list[int], right_nodes: list[int]) -> float:
        left = set(left_nodes)
        right = set(right_nodes)
        if not left and not right:
            return 1.0
        return len(left & right) / max(1, len(left | right))

    @staticmethod
    def summarize_missing_scores(records: list[SampledSubgraphRecord]) -> dict[str, Any]:
        scores = [float(record.missing_score) for record in records]
        if not scores:
            return {"count": 0, "avg": None, "min": None, "max": None}
        return {
            "count": len(scores),
            "avg": round(sum(scores) / len(scores), 6),
            "min": round(min(scores), 6),
            "max": round(max(scores), 6),
        }

    def collect_subgraph_evidence(
        self,
        record: SampledSubgraphRecord,
    ) -> SampledSubgraphRecord:
        node_names = self.graph_data["node_names"]
        entity_node_names = sorted(
            node_names[idx]
            for idx in record.node_indices
            if self.graph_data["node_type_ids"][idx] == ENTITY_NODE_TYPE
        )
        entity_texts = sorted(
            self.graph_data["entity_text_by_node"].get(node_name, node_name)
            for node_name in entity_node_names
        )
        max_known = max(1, self.config.subgraph_max_known_triples_per_prompt)
        max_chunks = max(1, self.config.subgraph_max_chunks_per_prompt)
        entity_norms = {text_processing(entity_text) for entity_text in entity_texts}

        fact_triples: list[tuple[str, str, str]] = []
        seen_triples = set()
        chunk_ids = []
        seen_chunk_ids: set[str] = set()
        node_set = set(record.node_indices)
        for edge_record in self.graph_data["edge_records"]:
            if edge_record["edge_type"] != FACT_EDGE_TYPE:
                continue
            if edge_record["src"] not in node_set or edge_record["dst"] not in node_set:
                continue
            left_name = node_names[edge_record["src"]]
            right_name = node_names[edge_record["dst"]]
            left_text = text_processing(self.graph_data["entity_text_by_node"].get(left_name, left_name))
            right_text = text_processing(self.graph_data["entity_text_by_node"].get(right_name, right_name))
            pair_key = tuple(sorted((left_text, right_text)))
            for triple, chunk_id in self.graph_data["fact_pair_evidence_lookup"].get(pair_key, []):
                if triple in seen_triples:
                    continue
                if chunk_id not in seen_chunk_ids:
                    if len(chunk_ids) >= max_chunks:
                        continue
                    chunk_ids.append(chunk_id)
                    seen_chunk_ids.add(chunk_id)
                fact_triples.append(triple)
                seen_triples.add(triple)
                if len(fact_triples) >= max_known:
                    break
            if len(fact_triples) >= max_known:
                break

        known_triples = [
            triple for triple in fact_triples if triple[0] in entity_norms and triple[2] in entity_norms
        ]
        if not known_triples:
            known_triples = fact_triples

        record.entity_node_names = entity_node_names
        record.entity_texts = entity_texts
        record.chunk_ids = chunk_ids
        record.known_triples = known_triples[:max_known]
        return record

    def sample_inference_records(self) -> tuple[list[SampledSubgraphRecord], dict[str, Any]]:
        rng = random.Random(int(self.config.random_seed) + 99991)
        root_candidates = self.graph_data["root_candidates"][:]
        rng.shuffle(root_candidates)
        root_budget = min(
            self._resolve_subgraph_inference_root_budget(),
            max(1, len(root_candidates)),
        )
        root_selection_hops = max(1, self.config.subgraph_walk_length)

        sampled_records = []
        remaining_roots = set(root_candidates)
        current_root_idx = None
        attempts = 0
        jump_fallbacks = 0
        while remaining_roots and len(sampled_records) < root_budget:
            next_root_idx = None
            if current_root_idx is not None:
                root_distance_map = self.entity_hop_distances_from_root(
                    current_root_idx,
                    root_selection_hops,
                )
                for root_idx in root_candidates:
                    if root_idx not in remaining_roots:
                        continue
                    if root_distance_map.get(root_idx) == root_selection_hops:
                        next_root_idx = root_idx
                        break
            if next_root_idx is None:
                jump_fallbacks += 1 if current_root_idx is not None else 0
                next_root_idx = next(
                    (root_idx for root_idx in root_candidates if root_idx in remaining_roots),
                    None,
                )
            if next_root_idx is None:
                break

            current_root_idx = next_root_idx
            remaining_roots.remove(current_root_idx)
            if len(sampled_records) >= root_budget:
                break
            attempts += 1
            record = self.sample_subgraph_record(
                root_node_idx=current_root_idx,
                rng=rng,
                subgraph_id=f"infer-{len(sampled_records)}",
            )
            if record is not None:
                sampled_records.append(record)

        return sampled_records, {
            "raw_inference_root_budget": self.config.subgraph_inference_root_budget,
            "resolved_inference_root_budget": root_budget,
            "num_inference_roots_attempted": attempts,
            "root_selection_hops": root_selection_hops,
            "num_root_jump_fallbacks": jump_fallbacks,
            "num_sampled_subgraphs": len(sampled_records),
        }

    def select_scored_records(
        self,
        scored_records: list[SampledSubgraphRecord],
        selection_stats: dict[str, Any],
    ) -> list[SampledSubgraphRecord]:
        missing_threshold = self.config.subgraph_missing_score_threshold
        llm_budget = self._resolve_subgraph_llm_budget()
        overlap_threshold = min(max(self.config.subgraph_overlap_threshold, 0.0), 1.0)
        scored_records.sort(key=lambda item: item.missing_score, reverse=True)
        selected_records = []
        for record in scored_records:
            if record.missing_score < missing_threshold:
                continue
            if any(
                self.node_jaccard(record.node_indices, selected.node_indices) > overlap_threshold
                for selected in selected_records
            ):
                continue
            record_with_evidence = self.collect_subgraph_evidence(record)
            if len(record_with_evidence.chunk_ids) == 0:
                continue
            selected_records.append(record_with_evidence)
            if len(selected_records) >= llm_budget:
                break

        selected_ids = {record.subgraph_id for record in selected_records}
        unselected_records = [
            record for record in scored_records if record.subgraph_id not in selected_ids
        ]
        scored_score_stats = self.summarize_missing_scores(scored_records)
        selected_score_stats = self.summarize_missing_scores(selected_records)
        unselected_score_stats = self.summarize_missing_scores(unselected_records)
        selection_stats.update(
            {
                "num_scored_subgraphs": len(scored_records),
                "num_selected_subgraphs_for_llm": len(selected_records),
                "num_unselected_subgraphs_for_llm": len(unselected_records),
                "missing_score_threshold": missing_threshold,
                "raw_llm_budget": self.config.subgraph_llm_budget,
                "llm_budget": llm_budget,
                "overlap_threshold": overlap_threshold,
                "avg_scored_missing_score": scored_score_stats["avg"],
                "min_scored_missing_score": scored_score_stats["min"],
                "max_scored_missing_score": scored_score_stats["max"],
                "avg_selected_missing_score": selected_score_stats["avg"] if selected_score_stats["avg"] is not None else 0.0,
                "min_selected_missing_score": selected_score_stats["min"],
                "max_selected_missing_score": selected_score_stats["max"],
                "avg_unselected_missing_score": unselected_score_stats["avg"],
                "min_unselected_missing_score": unselected_score_stats["min"],
                "max_unselected_missing_score": unselected_score_stats["max"],
            }
        )
        self.graph_data["hidden_triplet_subgraph_selection_stats"] = selection_stats
        return selected_records
