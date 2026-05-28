import json
import logging
import math
import os
import random
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import igraph as ig
try:
    import faiss
except ImportError:  # pragma: no cover - handled explicitly at runtime
    faiss = None
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GCNConv, SAGEConv, global_max_pool, global_mean_pool
try:
    from torch_geometric.nn import GATv2Conv
except ImportError:  # pragma: no cover - depends on the installed PyG version
    GATv2Conv = None
try:
    from torch_geometric.nn import RGCNConv
except ImportError:  # pragma: no cover - depends on the installed PyG version
    RGCNConv = None
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from .prompts.templates.ner import one_shot_ner_paragraph
from .utils.llm_utils import fix_broken_generated_json, filter_invalid_triples
from .utils.misc_utils import extract_entity_nodes, reformat_openie_results, text_processing

logger = logging.getLogger(__name__)

FACT_EDGE_TYPE = 0
SYNONYM_EDGE_TYPE = 1
ENTITY_CHUNK_EDGE_TYPE = 2
NUM_ENTITY_EDGE_TYPES = 2
NUM_SUBGRAPH_EDGE_TYPES = 3

ENTITY_NODE_TYPE = 0
CHUNK_NODE_TYPE = 1
OTHER_NODE_TYPE = 2
NUM_SUBGRAPH_NODE_TYPES = 3

# Legacy pairwise link-prediction mining is no longer configurable through
# BaseConfig. Keep its historical defaults here so the leftover helper methods
# and maintenance scripts remain runnable without surfacing deprecated knobs.
LEGACY_PAIRWISE_MASK_RATIO = 0.15
LEGACY_PAIRWISE_CONTRASTIVE_TEMPERATURE = 0.2
LEGACY_PAIRWISE_NEGATIVES_PER_POSITIVE = 20
LEGACY_PAIRWISE_USE_FIXED_EVAL_SPLIT = False
LEGACY_PAIRWISE_EVAL_RATIO = 0.05
LEGACY_PAIRWISE_RAW_SIMILARITY_UPPER_THRESHOLD = 0.98
LEGACY_PAIRWISE_CANDIDATE_TOPK = 5
LEGACY_PAIRWISE_MAX_CANDIDATE_PAIRS = 0.2
LEGACY_PAIRWISE_MAX_CHUNKS_PER_CALL = 4
LEGACY_PAIRWISE_MAX_PAIRS_PER_CALL = 10

@dataclass
class CandidatePairRecord:
    pair_ids: Tuple[str, str]
    pair_names: Tuple[str, str]
    score: float
    chunk_ids: List[str]


@dataclass
class CandidateBatch:
    pair_records: List[CandidatePairRecord]
    chunk_ids: List[str]


@dataclass
class SampledSubgraphRecord:
    subgraph_id: str
    root_node_idx: int
    root_node_name: str
    root_node_text: str
    node_indices: List[int]
    missing_score: float = 0.0
    entity_node_names: List[str] = field(default_factory=list)
    entity_texts: List[str] = field(default_factory=list)
    chunk_ids: List[str] = field(default_factory=list)
    known_triples: List[Tuple[str, str, str]] = field(default_factory=list)
    unique_fact_edges: int = 0


@dataclass
class SubgraphCompletionBatch:
    subgraph: SampledSubgraphRecord
    chunk_ids: List[str]
    known_triples: List[Tuple[str, str, str]]


class InMemoryEntityEmbeddingStore:
    """Minimal in-memory store that matches the original graph-building API."""

    def __init__(self, rows: Dict[str, Dict[str, Any]], embeddings: List[np.ndarray]):
        self.hash_ids = list(rows.keys())
        self.hash_id_to_row = {hash_id: dict(row) for hash_id, row in rows.items()}
        self.hash_id_to_idx = {hash_id: idx for idx, hash_id in enumerate(self.hash_ids)}
        self._embeddings = [np.asarray(embedding, dtype=np.float32) for embedding in embeddings]

    def get_all_id_to_rows(self) -> Dict[str, Dict[str, Any]]:
        return {hash_id: dict(row) for hash_id, row in self.hash_id_to_row.items()}

    def get_all_ids(self) -> List[str]:
        return list(self.hash_ids)

    def get_embeddings(self, hash_ids: List[str], dtype=np.float32) -> np.ndarray:
        if not hash_ids:
            return np.array([], dtype=dtype)
        indices = [self.hash_id_to_idx[hash_id] for hash_id in hash_ids]
        return np.array([self._embeddings[idx] for idx in indices], dtype=dtype)


class PyGEntityLinkEncoder(nn.Module):
    """Configurable PyG encoder used for entity-entity link prediction."""

    def __init__(self, input_dim: int, hidden_dim: int, encoder_type: str = "gcn"):
        super().__init__()
        self.encoder_type = encoder_type.lower()
        if self.encoder_type == "gcn":
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
        elif self.encoder_type == "graphsage":
            self.conv1 = SAGEConv(input_dim, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        elif self.encoder_type == "gatv2":
            if GATv2Conv is None:
                raise ImportError("GATv2Conv is not available in the installed torch_geometric version.")
            self.input_projection = nn.Linear(input_dim, hidden_dim)
            self.conv1 = GATv2Conv(hidden_dim, hidden_dim, heads=4, concat=False, dropout=0.1)
            self.conv2 = GATv2Conv(hidden_dim, hidden_dim, heads=4, concat=False, dropout=0.1)
            self.norm1 = nn.LayerNorm(hidden_dim)
            self.norm2 = nn.LayerNorm(hidden_dim)
        elif self.encoder_type == "rgcn":
            if RGCNConv is None:
                raise ImportError("RGCNConv is not available in the installed torch_geometric version.")
            self.conv1 = RGCNConv(input_dim, hidden_dim, num_relations=NUM_ENTITY_EDGE_TYPES)
            self.conv2 = RGCNConv(hidden_dim, hidden_dim, num_relations=NUM_ENTITY_EDGE_TYPES)
        else:
            raise ValueError(
                f"Unsupported hidden triplet GNN encoder type: {encoder_type}. "
                "Expected one of: gcn, graphsage, gatv2, rgcn."
            )

    def forward(self, data: Data) -> torch.Tensor:
        # Intentionally ignore the original graph edge weights during message
        # passing. In this pipeline, raw weights are only used upstream to
        # distinguish fact edges from synonym edges, not to scale propagation.
        if self.encoder_type == "gatv2":
            hidden = self.input_projection(data.x)
            hidden = self.norm1(hidden + self.conv1(hidden, data.edge_index))
            hidden = F.gelu(hidden)
            hidden = F.dropout(hidden, p=0.1, training=self.training)
            return self.norm2(hidden + self.conv2(hidden, data.edge_index))

        if self.encoder_type == "rgcn":
            if not hasattr(data, "edge_type"):
                raise ValueError("RGCN encoder requires support_graph.edge_type.")
            # R-GCN keeps fact and synonym edges as separate relations, so each
            # relation gets its own learned message transformation.
            hidden = self.conv1(data.x, data.edge_index, data.edge_type)
            hidden = F.relu(hidden)
            hidden = F.dropout(hidden, p=0.1, training=self.training)
            return self.conv2(hidden, data.edge_index, data.edge_type)

        hidden = self.conv1(data.x, data.edge_index)
        hidden = F.relu(hidden)
        hidden = F.dropout(hidden, p=0.1, training=self.training)
        return self.conv2(hidden, data.edge_index)


class DotProductLinkPredictor(nn.Module):
    """Decode an edge score directly from the two endpoint embeddings."""

    def forward(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src_idx, dst_idx = edge_index
        return (z[src_idx] * z[dst_idx]).sum(dim=1)


class PyGSubgraphMissingnessClassifier(nn.Module):
    """Graph-level classifier that predicts whether a sampled subgraph is incomplete."""

    def __init__(self, input_dim: int, hidden_dim: int, encoder_type: str = "rgcn"):
        super().__init__()
        self.encoder_type = encoder_type.lower()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.node_type_embedding = nn.Embedding(NUM_SUBGRAPH_NODE_TYPES, hidden_dim)

        if self.encoder_type == "gcn":
            self.conv1 = GCNConv(hidden_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
        elif self.encoder_type == "graphsage":
            self.conv1 = SAGEConv(hidden_dim, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        elif self.encoder_type == "gatv2":
            if GATv2Conv is None:
                raise ImportError("GATv2Conv is not available in the installed torch_geometric version.")
            self.conv1 = GATv2Conv(hidden_dim, hidden_dim, heads=4, concat=False, dropout=0.1)
            self.conv2 = GATv2Conv(hidden_dim, hidden_dim, heads=4, concat=False, dropout=0.1)
        elif self.encoder_type == "rgcn":
            if RGCNConv is None:
                raise ImportError("RGCNConv is not available in the installed torch_geometric version.")
            self.conv1 = RGCNConv(hidden_dim, hidden_dim, num_relations=NUM_SUBGRAPH_EDGE_TYPES)
            self.conv2 = RGCNConv(hidden_dim, hidden_dim, num_relations=NUM_SUBGRAPH_EDGE_TYPES)
        else:
            raise ValueError(
                f"Unsupported hidden triplet subgraph GNN encoder type: {encoder_type}. "
                "Expected one of: gcn, graphsage, gatv2, rgcn."
            )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, 1),
        )

    def _message_pass(self, hidden: torch.Tensor, data: Data) -> torch.Tensor:
        if self.encoder_type == "rgcn":
            if not hasattr(data, "edge_type"):
                raise ValueError("RGCN subgraph encoder requires data.edge_type.")
            hidden = self.conv1(hidden, data.edge_index, data.edge_type)
            hidden = self.norm1(F.gelu(hidden))
            hidden = F.dropout(hidden, p=0.1, training=self.training)
            hidden = self.conv2(hidden, data.edge_index, data.edge_type)
            return self.norm2(F.gelu(hidden))

        hidden = self.conv1(hidden, data.edge_index)
        hidden = self.norm1(F.gelu(hidden))
        hidden = F.dropout(hidden, p=0.1, training=self.training)
        hidden = self.conv2(hidden, data.edge_index)
        return self.norm2(F.gelu(hidden))

    def forward(self, data: Data) -> torch.Tensor:
        node_types = getattr(data, "node_type", None)
        if node_types is None:
            node_types = torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device)
        hidden = self.input_projection(data.x) + self.node_type_embedding(node_types.long())
        hidden = self._message_pass(hidden, data)

        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(hidden.size(0), dtype=torch.long, device=hidden.device)
        pooled = torch.cat([global_mean_pool(hidden, batch), global_max_pool(hidden, batch)], dim=-1)
        return self.classifier(pooled).view(-1)


class GNNHiddenTripletMiner:
    def __init__(self, hipporag):
        self.hipporag = hipporag
        self.global_config = hipporag.global_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Cache the global chunk-retrieval index separately from the entity-pair
        # ANN index used inside _generate_candidate_records(). The pair ANN index
        # depends on the current GNN embeddings and must be rebuilt every run,
        # while the chunk index is based on the static HippoRAG passage
        # embeddings and can be reused across all pair->chunk searches.
        self._cached_chunk_search_node_keys = None
        self._cached_normalized_chunk_search_embeddings = None
        self._cached_global_chunk_faiss_index = None

        llm_label = self.global_config.llm_name.replace("/", "_")
        self.audit_path = os.path.join(
            self.global_config.save_dir,
            f"hidden_triplet_mining_{llm_label}.json",
        )
        self.hidden_triplet_results_path = self.hipporag.hidden_triplet_results_path
        self.hidden_triplet_gnn_triples_dump_path = self.hipporag.hidden_triplet_gnn_triples_dump_path
        self.augmented_graph_path = self.hipporag._augmented_graph_pickle_filename

    @staticmethod
    def _set_gnn_training_seed(seed: int):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except TypeError:
                torch.use_deterministic_algorithms(True)

    def mine(self) -> Dict[str, Any]:
        mining_start_time = time.perf_counter()

        if not self.hipporag.ready_to_retrieve:
            self.hipporag.prepare_retrieval_objects()

        graph_data = self._extract_full_graph_for_subgraph_mining()
        num_fact_edges = int(graph_data["num_fact_edges"])
        if num_fact_edges == 0:
            logger.info("Hidden triplet mining skipped because no fact edges were found in the full graph.")
            return {
                "num_fact_edges": 0,
                "num_synonym_edges": int(graph_data["num_synonym_edges"]),
                "num_candidate_pairs": 0,
                "num_candidate_subgraphs": 0,
                "num_llm_batches": 0,
                "num_new_triples": 0,
                "gnn_training_time_sec": 0.0,
                "gnn_train_loss": 0.0,
                "gnn_eval_loss": 0.0,
                "gnn_best_eval_loss": 0.0,
                "gnn_eval_accuracy": 0.0,
                "gnn_eval_pairwise_accuracy": None,
                "gnn_eval_ranking_accuracy": 0.0,
                "total_mining_time_sec": round(time.perf_counter() - mining_start_time, 4),
            }

        train_output = self._train_subgraph_missingness_detector(graph_data)
        selected_subgraphs = self._select_missing_subgraphs_for_llm(
            graph_data=graph_data,
            model=train_output["model"],
        )
        mined_triples, batch_logs = self._extract_hidden_triples_from_subgraphs(selected_subgraphs)
        added_triples = self._augment_graph_with_triples(mined_triples)
        total_mining_time_sec = time.perf_counter() - mining_start_time
        self._save_subgraph_audit(
            graph_data,
            train_output,
            selected_subgraphs,
            batch_logs,
            added_triples,
            total_mining_time_sec,
        )

        self.hipporag.ready_to_retrieve = False

        summary = {
            "hidden_triplet_mining_strategy": "subgraph_missingness",
            "num_graph_nodes": int(graph_data["num_nodes"]),
            "num_graph_edges": int(graph_data["num_edges"]),
            "num_fact_edges": num_fact_edges,
            "num_synonym_edges": int(graph_data["num_synonym_edges"]),
            "num_entity_chunk_edges": int(graph_data["num_entity_chunk_edges"]),
            "num_candidate_pairs": 0,
            "num_candidate_subgraphs": len(selected_subgraphs),
            "num_llm_batches": len(batch_logs),
            "num_mined_triples": len(mined_triples),
            "num_new_triples": len(added_triples),
            "hidden_triplet_candidate_generation": graph_data.get("hidden_triplet_subgraph_selection_stats", {}),
            "resolved_train_roots_per_epoch": int(train_output["resolved_train_roots_per_epoch"]),
            "resolved_eval_roots": int(train_output["resolved_eval_roots"]),
            "gnn_training_time_sec": round(float(train_output["training_time_sec"]), 4),
            "gnn_train_loss": round(float(train_output["train_loss"]), 6),
            "gnn_eval_loss": None if train_output["eval_loss"] is None else round(float(train_output["eval_loss"]), 6),
            "gnn_best_eval_loss": (
                None
                if train_output["best_eval_loss"] is None
                else round(float(train_output["best_eval_loss"]), 6)
            ),
            "gnn_eval_accuracy": None if train_output["eval_accuracy"] is None else round(float(train_output["eval_accuracy"]), 6),
            "gnn_eval_pairwise_accuracy": None,
            "gnn_eval_ranking_accuracy": (
                None
                if train_output["eval_ranking_accuracy"] is None
                else round(float(train_output["eval_ranking_accuracy"]), 6)
            ),
            "gnn_eval_f1": None if train_output["eval_f1"] is None else round(float(train_output["eval_f1"]), 6),
            "gnn_encoder_type": self.global_config.hidden_triplet_gnn_encoder_type,
            "gnn_completed_epochs": int(train_output["completed_epochs"]),
            "gnn_early_stopped": bool(train_output["early_stopped"]),
            "gnn_best_epoch": train_output["best_epoch"],
            "gnn_best_eval_pairwise_accuracy": None,
            "gnn_best_eval_ranking_accuracy": (
                None
                if train_output["best_eval_ranking_accuracy"] is None
                else round(float(train_output["best_eval_ranking_accuracy"]), 6)
            ),
            "total_mining_time_sec": round(total_mining_time_sec, 4),
            "hidden_triplet_mining_cache_hit": False,
            "hidden_triplet_mining_skipped": False,
            "hidden_triplet_mining_graph_source": "base_graph",
            "hidden_triplet_reproduction_profile": getattr(
                self.global_config,
                "hidden_triplet_reproduction_profile",
                {},
            ),
            "hidden_triplet_config_signature": self.hipporag.get_hidden_triplet_mining_config_signature(),
        }
        self._save_gnn_triple_dump(summary, added_triples, batch_logs)
        logger.info(f"Hidden triplet mining summary: {summary}")
        return summary

    def _cfg_int(self, name: str, default: int) -> int:
        return int(getattr(self.global_config, name, default))

    def _cfg_float(self, name: str, default: float) -> float:
        return float(getattr(self.global_config, name, default))

    def _cfg_bool(self, name: str, default: bool) -> bool:
        value = getattr(self.global_config, name, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"yes", "true", "t", "y", "1"}
        return bool(value)

    def _resolve_subgraph_llm_budget(self, graph_data: Dict[str, Any]) -> int:
        raw_budget = self._cfg_float("hidden_triplet_subgraph_llm_budget", 100.0)

        if raw_budget >= 1.0 and float(raw_budget).is_integer():
            return max(1, int(raw_budget))

        if 0.0 < raw_budget < 1.0:
            node_type_ids = graph_data.get("node_type_ids", [])
            num_chunks = sum(1 for node_type_id in node_type_ids if node_type_id == CHUNK_NODE_TYPE)
            resolved_budget = int(raw_budget * num_chunks * 2.0)
            return max(1, resolved_budget)

        return max(1, int(raw_budget))

    def _resolve_subgraph_inference_root_budget(self, graph_data: Dict[str, Any]) -> int:
        raw_budget = self._cfg_float("hidden_triplet_subgraph_inference_root_budget", 512.0)

        if raw_budget >= 1.0 and float(raw_budget).is_integer():
            return max(1, int(raw_budget))

        if 0.0 < raw_budget < 1.0:
            node_type_ids = graph_data.get("node_type_ids", [])
            num_entity_nodes = sum(1 for node_type_id in node_type_ids if node_type_id == ENTITY_NODE_TYPE)
            resolved_budget = int(raw_budget * num_entity_nodes)
            return max(1, resolved_budget)

        return max(1, int(raw_budget))

    def _resolve_subgraph_root_sample_count(
        self,
        graph_data: Dict[str, Any],
        count_cfg_name: str,
        default_count: int,
        allow_zero: bool = False,
    ) -> int:
        requested_count = self._cfg_int(count_cfg_name, default_count)
        requested_count = max(0, requested_count)
        max_available_roots = len(graph_data.get("root_candidates", []))

        if max_available_roots == 0:
            return 0
        if requested_count == 0:
            return 0 if allow_zero else 1

        resolved_count = requested_count if allow_zero else max(1, requested_count)
        return min(resolved_count, max_available_roots)

    @staticmethod
    def _is_chunk_node(node_key: str) -> bool:
        return isinstance(node_key, str) and node_key.startswith("chunk-")

    def _edge_type_for_full_graph_edge(self, src_name: str, dst_name: str, weight: float) -> Optional[int]:
        src_is_entity = self._is_entity_node(src_name)
        dst_is_entity = self._is_entity_node(dst_name)
        src_is_chunk = self._is_chunk_node(src_name)
        dst_is_chunk = self._is_chunk_node(dst_name)

        if src_is_entity and dst_is_entity:
            if self._is_fact_edge(weight):
                return FACT_EDGE_TYPE
            if self._is_synonym_edge(weight):
                return SYNONYM_EDGE_TYPE
            return None
        if (src_is_entity and dst_is_chunk) or (src_is_chunk and dst_is_entity):
            return ENTITY_CHUNK_EDGE_TYPE
        return None

    def _node_type_id(self, node_name: str) -> int:
        if self._is_entity_node(node_name):
            return ENTITY_NODE_TYPE
        if self._is_chunk_node(node_name):
            return CHUNK_NODE_TYPE
        return OTHER_NODE_TYPE

    def _build_chunk_triple_lookup(self) -> Dict[str, List[Tuple[str, str, str]]]:
        base_openie_docs, _ = self.hipporag.load_existing_openie([])
        chunk_triples: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
        seen_by_chunk: Dict[str, Set[Tuple[str, str, str]]] = defaultdict(set)
        for doc in base_openie_docs:
            chunk_id = doc.get("idx")
            if not isinstance(chunk_id, str):
                continue
            for raw_triple in doc.get("extracted_triples", []):
                if not isinstance(raw_triple, (list, tuple)) or len(raw_triple) != 3:
                    continue
                triple = tuple(text_processing(list(raw_triple)))
                if len(triple) != 3 or triple[0] == "" or triple[2] == "":
                    continue
                if triple in seen_by_chunk[chunk_id]:
                    continue
                seen_by_chunk[chunk_id].add(triple)
                chunk_triples[chunk_id].append(triple)
        return dict(chunk_triples)

    @staticmethod
    def _build_triple_provenance_chunk_lookup(
        chunk_triple_lookup: Dict[str, List[Tuple[str, str, str]]],
    ) -> Dict[Tuple[str, str, str], str]:
        triple_chunks: Dict[Tuple[str, str, str], str] = {}
        for chunk_id, triples in chunk_triple_lookup.items():
            for triple in triples:
                if triple not in triple_chunks:
                    triple_chunks[triple] = chunk_id
        return triple_chunks

    @staticmethod
    def _build_fact_pair_evidence_lookup(
        chunk_triple_lookup: Dict[str, List[Tuple[str, str, str]]],
    ) -> Dict[Tuple[str, str], List[Tuple[Tuple[str, str, str], str]]]:
        triple_provenance_chunk_lookup = GNNHiddenTripletMiner._build_triple_provenance_chunk_lookup(
            chunk_triple_lookup
        )
        pair_evidence: Dict[Tuple[str, str], List[Tuple[Tuple[str, str, str], str]]] = defaultdict(list)
        for triple, chunk_id in sorted(triple_provenance_chunk_lookup.items()):
            pair_key = tuple(sorted((triple[0], triple[2])))
            pair_evidence[pair_key].append((triple, chunk_id))
        return dict(pair_evidence)

    def _extract_full_graph_for_subgraph_mining(self) -> Dict[str, Any]:
        graph = self.hipporag.graph
        node_names = list(graph.vs["name"]) if "name" in graph.vs.attributes() else [str(idx) for idx in range(graph.vcount())]
        node_type_ids = [self._node_type_id(node_name) for node_name in node_names]

        entity_embedding_by_key = {
            node_key: np.asarray(embedding, dtype=np.float32)
            for node_key, embedding in zip(
                list(getattr(self.hipporag, "entity_node_keys", [])),
                np.asarray(getattr(self.hipporag, "entity_embeddings", []), dtype=np.float32),
            )
        }
        chunk_embedding_by_key = {
            node_key: np.asarray(embedding, dtype=np.float32)
            for node_key, embedding in zip(
                list(getattr(self.hipporag, "passage_node_keys", [])),
                np.asarray(getattr(self.hipporag, "passage_embeddings", []), dtype=np.float32),
            )
        }
        embedding_dim = 0
        if entity_embedding_by_key:
            embedding_dim = len(next(iter(entity_embedding_by_key.values())))
        elif chunk_embedding_by_key:
            embedding_dim = len(next(iter(chunk_embedding_by_key.values())))
        if embedding_dim == 0:
            raise ValueError("Cannot build subgraph GNN inputs because no entity/chunk embeddings were loaded.")

        node_features = np.zeros((len(node_names), embedding_dim), dtype=np.float32)
        entity_text_by_node: Dict[str, str] = {}
        for node_idx, node_name in enumerate(node_names):
            if self._is_entity_node(node_name):
                node_features[node_idx] = entity_embedding_by_key.get(node_name, np.zeros(embedding_dim, dtype=np.float32))
                try:
                    entity_text = self.hipporag.get_entity_row(node_name)["content"]
                except Exception:
                    entity_text = node_name
                entity_text_by_node[node_name] = entity_text
            elif self._is_chunk_node(node_name):
                node_features[node_idx] = chunk_embedding_by_key.get(node_name, np.zeros(embedding_dim, dtype=np.float32))

        edge_by_key: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
        for edge in graph.es:
            if "weight" not in edge.attributes():
                continue
            src_idx, dst_idx = edge.tuple
            if src_idx == dst_idx:
                continue
            src_name = node_names[src_idx]
            dst_name = node_names[dst_idx]
            try:
                raw_weight = float(edge["weight"])
            except (TypeError, ValueError):
                raw_weight = 1.0
            if not math.isfinite(raw_weight) or raw_weight <= 0:
                continue
            edge_type = self._edge_type_for_full_graph_edge(src_name, dst_name, raw_weight)
            if edge_type is None:
                continue
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
            # Sum duplicate/reverse-direction weights for random-walk sampling;
            # message passing still only consumes edge_type, not raw edge weight.
            edge_by_key[edge_key]["weight"] += raw_weight

        edge_records = list(edge_by_key.values())
        fact_edge_keys = {record["edge_key"] for record in edge_records if record["edge_type"] == FACT_EDGE_TYPE}
        synonym_edge_keys = {record["edge_key"] for record in edge_records if record["edge_type"] == SYNONYM_EDGE_TYPE}
        entity_chunk_edge_keys = {
            record["edge_key"] for record in edge_records if record["edge_type"] == ENTITY_CHUNK_EDGE_TYPE
        }

        fact_walk_scale = self._cfg_float("hidden_triplet_subgraph_fact_walk_scale", 1.0)
        synonym_walk_scale = self._cfg_float("hidden_triplet_subgraph_synonym_walk_scale", 0.5)
        weight_clip = self._cfg_float("hidden_triplet_subgraph_walk_weight_clip", 5.0)
        edge_type_scale = {
            FACT_EDGE_TYPE: fact_walk_scale,
            SYNONYM_EDGE_TYPE: synonym_walk_scale,
        }
        weighted_adjacency: List[Dict[str, List[float]]] = [
            {"neighbors": [], "weights": []}
            for _ in range(len(node_names))
        ]
        adjacency_accumulator: List[Dict[int, float]] = [defaultdict(float) for _ in node_names]
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
        for node_idx, neighbor_map in enumerate(adjacency_accumulator):
            weighted_adjacency[node_idx] = {
                "neighbors": list(neighbor_map.keys()),
                "weights": list(neighbor_map.values()),
            }

        root_candidates = [
            node_idx
            for node_idx, node_name in enumerate(node_names)
            if self._is_entity_node(node_name)
            and not self._should_filter_candidate_entity_text(entity_text_by_node.get(node_name, ""))
            and len(weighted_adjacency[node_idx]["neighbors"]) > 0
        ]

        edge_records_by_node: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for record in edge_records:
            edge_records_by_node[record["src"]].append(record)
            edge_records_by_node[record["dst"]].append(record)

        chunk_triple_lookup = self._build_chunk_triple_lookup()
        fact_pair_evidence_lookup = self._build_fact_pair_evidence_lookup(chunk_triple_lookup)
        graph_data = {
            "graph": graph,
            "node_names": node_names,
            "node_type_ids": node_type_ids,
            "node_features": node_features,
            "entity_text_by_node": entity_text_by_node,
            "edge_records": edge_records,
            "edge_records_by_node": dict(edge_records_by_node),
            "weighted_adjacency": weighted_adjacency,
            "root_candidates": root_candidates,
            "fact_pair_evidence_lookup": fact_pair_evidence_lookup,
            "num_nodes": len(node_names),
            "num_edges": len(edge_records),
            "num_fact_edges": len(fact_edge_keys),
            "num_synonym_edges": len(synonym_edge_keys),
            "num_entity_chunk_edges": len(entity_chunk_edge_keys),
        }
        logger.info(
            "Prepared full graph for subgraph missingness mining: nodes=%d edges=%d fact_edges=%d "
            "synonym_edges=%d entity_chunk_edges=%d root_candidates=%d",
            graph_data["num_nodes"],
            graph_data["num_edges"],
            graph_data["num_fact_edges"],
            graph_data["num_synonym_edges"],
            graph_data["num_entity_chunk_edges"],
            len(root_candidates),
        )
        return graph_data

    def _sample_weighted_random_walk_nodes(
        self,
        graph_data: Dict[str, Any],
        root_node_idx: int,
        rng: random.Random,
        walks_per_root: Optional[int] = None,
    ) -> List[int]:
        walk_length = max(1, self._cfg_int("hidden_triplet_subgraph_walk_length", 3))
        walks_per_root = max(
            1,
            int(walks_per_root)
            if walks_per_root is not None
            else self._cfg_int("hidden_triplet_subgraph_walks_per_root", 20),
        )
        weighted_adjacency = graph_data["weighted_adjacency"]
        visited = {int(root_node_idx)}

        node_type_ids = graph_data["node_type_ids"]

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

    def _count_fact_edges_in_node_set(self, graph_data: Dict[str, Any], node_indices: Set[int]) -> int:
        return sum(
            1
            for record in graph_data["edge_records"]
            if record["edge_type"] == FACT_EDGE_TYPE
            and record["src"] in node_indices
            and record["dst"] in node_indices
        )

    def _sample_subgraph_record(
        self,
        graph_data: Dict[str, Any],
        root_node_idx: int,
        rng: random.Random,
        subgraph_id: str,
        walks_per_root: Optional[int] = None,
    ) -> Optional[SampledSubgraphRecord]:
        node_indices = self._sample_weighted_random_walk_nodes(
            graph_data,
            root_node_idx,
            rng,
            walks_per_root=walks_per_root,
        )
        return self._build_subgraph_record_from_node_indices(
            graph_data=graph_data,
            root_node_idx=root_node_idx,
            node_indices=node_indices,
            subgraph_id=subgraph_id,
        )

    def _build_subgraph_record_from_node_indices(
        self,
        graph_data: Dict[str, Any],
        root_node_idx: int,
        node_indices: List[int],
        subgraph_id: str,
    ) -> Optional[SampledSubgraphRecord]:
        node_set = set(node_indices)
        min_nodes = max(2, self._cfg_int("hidden_triplet_subgraph_min_nodes", 4))
        min_fact_edges = max(1, self._cfg_int("hidden_triplet_subgraph_min_fact_edges", 1))
        if len(node_indices) < min_nodes:
            return None
        unique_fact_edges = self._count_fact_edges_in_node_set(graph_data, node_set)
        if unique_fact_edges < min_fact_edges:
            return None

        node_names = graph_data["node_names"]
        entity_node_names = sorted(
            node_names[idx]
            for idx in node_indices
            if graph_data["node_type_ids"][idx] == ENTITY_NODE_TYPE
        )
        entity_texts = sorted(
            graph_data["entity_text_by_node"].get(node_name, node_name)
            for node_name in entity_node_names
        )
        chunk_ids = sorted(
            node_names[idx]
            for idx in node_indices
            if graph_data["node_type_ids"][idx] == CHUNK_NODE_TYPE
        )
        root_name = node_names[root_node_idx]
        return SampledSubgraphRecord(
            subgraph_id=subgraph_id,
            root_node_idx=int(root_node_idx),
            root_node_name=root_name,
            root_node_text=graph_data["entity_text_by_node"].get(root_name, root_name),
            node_indices=node_indices,
            entity_node_names=entity_node_names,
            entity_texts=entity_texts,
            chunk_ids=chunk_ids,
            unique_fact_edges=unique_fact_edges,
        )

    def _build_subgraph_data(
        self,
        graph_data: Dict[str, Any],
        node_indices: List[int],
        label: float,
        removed_edge_keys: Optional[Set[Tuple[int, int, int]]] = None,
        removed_node_indices: Optional[Set[int]] = None,
        corruption_type: int = 0,
    ) -> Data:
        removed_edge_keys = removed_edge_keys or set()
        removed_node_indices = removed_node_indices or set()
        active_node_indices = sorted(idx for idx in node_indices if idx not in removed_node_indices)
        local_idx = {global_idx: local for local, global_idx in enumerate(active_node_indices)}
        src_nodes = []
        dst_nodes = []
        edge_types = []
        for record in graph_data["edge_records"]:
            if record["edge_key"] in removed_edge_keys:
                continue
            if record["src"] not in local_idx or record["dst"] not in local_idx:
                continue
            src_local = local_idx[record["src"]]
            dst_local = local_idx[record["dst"]]
            src_nodes.extend([src_local, dst_local])
            dst_nodes.extend([dst_local, src_local])
            edge_types.extend([record["edge_type"], record["edge_type"]])

        if src_nodes:
            edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
            edge_type = torch.tensor(edge_types, dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_type = torch.empty((0,), dtype=torch.long)

        node_features = torch.tensor(graph_data["node_features"][active_node_indices], dtype=torch.float32)
        node_type = torch.tensor(
            [graph_data["node_type_ids"][idx] for idx in active_node_indices],
            dtype=torch.long,
        )
        return Data(
            x=node_features,
            edge_index=edge_index,
            edge_type=edge_type,
            node_type=node_type,
            y=torch.tensor([float(label)], dtype=torch.float32),
            corruption_type=torch.tensor([int(corruption_type)], dtype=torch.long),
            global_node_indices=torch.tensor(active_node_indices, dtype=torch.long),
        )

    def _fact_edge_keys_in_subgraph(
        self,
        graph_data: Dict[str, Any],
        node_indices: List[int],
    ) -> List[Tuple[int, int, int]]:
        node_set = set(node_indices)
        return [
            record["edge_key"]
            for record in graph_data["edge_records"]
            if record["edge_type"] == FACT_EDGE_TYPE
            and record["src"] in node_set
            and record["dst"] in node_set
        ]

    def _entity_delete_candidates(
        self,
        graph_data: Dict[str, Any],
        node_indices: List[int],
        root_node_idx: int,
    ) -> List[int]:
        node_set = set(node_indices)
        candidates = []
        for node_idx in node_indices:
            if node_idx == root_node_idx:
                continue
            if graph_data["node_type_ids"][node_idx] != ENTITY_NODE_TYPE:
                continue
            has_fact = False
            for record in graph_data["edge_records_by_node"].get(node_idx, []):
                other_idx = record["dst"] if record["src"] == node_idx else record["src"]
                if other_idx not in node_set:
                    continue
                if record["edge_type"] == FACT_EDGE_TYPE:
                    has_fact = True
            if has_fact:
                candidates.append(node_idx)
        return candidates

    def _build_training_views_for_subgraph(
        self,
        graph_data: Dict[str, Any],
        clean_record: SampledSubgraphRecord,
        corruption_record: SampledSubgraphRecord,
        rng: random.Random,
    ) -> Dict[str, Any]:
        full_data = self._build_subgraph_data(
            graph_data=graph_data,
            node_indices=clean_record.node_indices,
            label=0.0,
            corruption_type=0,
        )
        views = {"full": full_data, "corrupted": []}

        fact_edge_keys = self._fact_edge_keys_in_subgraph(graph_data, corruption_record.node_indices)
        min_fact_edges = max(1, self._cfg_int("hidden_triplet_subgraph_min_fact_edges", 1))
        max_removable_fact_edges = max(0, len(fact_edge_keys) - min_fact_edges)
        if max_removable_fact_edges > 0:
            mask_ratio = min(max(self._cfg_float("hidden_triplet_subgraph_fact_mask_ratio", 0.2), 0.0), 1.0)
            mask_count = min(max_removable_fact_edges, max(1, int(round(len(fact_edge_keys) * mask_ratio))))
            removed_fact_edges = set(rng.sample(fact_edge_keys, mask_count))
            views["corrupted"].append(
                self._build_subgraph_data(
                    graph_data=graph_data,
                    node_indices=corruption_record.node_indices,
                    label=1.0,
                    removed_edge_keys=removed_fact_edges,
                    corruption_type=1,
                )
            )

        entity_candidates = self._entity_delete_candidates(
            graph_data,
            corruption_record.node_indices,
            corruption_record.root_node_idx,
        )
        if entity_candidates:
            delete_ratio = min(max(self._cfg_float("hidden_triplet_subgraph_entity_delete_ratio", 0.08), 0.0), 1.0)
            delete_count = max(1, int(round(len(entity_candidates) * delete_ratio)))
            delete_count = min(delete_count, len(entity_candidates))
            deleted_entities = set(rng.sample(entity_candidates, delete_count))
            remaining_nodes = set(corruption_record.node_indices) - deleted_entities
            if self._count_fact_edges_in_node_set(graph_data, remaining_nodes) >= min_fact_edges:
                views["corrupted"].append(
                    self._build_subgraph_data(
                        graph_data=graph_data,
                        node_indices=corruption_record.node_indices,
                        label=1.0,
                        removed_node_indices=deleted_entities,
                        corruption_type=2,
                    )
                )

        return views

    def _sample_training_groups(
        self,
        graph_data: Dict[str, Any],
        rng: random.Random,
        num_roots: int,
        prefix: str,
    ) -> List[Dict[str, Any]]:
        root_candidates = graph_data["root_candidates"][:]
        if not root_candidates:
            raise ValueError("No valid entity root candidates are available for subgraph GNN training.")

        groups = []
        attempts = 0
        max_attempts = max(num_roots * 20, 100)
        inference_walks_per_root = max(1, self._cfg_int("hidden_triplet_subgraph_walks_per_root", 20))
        negative_walks_per_root = max(
            inference_walks_per_root,
            self._cfg_int("hidden_triplet_subgraph_negative_walks_per_root", inference_walks_per_root * 2),
        )
        while len(groups) < num_roots and attempts < max_attempts:
            attempts += 1
            root_idx = rng.choice(root_candidates)
            clean_record = self._sample_subgraph_record(
                graph_data=graph_data,
                root_node_idx=root_idx,
                rng=rng,
                subgraph_id=f"{prefix}-{attempts}-negative-sampled",
                walks_per_root=negative_walks_per_root,
            )
            if clean_record is None:
                continue
            corruption_record = self._sample_subgraph_record(
                graph_data=graph_data,
                root_node_idx=root_idx,
                rng=rng,
                subgraph_id=f"{prefix}-{attempts}-sampled",
                walks_per_root=inference_walks_per_root,
            )
            if corruption_record is None:
                continue
            views = self._build_training_views_for_subgraph(
                graph_data=graph_data,
                clean_record=clean_record,
                corruption_record=corruption_record,
                rng=rng,
            )
            if not views["corrupted"]:
                continue
            groups.append(
                {
                    "clean_record": clean_record,
                    "corruption_record": corruption_record,
                    **views,
                }
            )
        if len(groups) == 0:
            raise ValueError("Failed to construct any valid subgraph missingness training examples.")
        return groups

    @staticmethod
    def _flatten_subgraph_groups(groups: List[Dict[str, Any]]) -> Tuple[List[Data], List[Tuple[int, int]]]:
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

    def _evaluate_subgraph_missingness_model(
        self,
        model: PyGSubgraphMissingnessClassifier,
        groups: List[Dict[str, Any]],
    ) -> Dict[str, Optional[float]]:
        if not groups:
            return {
                "eval_loss": None,
                "eval_accuracy": None,
                "eval_f1": None,
                "eval_ranking_accuracy": None,
            }

        data_list, ranking_pairs = self._flatten_subgraph_groups(groups)
        batch = Batch.from_data_list(data_list).to(self.device)
        was_training = model.training
        model.eval()
        with torch.no_grad():
            logits = model(batch)
            labels = batch.y.view(-1).to(self.device)
            eval_loss = F.binary_cross_entropy_with_logits(logits, labels).item()
            probs = torch.sigmoid(logits)
            preds = probs >= 0.5
            target = labels >= 0.5
            accuracy = (preds == target).float().mean().item()
            tp = (preds & target).sum().item()
            fp = (preds & ~target).sum().item()
            fn = (~preds & target).sum().item()
            precision = tp / max(1, tp + fp)
            recall = tp / max(1, tp + fn)
            f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
            if ranking_pairs:
                ranking_correct = [
                    float(logits[corrupted_idx] > logits[full_idx])
                    for full_idx, corrupted_idx in ranking_pairs
                ]
                ranking_accuracy = sum(ranking_correct) / len(ranking_correct)
            else:
                ranking_accuracy = None
        if was_training:
            model.train()
        return {
            "eval_loss": eval_loss,
            "eval_accuracy": accuracy,
            "eval_f1": f1,
            "eval_ranking_accuracy": ranking_accuracy,
        }

    def _train_subgraph_missingness_detector(self, graph_data: Dict[str, Any]) -> Dict[str, Any]:
        train_start_time = time.perf_counter()
        rng_seed = self.global_config.seed if self.global_config.seed is not None else 42
        self._set_gnn_training_seed(int(rng_seed))
        rng = random.Random(rng_seed)

        model = PyGSubgraphMissingnessClassifier(
            input_dim=int(graph_data["node_features"].shape[1]),
            hidden_dim=int(self.global_config.hidden_triplet_gnn_hidden_dim),
            encoder_type=self.global_config.hidden_triplet_gnn_encoder_type,
        ).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(self.global_config.hidden_triplet_learning_rate))
        num_epochs = max(1, int(self.global_config.hidden_triplet_gnn_epochs))
        roots_per_epoch = self._resolve_subgraph_root_sample_count(
            graph_data=graph_data,
            count_cfg_name="hidden_triplet_subgraph_train_roots_per_epoch",
            default_count=128,
            allow_zero=False,
        )
        eval_roots = self._resolve_subgraph_root_sample_count(
            graph_data=graph_data,
            count_cfg_name="hidden_triplet_subgraph_eval_roots",
            default_count=64,
            allow_zero=True,
        )
        graph_refresh_interval = max(1, int(self.global_config.hidden_triplet_graph_refresh_interval))
        early_stopping_patience = max(0, int(self.global_config.hidden_triplet_early_stopping_patience))
        early_stopping_min_delta = max(0.0, float(self.global_config.hidden_triplet_early_stopping_min_delta))
        early_stopping_enabled = early_stopping_patience > 0 and eval_roots > 0

        eval_groups = (
            self._sample_training_groups(
                graph_data=graph_data,
                rng=random.Random(rng_seed + 104729),
                num_roots=eval_roots,
                prefix="eval",
            )
            if eval_roots > 0
            else []
        )

        training_logs: List[Dict[str, Any]] = []
        last_loss = None
        last_eval_metrics = {
            "eval_loss": None,
            "eval_accuracy": None,
            "eval_f1": None,
            "eval_ranking_accuracy": None,
        }
        completed_epochs = 0
        stopped_early = False
        best_epoch = None
        best_eval_loss = None
        best_eval_ranking_accuracy = None
        best_state_dict = None

        for epoch_idx in range(num_epochs):
            model.train()
            train_groups = self._sample_training_groups(
                graph_data=graph_data,
                rng=rng,
                num_roots=roots_per_epoch,
                prefix=f"train-e{epoch_idx + 1}",
            )
            data_list, _ = self._flatten_subgraph_groups(train_groups)
            batch = Batch.from_data_list(data_list).to(self.device)
            logits = model(batch)
            labels = batch.y.view(-1).to(self.device)
            bce_loss = F.binary_cross_entropy_with_logits(logits, labels)
            loss = bce_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            last_loss = float(loss.detach().item())
            completed_epochs = epoch_idx + 1
            should_log_epoch = completed_epochs % graph_refresh_interval == 0 or completed_epochs == num_epochs
            if should_log_epoch:
                last_eval_metrics = self._evaluate_subgraph_missingness_model(model, eval_groups)
                current_eval_loss = last_eval_metrics["eval_loss"]
                current_ranking = last_eval_metrics["eval_ranking_accuracy"]
                is_best = False
                if current_eval_loss is not None:
                    improved = (
                        best_eval_loss is None
                        or current_eval_loss < best_eval_loss - early_stopping_min_delta
                    )
                    if improved:
                        best_eval_loss = current_eval_loss
                        best_eval_ranking_accuracy = current_ranking
                        best_epoch = completed_epochs
                        best_state_dict = {
                            key: value.detach().cpu().clone()
                            for key, value in model.state_dict().items()
                        }
                        is_best = True

                log_entry = {
                    "epoch": completed_epochs,
                    "total_epochs": num_epochs,
                    "train_loss": last_loss,
                    "bce_loss": float(bce_loss.detach().item()),
                    "num_train_groups": len(train_groups),
                    "num_train_graph_views": len(data_list),
                    "eval_loss": last_eval_metrics["eval_loss"],
                    "eval_accuracy": last_eval_metrics["eval_accuracy"],
                    "eval_f1": last_eval_metrics["eval_f1"],
                    "eval_ranking_accuracy": current_ranking,
                    "is_best_checkpoint": is_best,
                    "best_epoch": best_epoch,
                    "best_eval_loss": best_eval_loss,
                    "best_eval_ranking_accuracy": best_eval_ranking_accuracy,
                    "early_stopped": False,
                }
                training_logs.append(log_entry)
                logger.info(
                    "Subgraph GNN epoch %d/%d: train_loss=%.6f bce=%.6f "
                    "eval_loss=%s eval_accuracy=%s eval_f1=%s eval_ranking_accuracy=%s",
                    completed_epochs,
                    num_epochs,
                    last_loss,
                    float(bce_loss.detach().item()),
                    "None" if last_eval_metrics["eval_loss"] is None else f"{last_eval_metrics['eval_loss']:.6f}",
                    "None" if last_eval_metrics["eval_accuracy"] is None else f"{last_eval_metrics['eval_accuracy']:.4f}",
                    "None" if last_eval_metrics["eval_f1"] is None else f"{last_eval_metrics['eval_f1']:.4f}",
                    "None" if current_ranking is None else f"{current_ranking:.4f}",
                )

                if (
                    early_stopping_enabled
                    and best_epoch is not None
                    and completed_epochs - best_epoch >= early_stopping_patience
                ):
                    stopped_early = True
                    log_entry["early_stopped"] = True
                    logger.info(
                        "Early stopping subgraph GNN training at epoch %d. Best epoch=%d best_eval_loss=%.6f",
                        completed_epochs,
                        best_epoch,
                        best_eval_loss,
                    )
                    break

        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)
            last_eval_metrics = self._evaluate_subgraph_missingness_model(model, eval_groups)

        training_time_sec = time.perf_counter() - train_start_time
        logger.info(
            "Finished subgraph missingness GNN training in %.4f seconds. train_roots_per_epoch=%d eval_roots=%d "
            "train_loss=%.6f eval_loss=%s eval_accuracy=%s eval_ranking_accuracy=%s",
            training_time_sec,
            roots_per_epoch,
            eval_roots,
            0.0 if last_loss is None else last_loss,
            "None" if last_eval_metrics["eval_loss"] is None else f"{last_eval_metrics['eval_loss']:.6f}",
            "None" if last_eval_metrics["eval_accuracy"] is None else f"{last_eval_metrics['eval_accuracy']:.4f}",
            "None" if last_eval_metrics["eval_ranking_accuracy"] is None else f"{last_eval_metrics['eval_ranking_accuracy']:.4f}",
        )
        return {
            "model": model,
            "training_time_sec": training_time_sec,
            "resolved_train_roots_per_epoch": roots_per_epoch,
            "resolved_eval_roots": eval_roots,
            "train_loss": 0.0 if last_loss is None else last_loss,
            "eval_loss": last_eval_metrics["eval_loss"],
            "best_eval_loss": best_eval_loss,
            "eval_accuracy": last_eval_metrics["eval_accuracy"],
            "eval_f1": last_eval_metrics["eval_f1"],
            "eval_ranking_accuracy": last_eval_metrics["eval_ranking_accuracy"],
            "completed_epochs": completed_epochs,
            "early_stopped": stopped_early,
            "best_epoch": best_epoch,
            "best_eval_ranking_accuracy": best_eval_ranking_accuracy,
            "training_logs": training_logs,
        }

    def _entity_hop_distances_from_root(
        self,
        graph_data: Dict[str, Any],
        root_idx: int,
        hop_k: int,
    ) -> Dict[int, int]:
        weighted_adjacency = graph_data["weighted_adjacency"]
        node_type_ids = graph_data["node_type_ids"]

        best_entity_hops = {int(root_idx): 0}
        frontier = [(int(root_idx), 0)]
        while frontier:
            next_frontier: List[Tuple[int, int]] = []
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
    def _node_jaccard(left_nodes: List[int], right_nodes: List[int]) -> float:
        left = set(left_nodes)
        right = set(right_nodes)
        if not left and not right:
            return 1.0
        return len(left & right) / max(1, len(left | right))

    @staticmethod
    def _summarize_record_missing_scores(records: List[SampledSubgraphRecord]) -> Dict[str, Any]:
        scores = [float(record.missing_score) for record in records]
        if not scores:
            return {
                "count": 0,
                "avg": None,
                "min": None,
                "max": None,
            }
        return {
            "count": len(scores),
            "avg": round(sum(scores) / len(scores), 6),
            "min": round(min(scores), 6),
            "max": round(max(scores), 6),
        }

    def _collect_subgraph_evidence(
        self,
        graph_data: Dict[str, Any],
        record: SampledSubgraphRecord,
    ) -> SampledSubgraphRecord:
        node_names = graph_data["node_names"]
        entity_node_names = sorted(
            node_names[idx]
            for idx in record.node_indices
            if graph_data["node_type_ids"][idx] == ENTITY_NODE_TYPE
        )
        entity_texts = sorted(
            graph_data["entity_text_by_node"].get(node_name, node_name)
            for node_name in entity_node_names
        )
        max_known = max(1, self._cfg_int("hidden_triplet_subgraph_max_known_triples_per_prompt", 80))
        max_chunks = max(1, self._cfg_int("hidden_triplet_subgraph_max_chunks_per_prompt", 6))
        entity_norms = {text_processing(entity_text) for entity_text in entity_texts}

        fact_triples: List[Tuple[str, str, str]] = []
        seen_triples = set()
        chunk_ids: List[str] = []
        seen_chunk_ids: Set[str] = set()
        node_set = set(record.node_indices)
        for edge_record in graph_data["edge_records"]:
            if edge_record["edge_type"] != FACT_EDGE_TYPE:
                continue
            if edge_record["src"] not in node_set or edge_record["dst"] not in node_set:
                continue
            left_name = node_names[edge_record["src"]]
            right_name = node_names[edge_record["dst"]]
            left_text = text_processing(graph_data["entity_text_by_node"].get(left_name, left_name))
            right_text = text_processing(graph_data["entity_text_by_node"].get(right_name, right_name))
            pair_key = tuple(sorted((left_text, right_text)))
            for triple, chunk_id in graph_data["fact_pair_evidence_lookup"].get(pair_key, []):
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

        known_triples = [triple for triple in fact_triples if triple[0] in entity_norms and triple[2] in entity_norms]
        if len(known_triples) == 0:
            known_triples = fact_triples

        record.entity_node_names = entity_node_names
        record.entity_texts = entity_texts
        record.chunk_ids = chunk_ids
        record.known_triples = known_triples[:max_known]
        return record

    def _score_subgraph_records(
        self,
        graph_data: Dict[str, Any],
        model: PyGSubgraphMissingnessClassifier,
        records: List[SampledSubgraphRecord],
    ) -> List[SampledSubgraphRecord]:
        if not records:
            return []
        data_list = [
            self._build_subgraph_data(
                graph_data=graph_data,
                node_indices=record.node_indices,
                label=0.0,
                corruption_type=0,
            )
            for record in records
        ]
        batch = Batch.from_data_list(data_list).to(self.device)
        model.eval()
        with torch.no_grad():
            scores = torch.sigmoid(model(batch)).detach().cpu().numpy().tolist()
        for record, score in zip(records, scores):
            record.missing_score = float(score)
        return records

    def _select_missing_subgraphs_for_llm(
        self,
        graph_data: Dict[str, Any],
        model: PyGSubgraphMissingnessClassifier,
    ) -> List[SampledSubgraphRecord]:
        rng_seed = self.global_config.seed if self.global_config.seed is not None else 42
        rng = random.Random(int(rng_seed) + 99991)
        root_candidates = graph_data["root_candidates"][:]
        rng.shuffle(root_candidates)
        root_budget = min(
            self._resolve_subgraph_inference_root_budget(graph_data),
            max(1, len(root_candidates)),
        )
        missing_threshold = self._cfg_float("hidden_triplet_subgraph_missing_score_threshold", 0.7)
        llm_budget = self._resolve_subgraph_llm_budget(graph_data)
        overlap_threshold = min(max(self._cfg_float("hidden_triplet_subgraph_overlap_threshold", 0.5), 0.0), 1.0)
        root_selection_hops = max(1, self._cfg_int("hidden_triplet_subgraph_walk_length", 3))

        sampled_records: List[SampledSubgraphRecord] = []
        remaining_roots = set(root_candidates)
        current_root_idx: Optional[int] = None
        attempts = 0
        jump_fallbacks = 0
        while remaining_roots and len(sampled_records) < root_budget:
            next_root_idx = None
            if current_root_idx is not None:
                root_distance_map = self._entity_hop_distances_from_root(
                    graph_data=graph_data,
                    root_idx=current_root_idx,
                    hop_k=root_selection_hops,
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
            record = self._sample_subgraph_record(
                graph_data=graph_data,
                root_node_idx=current_root_idx,
                rng=rng,
                subgraph_id=f"infer-{len(sampled_records)}",
            )
            if record is None:
                continue
            sampled_records.append(record)

        scored_records = self._score_subgraph_records(graph_data, model, sampled_records)
        scored_records.sort(key=lambda item: item.missing_score, reverse=True)
        selected_records: List[SampledSubgraphRecord] = []
        for record in scored_records:
            if record.missing_score < missing_threshold:
                continue
            if any(self._node_jaccard(record.node_indices, selected.node_indices) > overlap_threshold for selected in selected_records):
                continue
            record_with_evidence = self._collect_subgraph_evidence(graph_data, record)
            if len(record_with_evidence.chunk_ids) == 0:
                continue
            selected_records.append(record_with_evidence)
            if len(selected_records) >= llm_budget:
                break

        selected_subgraph_ids = {record.subgraph_id for record in selected_records}
        unselected_records = [
            record
            for record in scored_records
            if record.subgraph_id not in selected_subgraph_ids
        ]
        scored_score_stats = self._summarize_record_missing_scores(scored_records)
        selected_score_stats = self._summarize_record_missing_scores(selected_records)
        unselected_score_stats = self._summarize_record_missing_scores(unselected_records)
        graph_data["hidden_triplet_subgraph_selection_stats"] = {
            "num_inference_roots_attempted": attempts,
            "root_selection_hops": root_selection_hops,
            "num_root_jump_fallbacks": jump_fallbacks,
            "num_sampled_subgraphs": len(sampled_records),
            "num_scored_subgraphs": len(scored_records),
            "num_selected_subgraphs_for_llm": len(selected_records),
            "num_unselected_subgraphs_for_llm": len(unselected_records),
            "missing_score_threshold": missing_threshold,
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
        logger.info(
            "Selected %d/%d scored subgraphs for LLM completion. threshold=%.4f llm_budget=%d",
            len(selected_records),
            len(scored_records),
            missing_threshold,
            llm_budget,
        )
        return selected_records

    def _extract_hidden_triples_from_subgraphs(
        self,
        subgraphs: List[SampledSubgraphRecord],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        if len(subgraphs) == 0:
            return [], []

        batches = [
            SubgraphCompletionBatch(
                subgraph=subgraph,
                chunk_ids=list(subgraph.chunk_ids),
                known_triples=list(subgraph.known_triples),
            )
            for subgraph in subgraphs
            if len(subgraph.chunk_ids) > 0
        ]
        if len(batches) == 0:
            return [], []

        max_workers = max(1, self._cfg_int("llm_concurrency", 32))
        mined_triples: List[Dict[str, Any]] = []
        batch_logs_by_idx: Dict[int, Dict[str, Any]] = {}
        logger.info("Extracting hidden triples from %d high-missingness subgraphs with LLM.", len(batches))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_batch_idx = {
                executor.submit(self._process_subgraph_completion_batch, batch): batch_idx
                for batch_idx, batch in enumerate(batches)
            }
            with logging_redirect_tqdm():
                for future in tqdm(
                    as_completed(future_to_batch_idx),
                    total=len(future_to_batch_idx),
                    desc="Completing missing subgraphs",
                    position=0,
                    dynamic_ncols=True,
                ):
                    batch_idx = future_to_batch_idx[future]
                    batch = batches[batch_idx]
                    try:
                        triples, batch_log = future.result()
                    except Exception as exc:
                        logger.exception(
                            "Subgraph hidden-triplet completion failed for batch %d; continuing.",
                            batch_idx,
                        )
                        triples = []
                        batch_log = {
                            "subgraph_id": batch.subgraph.subgraph_id,
                            "root_node_text": batch.subgraph.root_node_text,
                            "missing_score": batch.subgraph.missing_score,
                            "chunk_ids": batch.chunk_ids,
                            "num_valid_triples": 0,
                            "failed": True,
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                        }
                    mined_triples.extend(triples)
                    batch_logs_by_idx[batch_idx] = batch_log

        unique_triples: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        for triple_item in mined_triples:
            triple = tuple(triple_item["triple"])
            if triple not in unique_triples:
                unique_triples[triple] = triple_item
            else:
                unique_triples[triple]["chunk_ids"] = sorted(
                    set(unique_triples[triple]["chunk_ids"]) | set(triple_item["chunk_ids"])
                )
                unique_triples[triple]["score"] = max(unique_triples[triple]["score"], triple_item["score"])

        batch_logs = [
            batch_logs_by_idx[idx]
            for idx in range(len(batches))
            if idx in batch_logs_by_idx
        ]
        return list(unique_triples.values()), batch_logs

    def _process_subgraph_completion_batch(
        self,
        batch: SubgraphCompletionBatch,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        messages = self._build_subgraph_completion_messages(batch)
        raw_response, metadata = self._infer(messages)
        parsed_results = self._parse_subgraph_completion_response(raw_response)
        valid_triples = self._collect_valid_triples_from_subgraph_results(parsed_results, batch)
        batch_log = {
            "subgraph_id": batch.subgraph.subgraph_id,
            "root_node_name": batch.subgraph.root_node_name,
            "root_node_text": batch.subgraph.root_node_text,
            "missing_score": round(float(batch.subgraph.missing_score), 6),
            "num_subgraph_nodes": len(batch.subgraph.node_indices),
            "num_entity_nodes": len(batch.subgraph.entity_node_names),
            "num_chunks": len(batch.chunk_ids),
            "num_known_triples": len(batch.known_triples),
            "chunk_ids": batch.chunk_ids,
            "messages_used_for_llm": messages,
            "raw_response": raw_response,
            "parsed_results": parsed_results,
            "metadata": metadata,
            "num_valid_triples": len(valid_triples),
        }
        return valid_triples, batch_log

    def _build_subgraph_completion_messages(self, batch: SubgraphCompletionBatch) -> List[Dict[str, str]]:
        passages = []
        for chunk_id in batch.chunk_ids:
            passage = self.hipporag.chunk_embedding_store.get_row(chunk_id)["content"]
            passages.append(
                {
                    "chunk_id": chunk_id,
                    "passage": passage[: self.global_config.hidden_triplet_max_chars_per_chunk],
                }
            )

        known_triples = [list(triple) for triple in batch.known_triples]
        entity_list = batch.subgraph.entity_texts[: self._cfg_int("hidden_triplet_subgraph_max_entities_per_prompt", 80)]
        incompleteness_hint = {
            "gnn_missing_score": round(float(batch.subgraph.missing_score), 6),
            "possible_missingness_types": [
                "missing_relations",
                "missing_entities",
                "missing_relations_and_entities",
            ],
            "instruction": (
                "Treat this sampled subgraph as potentially incomplete. Look for important gaps in both "
                "relations and entities, not just shallow restatements of the known triples."
            ),
        }
        one_shot_root_entity = "Acme Labs"
        one_shot_entities = [
            "Acme Labs",
            "Lina Chen",
            "Aurora diagnostics platform",
            "Northwind Health",
            "March 2023",
        ]
        one_shot_known_triples = [
            ["Lina Chen", "founded", "Acme Labs"],
            ["Acme Labs", "developed", "Aurora diagnostics platform"],
        ]
        one_shot_passages = [
            {
                "chunk_id": "chunk-acme-1",
                "passage": (
                    "Lina Chen founded Acme Labs in 2021. Acme Labs developed the Aurora diagnostics platform."
                ),
            },
            {
                "chunk_id": "chunk-acme-2",
                "passage": (
                    "Northwind Health acquired Acme Labs in March 2023. After the deal, it integrated the platform "
                    "into its clinical workflow suite."
                ),
            },
            {
                "chunk_id": "chunk-acme-3",
                "passage": (
                    "Hospitals already using Northwind's analytics stack could deploy Aurora only after that "
                    "integration was completed."
                ),
            },
        ]
        one_shot_response = {
            "new_triples": [
                {
                    "triple": ["Acme Labs", "founded in", "2021"],
                    "chunk_ids": ["chunk-acme-1"],
                },
                {
                    "triple": ["Northwind Health", "acquired", "Acme Labs"],
                    "chunk_ids": ["chunk-acme-2"],
                },
                {
                    "triple": ["Northwind Health", "acquired in", "March 2023"],
                    "chunk_ids": ["chunk-acme-2"],
                },
                {
                    "triple": ["Northwind Health", "integrated", "Aurora diagnostics platform"],
                    "chunk_ids": ["chunk-acme-1", "chunk-acme-2"],
                },
                {
                    "triple": [
                        "Aurora diagnostics platform",
                        "deployed only after",
                        "integration into Northwind Health clinical workflow suite",
                    ],
                    "chunk_ids": ["chunk-acme-2", "chunk-acme-3"],
                },
            ]
        }
        system_prompt = (
            "Your task is to complete a local RDF graph using the provided evidence passages. "
            "You are given a sampled subgraph that a GNN marked as likely incomplete. The incompleteness may come "
            "from missing relation edges, missing entity nodes, or both. You are also given the entities already "
            "present in that subgraph and known triples already extracted from the same local evidence. "
            "Before deciding what to add, silently summarize what the current subgraph already says, identify the "
            "most important missing entities and relations, and then recover the highest-value missing facts. "
            "Do not stop at shallow supplementation. Use the passages and the existing triples together to infer "
            "important but omitted facts that are directly stated or unambiguously supported, including temporal, "
            "conditional, prerequisite, role, ownership, part-whole, sequence, and other structurally important "
            "relations when the evidence clearly supports them. Prefer completions that require connecting multiple "
            "known triples or resolving information across multiple passages when that produces a more important and "
            "better-grounded fact. You should also propose high-value summary triples when they compactly capture the "
            "main role, status, outcome, function, or relationship pattern expressed by the current subgraph and its "
            "evidence passages. These summary triples must still be grounded in the provided evidence and should be "
            "useful for downstream retrieval, not vague restatements. "
            "Return only new, durable, retrieval-useful triples. Do not repeat any known_triples and do not merely "
            "rephrase them. Avoid weak co-occurrence, temporary scene interactions, speculative inferences, "
            "or vague predicates like 'related to'. "
            "Each returned triple must include at least one chunk_id copied exactly from the provided passages. If a "
            "triple depends on evidence spread across multiple passages, include all relevant chunk_ids. "
            "Prefer existing entity strings when possible, but you may introduce a new entity only if it is explicitly "
            "named in the evidence passages and the triple meaningfully fills a gap in the subgraph. "
            "Return JSON only with the key 'new_triples'."
        )
        one_shot_user_prompt = (
            f"Subgraph incompleteness hint:\n"
            f"{json.dumps({'gnn_missing_score': 0.91, 'possible_missingness_types': ['missing_relations', 'missing_entities']}, ensure_ascii=False, indent=2)}\n\n"
            f"Subgraph root entity:\n{json.dumps(one_shot_root_entity, ensure_ascii=False)}\n\n"
            f"Existing entities in the sampled subgraph:\n{json.dumps(one_shot_entities, ensure_ascii=False, indent=2)}\n\n"
            f"Known triples in this sampled subgraph:\n{json.dumps(one_shot_known_triples, ensure_ascii=False, indent=2)}\n\n"
            f"Evidence passages:\n{json.dumps(one_shot_passages, ensure_ascii=False, indent=2)}\n\n"
            "Required output schema:\n"
            "{\n"
            '  "new_triples": [\n'
            "    {\n"
            '      "triple": ["subject", "predicate", "object"],\n'
            '      "chunk_ids": ["chunk-id"]\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Return JSON only."
        )
        user_prompt = (
            f"Subgraph incompleteness hint:\n{json.dumps(incompleteness_hint, ensure_ascii=False, indent=2)}\n\n"
            f"Subgraph root entity:\n{json.dumps(batch.subgraph.root_node_text, ensure_ascii=False)}\n\n"
            f"Existing entities in the sampled subgraph:\n{json.dumps(entity_list, ensure_ascii=False, indent=2)}\n\n"
            f"Known triples in this sampled subgraph:\n{json.dumps(known_triples, ensure_ascii=False, indent=2)}\n\n"
            f"Evidence passages:\n{json.dumps(passages, ensure_ascii=False, indent=2)}\n\n"
            "Silently follow this extraction strategy before producing the final JSON:\n"
            "1. Summarize the current subgraph coverage.\n"
            "2. Identify likely missing relations, likely missing entities, or both.\n"
            "3. Prioritize deeper completions that connect multiple known triples, resolve references across "
            "multiple passages, or enrich the subgraph with important time, condition, prerequisite, ownership, "
            "role, sequence, or part-whole information.\n"
            "4. When a new triple depends on multiple passages, attach all relevant chunk_ids instead of citing only "
            "one chunk.\n"
            "5. Keep only facts that are directly stated or unambiguously supported by the evidence passages.\n\n"
            "Required output schema:\n"
            "{\n"
            '  "new_triples": [\n'
            "    {\n"
            '      "triple": ["subject", "predicate", "object"],\n'
            '      "chunk_ids": ["chunk-id"]\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Return JSON only."
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": one_shot_user_prompt},
            {"role": "assistant", "content": json.dumps(one_shot_response, ensure_ascii=False, indent=2)},
            {"role": "user", "content": user_prompt},
        ]

    def _parse_subgraph_completion_response(self, raw_response: str) -> List[Dict[str, Any]]:
        json_match = re.search(r"\{.*\}", raw_response, re.DOTALL)
        if json_match is None:
            return []
        json_str = fix_broken_generated_json(json_match.group(0))
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError:
            return []
        triples = parsed.get("new_triples", [])
        if not isinstance(triples, list):
            return []
        normalized_items = []
        for item in triples:
            if isinstance(item, dict):
                triple_value = item.get("triple", [])
                chunk_ids = item.get("chunk_ids", [])
            else:
                triple_value = item
                chunk_ids = []
            filtered_triples = filter_invalid_triples([triple_value])
            if len(filtered_triples) == 0:
                continue
            if isinstance(chunk_ids, str):
                chunk_ids = [chunk_ids]
            elif not isinstance(chunk_ids, list):
                chunk_ids = []
            normalized_items.append(
                {
                    "triple": filtered_triples[0],
                    "chunk_ids": [
                        chunk_id
                        for chunk_id in chunk_ids
                        if isinstance(chunk_id, str) and chunk_id.strip() != ""
                    ],
                }
            )
        return normalized_items

    def _collect_valid_triples_from_subgraph_results(
        self,
        parsed_results: List[Dict[str, Any]],
        batch: SubgraphCompletionBatch,
    ) -> List[Dict[str, Any]]:
        allowed_chunk_ids = set(batch.chunk_ids)
        chunk_id_alias_map = self._build_chunk_id_alias_map(batch.chunk_ids)
        known_triples = {tuple(text_processing(list(triple))) for triple in batch.known_triples}
        valid_triples = []
        for item in parsed_results:
            triple = item.get("triple", [])
            proc_triple = tuple(text_processing(list(triple)))
            if len(proc_triple) != 3 or proc_triple[0] == "" or proc_triple[2] == "" or proc_triple[0] == proc_triple[2]:
                continue
            if proc_triple in known_triples:
                continue
            cited_chunk_ids = self._resolve_model_chunk_ids(
                raw_chunk_ids=item.get("chunk_ids", []),
                allowed_chunk_ids=allowed_chunk_ids,
                chunk_id_alias_map=chunk_id_alias_map,
            )
            if len(cited_chunk_ids) == 0:
                if len(batch.chunk_ids) == 1:
                    cited_chunk_ids = list(batch.chunk_ids)
                else:
                    continue
            valid_triples.append(
                {
                    "triple": proc_triple,
                    "pair_names": (proc_triple[0], proc_triple[2]),
                    "chunk_ids": cited_chunk_ids,
                    "score": float(batch.subgraph.missing_score),
                }
            )
        return valid_triples

    def _save_subgraph_audit(
        self,
        graph_data: Dict[str, Any],
        train_output: Dict[str, Any],
        selected_subgraphs: List[SampledSubgraphRecord],
        batch_logs: List[Dict[str, Any]],
        added_triples: List[Dict[str, Any]],
        total_mining_time_sec: float,
    ):
        if os.path.isfile(self.audit_path):
            with open(self.audit_path, "r") as f:
                audit = json.load(f)
        else:
            audit = {"runs": []}

        audit["runs"].append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "hidden_triplet_mining_strategy": "subgraph_missingness",
                "num_graph_nodes": int(graph_data["num_nodes"]),
                "num_graph_edges": int(graph_data["num_edges"]),
                "num_fact_edges": int(graph_data["num_fact_edges"]),
                "num_synonym_edges": int(graph_data["num_synonym_edges"]),
                "num_entity_chunk_edges": int(graph_data["num_entity_chunk_edges"]),
                "hidden_triplet_reproduction_profile": getattr(
                    self.global_config,
                    "hidden_triplet_reproduction_profile",
                    {},
                ),
                "hidden_triplet_config_signature": self.hipporag.get_hidden_triplet_mining_config_signature(),
                "hidden_triplet_subgraph_selection": graph_data.get("hidden_triplet_subgraph_selection_stats", {}),
                "gnn_training_time_sec": round(float(train_output["training_time_sec"]), 6),
                "gnn_train_loss": round(float(train_output["train_loss"]), 6),
                "gnn_eval_loss": None if train_output["eval_loss"] is None else round(float(train_output["eval_loss"]), 6),
                "gnn_best_eval_loss": (
                    None
                    if train_output["best_eval_loss"] is None
                    else round(float(train_output["best_eval_loss"]), 6)
                ),
                "gnn_eval_accuracy": None if train_output["eval_accuracy"] is None else round(float(train_output["eval_accuracy"]), 6),
                "gnn_eval_f1": None if train_output["eval_f1"] is None else round(float(train_output["eval_f1"]), 6),
                "gnn_eval_ranking_accuracy": (
                    None
                    if train_output["eval_ranking_accuracy"] is None
                    else round(float(train_output["eval_ranking_accuracy"]), 6)
                ),
                "gnn_encoder_type": self.global_config.hidden_triplet_gnn_encoder_type,
                "gnn_training_logs": train_output["training_logs"],
                "total_mining_time_sec": round(float(total_mining_time_sec), 6),
                "selected_subgraphs": [
                    {
                        "subgraph_id": record.subgraph_id,
                        "root_node_name": record.root_node_name,
                        "root_node_text": record.root_node_text,
                        "missing_score": round(float(record.missing_score), 6),
                        "num_nodes": len(record.node_indices),
                        "num_entities": len(record.entity_node_names),
                        "num_chunks": len(record.chunk_ids),
                        "num_known_triples": len(record.known_triples),
                        "unique_fact_edges": record.unique_fact_edges,
                        "chunk_ids": record.chunk_ids,
                        "known_triples": [list(triple) for triple in record.known_triples],
                    }
                    for record in selected_subgraphs
                ],
                "batch_logs": batch_logs,
                "added_triples": [
                    {
                        "triple": triple_item["triple"],
                        "chunk_ids": triple_item["chunk_ids"],
                        "score": round(float(triple_item["score"]), 6),
                    }
                    for triple_item in added_triples
                ],
            }
        )

        with open(self.audit_path, "w") as f:
            json.dump(audit, f, ensure_ascii=False, indent=2)

    def _extract_entity_subgraph(self) -> Dict[str, Any]:
        """Project the HippoRAG backbone graph into an entity-only graph for GNN training."""
        graph = self.hipporag.graph
        raw_entity_node_keys = list(self.hipporag.entity_node_keys)
        raw_entity_embeddings = np.array(self.hipporag.entity_embeddings, dtype=np.float32)

        entity_node_keys = []
        filtered_embeddings = []
        node_key_to_text = {}
        for node_key, embedding in zip(raw_entity_node_keys, raw_entity_embeddings):
            node_text = self.hipporag.get_entity_row(node_key)["content"]
            if not self._is_valid_training_entity_text(node_text):
                continue
            entity_node_keys.append(node_key)
            filtered_embeddings.append(embedding)
            node_key_to_text[node_key] = node_text

        entity_embeddings = np.array(filtered_embeddings, dtype=np.float32)
        node_key_to_idx = {node_key: idx for idx, node_key in enumerate(entity_node_keys)}
        valid_entity_node_key_set = set(entity_node_keys)

        fact_edges: Dict[Tuple[str, str], float] = {}
        synonym_edges: Dict[Tuple[str, str], float] = {}
        synonym_neighbors: Dict[str, Set[str]] = defaultdict(set)
        observed_neighbors: Dict[str, Set[str]] = defaultdict(set)
        entity_chunk_ids: Dict[str, Set[str]] = {
            node_key: set(self.hipporag.ent_node_to_chunk_ids.get(node_key, set()))
            for node_key in entity_node_keys
        }
        chunk_to_entities: Dict[str, Set[str]] = defaultdict(set)
        for node_key, chunk_ids in entity_chunk_ids.items():
            for chunk_id in chunk_ids:
                chunk_to_entities[chunk_id].add(node_key)

        for edge in graph.es:
            if "weight" not in edge.attributes():
                continue

            src_name = graph.vs[edge.source]["name"]
            dst_name = graph.vs[edge.target]["name"]
            if not self._is_entity_node(src_name) or not self._is_entity_node(dst_name):
                continue
            if src_name not in valid_entity_node_key_set or dst_name not in valid_entity_node_key_set:
                continue

            pair = tuple(sorted((src_name, dst_name)))
            if pair[0] == pair[1]:
                continue

            observed_neighbors[pair[0]].add(pair[1])
            observed_neighbors[pair[1]].add(pair[0])

            weight = float(edge["weight"])
            if self._is_fact_edge(weight):
                fact_edges[pair] = max(fact_edges.get(pair, 0.0), weight)
            elif self._is_synonym_edge(weight):
                synonym_edges[pair] = max(synonym_edges.get(pair, 0.0), weight)
                synonym_neighbors[pair[0]].add(pair[1])
                synonym_neighbors[pair[1]].add(pair[0])

        return {
            "entity_node_keys": entity_node_keys,
            "entity_embeddings": entity_embeddings,
            "node_key_to_idx": node_key_to_idx,
            "node_key_to_text": node_key_to_text,
            "fact_edges": fact_edges,
            "synonym_edges": synonym_edges,
            "synonym_neighbors": synonym_neighbors,
            "observed_pairs": set(fact_edges.keys()) | set(synonym_edges.keys()),
            "observed_neighbors": observed_neighbors,
            "entity_chunk_ids": entity_chunk_ids,
            "chunk_to_entities": {chunk_id: sorted(node_keys) for chunk_id, node_keys in chunk_to_entities.items()},
            "entity_degree": {
                node_key: len(observed_neighbors.get(node_key, set()))
                for node_key in entity_node_keys
            },
        }

    def _train_link_predictor(self, graph_data: Dict[str, Any]) -> Dict[str, Any]:
        train_start_time = time.perf_counter()
        fact_pairs = list(graph_data["fact_edges"].keys())
        rng_seed = self.global_config.seed if self.global_config.seed is not None else 42
        self._set_gnn_training_seed(int(rng_seed))
        logger.info("Fixed hidden triplet GNN training random seed to %d.", int(rng_seed))
        rng = random.Random(rng_seed)
        fact_pairs = fact_pairs[:]
        rng.shuffle(fact_pairs)

        eval_pairs, training_fact_pool = self._prepare_fact_edge_splits(fact_pairs, rng)
        if len(training_fact_pool) == 0:
            training_fact_pool = fact_pairs

        # The encoder input dimension comes from the precomputed entity embeddings.
        # `support_graph` is built per-epoch after dynamic fact-edge remasking, so it
        # cannot be referenced during encoder initialization.
        encoder = PyGEntityLinkEncoder(
            input_dim=int(graph_data["entity_embeddings"].shape[1]),
            hidden_dim=self.global_config.hidden_triplet_gnn_hidden_dim,
            encoder_type=self.global_config.hidden_triplet_gnn_encoder_type,
        ).to(self.device)
        logger.info(
            "Using hidden triplet GNN encoder type: %s",
            self.global_config.hidden_triplet_gnn_encoder_type,
        )
        decoder = DotProductLinkPredictor().to(self.device)
        optimizer = torch.optim.Adam(
            encoder.parameters(),
            lr=self.global_config.hidden_triplet_learning_rate,
        )
        negatives_per_positive = LEGACY_PAIRWISE_NEGATIVES_PER_POSITIVE
        temperature = LEGACY_PAIRWISE_CONTRASTIVE_TEMPERATURE
        graph_refresh_interval = max(1, int(self.global_config.hidden_triplet_graph_refresh_interval))
        early_stopping_patience = max(0, int(self.global_config.hidden_triplet_early_stopping_patience))
        early_stopping_min_delta = max(0.0, float(self.global_config.hidden_triplet_early_stopping_min_delta))
        early_stopping_enabled = early_stopping_patience > 0 and len(eval_pairs) > 0
        eval_positive_edge_index = None
        eval_negative_edge_index = None
        eval_negative_sampling_stats = {}
        if len(eval_pairs) > 0:
            eval_positive_edge_index = self._pairs_to_edge_index(
                pairs=eval_pairs,
                node_key_to_idx=graph_data["node_key_to_idx"],
                undirected=False,
            ).to(self.device)
            # Keep evaluation negatives fixed across logging checkpoints so early
            # stopping reacts to model changes instead of eval-sampling noise.
            eval_negative_edge_index, eval_negative_sampling_stats = self._sample_negative_edges(
                num_nodes=len(graph_data["entity_node_keys"]),
                num_positive_edges=eval_positive_edge_index.size(1),
                negatives_per_positive=negatives_per_positive,
                graph_data=graph_data,
                rng=random.Random(rng_seed + 7919),
            )

        encoder.train()
        last_loss = None
        last_positive_pairs = []
        last_support_fact_pairs = []
        support_graph = None
        support_fact_pairs = []
        positive_pairs = []
        completed_epochs = 0
        stopped_early = False
        best_epoch = None
        best_eval_pairwise_accuracy = None
        best_state_dict = None
        best_positive_pairs = []
        best_support_fact_pairs = []
        best_train_loss = None
        training_logs: List[Dict[str, Any]] = []
        last_negative_sampling_stats: Dict[str, Any] = {}

        def evaluate_current_model(current_support_graph: Data) -> Dict[str, Any]:
            if eval_positive_edge_index is None or eval_negative_edge_index is None:
                return {
                    "eval_loss": None,
                    "eval_accuracy": None,
                    "eval_pairwise_accuracy": None,
                }

            was_training = encoder.training
            encoder.eval()
            with torch.no_grad():
                eval_z = encoder(current_support_graph)
                eval_pos_scores = decoder(eval_z, eval_positive_edge_index)
                eval_neg_scores = decoder(eval_z, eval_negative_edge_index)
                current_eval_metrics = self._evaluate_contrastive_link_prediction(
                    pos_scores=eval_pos_scores,
                    neg_scores=eval_neg_scores,
                    negatives_per_positive=negatives_per_positive,
                    temperature=temperature,
                )
            if was_training:
                encoder.train()
            return current_eval_metrics

        for epoch_idx in range(self.global_config.hidden_triplet_gnn_epochs):
            # Refresh the remasked fact-edge split and support graph every k epochs
            # so training stays more stable while still benefiting from dynamic
            # graph-level data augmentation across the full run.
            if epoch_idx % graph_refresh_interval == 0 or support_graph is None:
                support_fact_pairs, positive_pairs = self._sample_epoch_training_pairs(training_fact_pool, rng)
                support_graph = self._build_support_graph(graph_data, support_fact_pairs)
            positive_edge_index = self._pairs_to_edge_index(
                pairs=positive_pairs,
                node_key_to_idx=graph_data["node_key_to_idx"],
                undirected=False,
            ).to(self.device)
            negative_edge_index, negative_sampling_stats = self._sample_negative_edges(
                num_nodes=support_graph.num_nodes,
                num_positive_edges=positive_edge_index.size(1),
                negatives_per_positive=negatives_per_positive,
                graph_data=graph_data,
                rng=rng,
            )
            last_negative_sampling_stats = negative_sampling_stats

            z = encoder(support_graph)
            pos_scores = decoder(z, positive_edge_index)
            neg_scores = decoder(z, negative_edge_index)

            loss = self._contrastive_loss(
                pos_scores=pos_scores,
                neg_scores=neg_scores,
                negatives_per_positive=negatives_per_positive,
                temperature=temperature,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            last_loss = loss.detach().item()
            last_positive_pairs = positive_pairs
            last_support_fact_pairs = support_fact_pairs
            completed_epochs = epoch_idx + 1

            should_log_epoch = (
                completed_epochs % graph_refresh_interval == 0
                or completed_epochs == self.global_config.hidden_triplet_gnn_epochs
            )
            if should_log_epoch:
                current_eval_metrics = evaluate_current_model(support_graph)
                log_entry = {
                    "epoch": completed_epochs,
                    "total_epochs": int(self.global_config.hidden_triplet_gnn_epochs),
                    "refresh_interval": graph_refresh_interval,
                    "masked_fact_edges": len(last_positive_pairs),
                    "support_fact_edges": len(last_support_fact_pairs),
                    "eval_fact_edges": len(eval_pairs),
                    "train_loss": None if last_loss is None else float(last_loss),
                    "eval_loss": current_eval_metrics["eval_loss"],
                    "eval_accuracy": current_eval_metrics["eval_accuracy"],
                    "eval_pairwise_accuracy": current_eval_metrics["eval_pairwise_accuracy"],
                    "best_epoch": best_epoch,
                    "best_eval_pairwise_accuracy": best_eval_pairwise_accuracy,
                    "early_stopping_patience": early_stopping_patience,
                    "early_stopping_min_delta": early_stopping_min_delta,
                    "early_stopping_enabled": early_stopping_enabled,
                    "early_stopped": False,
                    "is_best_checkpoint": False,
                    "negative_sampling_stats": dict(last_negative_sampling_stats),
                }
                if current_eval_metrics["eval_pairwise_accuracy"] is None:
                    training_logs.append(log_entry)
                    logger.info(
                        "GNN training epoch %d/%d: refresh_interval=%d masked_fact_edges=%d "
                        "support_fact_edges=%d train_loss=%.6f",
                        completed_epochs,
                        self.global_config.hidden_triplet_gnn_epochs,
                        graph_refresh_interval,
                        len(last_positive_pairs),
                        len(last_support_fact_pairs),
                        last_loss,
                    )
                else:
                    current_pairwise = current_eval_metrics["eval_pairwise_accuracy"]
                    improved = (
                        best_eval_pairwise_accuracy is None
                        or current_pairwise > best_eval_pairwise_accuracy + early_stopping_min_delta
                    )
                    if improved:
                        best_eval_pairwise_accuracy = current_pairwise
                        best_epoch = completed_epochs
                        best_state_dict = {
                            key: value.detach().cpu().clone()
                            for key, value in encoder.state_dict().items()
                        }
                        best_positive_pairs = list(last_positive_pairs)
                        best_support_fact_pairs = list(last_support_fact_pairs)
                        best_train_loss = last_loss
                        log_entry["is_best_checkpoint"] = True
                    log_entry["best_epoch"] = best_epoch
                    log_entry["best_eval_pairwise_accuracy"] = best_eval_pairwise_accuracy
                    training_logs.append(log_entry)

                    logger.info(
                        "GNN training epoch %d/%d: refresh_interval=%d masked_fact_edges=%d "
                        "support_fact_edges=%d eval_fact_edges=%d train_loss=%.6f "
                        "eval_loss=%.6f eval_accuracy=%.4f pairwise_accuracy=%.4f "
                        "best_pairwise_accuracy=%s best_epoch=%s early_stop_patience=%d",
                        completed_epochs,
                        self.global_config.hidden_triplet_gnn_epochs,
                        graph_refresh_interval,
                        len(last_positive_pairs),
                        len(last_support_fact_pairs),
                        len(eval_pairs),
                        last_loss,
                        current_eval_metrics["eval_loss"],
                        current_eval_metrics["eval_accuracy"],
                        current_pairwise,
                        "None" if best_eval_pairwise_accuracy is None else f"{best_eval_pairwise_accuracy:.4f}",
                        "None" if best_epoch is None else str(best_epoch),
                        early_stopping_patience,
                    )

                    if (
                        early_stopping_enabled
                        and best_epoch is not None
                        and completed_epochs - best_epoch >= early_stopping_patience
                    ):
                        stopped_early = True
                        log_entry["early_stopped"] = True
                        logger.info(
                            "Early stopping GNN training at epoch %d: no eval_pairwise_accuracy "
                            "improvement greater than %.6f for %d epochs. Best epoch=%d best_pairwise_accuracy=%.4f",
                            completed_epochs,
                            early_stopping_min_delta,
                            early_stopping_patience,
                            best_epoch,
                            best_eval_pairwise_accuracy,
                        )
                        break

        if early_stopping_enabled and best_state_dict is not None:
            encoder.load_state_dict(best_state_dict)
            last_positive_pairs = best_positive_pairs
            last_support_fact_pairs = best_support_fact_pairs
            last_loss = best_train_loss

        encoder.eval()
        with torch.no_grad():
            eval_support_pairs = last_support_fact_pairs if len(last_support_fact_pairs) > 0 else training_fact_pool
            support_graph = self._build_support_graph(graph_data, eval_support_pairs)
            z = encoder(support_graph)
            embeddings = z.detach().cpu().numpy()
            if eval_positive_edge_index is not None and eval_negative_edge_index is not None:
                eval_pos_scores = decoder(z, eval_positive_edge_index)
                eval_neg_scores = decoder(z, eval_negative_edge_index)
                eval_metrics = self._evaluate_contrastive_link_prediction(
                    pos_scores=eval_pos_scores,
                    neg_scores=eval_neg_scores,
                    negatives_per_positive=negatives_per_positive,
                    temperature=temperature,
                )
            else:
                eval_metrics = {
                    "eval_loss": None,
                    "eval_accuracy": None,
                    "eval_pairwise_accuracy": None,
                }

        training_time_sec = time.perf_counter() - train_start_time
        if eval_metrics["eval_loss"] is None:
            logger.info(
                "Finished GNN training in %.4f seconds with dynamic masking only. "
                "refresh_interval=%d last_epoch_masked_fact_edges=%d support_fact_edges=%d train_loss=%.6f",
                training_time_sec,
                graph_refresh_interval,
                len(last_positive_pairs),
                len(last_support_fact_pairs),
                0.0 if last_loss is None else last_loss,
            )
        else:
            logger.info(
                "Finished GNN training in %.4f seconds with dynamic masking. "
                "refresh_interval=%d last_epoch_masked_fact_edges=%d support_fact_edges=%d eval_fact_edges=%d "
                "train_loss=%.6f eval_loss=%.6f eval_accuracy=%.4f pairwise_accuracy=%.4f",
                training_time_sec,
                graph_refresh_interval,
                len(last_positive_pairs),
                len(last_support_fact_pairs),
                len(eval_pairs),
                0.0 if last_loss is None else last_loss,
                eval_metrics["eval_loss"],
                eval_metrics["eval_accuracy"],
                eval_metrics["eval_pairwise_accuracy"],
            )

        return {
            "embeddings": embeddings,
            "training_time_sec": training_time_sec,
            "num_masked_fact_edges": len(last_positive_pairs),
            "num_support_fact_edges": len(last_support_fact_pairs),
            "num_eval_fact_edges": len(eval_pairs),
            "train_loss": 0.0 if last_loss is None else last_loss,
            "eval_loss": eval_metrics["eval_loss"],
            "eval_accuracy": eval_metrics["eval_accuracy"],
            "eval_pairwise_accuracy": eval_metrics["eval_pairwise_accuracy"],
            "graph_refresh_interval": graph_refresh_interval,
            "early_stopping_enabled": early_stopping_enabled,
            "early_stopped": stopped_early,
            "completed_epochs": completed_epochs,
            "best_epoch": best_epoch,
            "best_eval_pairwise_accuracy": best_eval_pairwise_accuracy,
            "early_stopping_patience": early_stopping_patience,
            "early_stopping_min_delta": early_stopping_min_delta,
            "training_logs": training_logs,
            "eval_negative_sampling_stats": eval_negative_sampling_stats,
            "last_negative_sampling_stats": last_negative_sampling_stats,
        }

    def _resolve_legacy_pairwise_candidate_cap(self, num_fact_edges: int) -> int:
        raw_value = LEGACY_PAIRWISE_MAX_CANDIDATE_PAIRS
        try:
            numeric_value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid legacy pairwise candidate cap={raw_value!r}; expected an int or float."
            ) from exc

        if numeric_value < 0:
            raise ValueError(
                f"Legacy pairwise candidate cap must be non-negative, got {raw_value!r}."
            )
        if numeric_value == 0:
            return 0
        if 0 < numeric_value < 1:
            return max(1, int(math.ceil(num_fact_edges * numeric_value)))
        return max(1, int(numeric_value))

    def _prepare_fact_edge_splits(
        self,
        fact_pairs: List[Tuple[str, str]],
        rng: random.Random,
    ) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        if len(fact_pairs) <= 1:
            return [], fact_pairs

        if not LEGACY_PAIRWISE_USE_FIXED_EVAL_SPLIT:
            return [], fact_pairs

        eval_count = max(1, int(len(fact_pairs) * LEGACY_PAIRWISE_EVAL_RATIO))
        eval_count = min(eval_count, max(1, len(fact_pairs) - 1))
        shuffled_pairs = fact_pairs[:]
        rng.shuffle(shuffled_pairs)
        eval_pairs = shuffled_pairs[:eval_count]
        training_fact_pool = shuffled_pairs[eval_count:]
        if len(training_fact_pool) == 0:
            training_fact_pool = eval_pairs
            eval_pairs = []
        return eval_pairs, training_fact_pool

    def _sample_epoch_training_pairs(
        self,
        training_fact_pool: List[Tuple[str, str]],
        rng: random.Random,
    ) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        if len(training_fact_pool) == 1:
            return training_fact_pool, training_fact_pool

        epoch_pairs = training_fact_pool[:]
        rng.shuffle(epoch_pairs)
        mask_count = max(1, int(len(epoch_pairs) * LEGACY_PAIRWISE_MASK_RATIO))
        mask_count = min(mask_count, max(1, len(epoch_pairs) - 1))
        positive_pairs = epoch_pairs[:mask_count]
        support_fact_pairs = epoch_pairs[mask_count:]
        if len(support_fact_pairs) == 0:
            support_fact_pairs = positive_pairs
        return support_fact_pairs, positive_pairs

    def _generate_candidate_records(
        self,
        graph_data: Dict[str, Any],
        embeddings: np.ndarray,
    ) -> List[CandidatePairRecord]:
        """Use a FAISS ANN index over learned embeddings to retrieve unseen entity pairs."""
        if faiss is None:
            raise ImportError(
                "faiss is required for candidate generation but is not installed in the current environment."
            )

        retrieval_start_time = time.perf_counter()
        entity_node_keys = graph_data["entity_node_keys"]
        node_key_to_text = graph_data["node_key_to_text"]
        node_key_to_idx = graph_data["node_key_to_idx"]
        observed_pairs = graph_data["observed_pairs"]
        observed_neighbors = graph_data["observed_neighbors"]
        candidate_records: List[CandidatePairRecord] = []
        preliminary_candidates: List[Dict[str, Any]] = []
        seen_pairs: Set[Tuple[str, str]] = set()
        max_candidate_pairs = int(
            graph_data.get(
                "legacy_pairwise_candidate_cap",
                self._resolve_legacy_pairwise_candidate_cap(len(graph_data["fact_edges"])),
            )
        )
        graph_data["legacy_pairwise_candidate_cap"] = max_candidate_pairs
        if max_candidate_pairs == 0:
            logger.info(
                "Skipping candidate generation because the legacy pairwise candidate cap resolved to 0."
            )
            graph_data["hidden_triplet_candidate_generation_stats"] = {
                "num_raw_surface_candidate_pairs": 0,
                "num_candidates_before_truncation": 0,
                "num_candidates_after_truncation": 0,
                "num_candidate_records_after_evidence_filter": 0,
            }
            return []
        topk = min(LEGACY_PAIRWISE_CANDIDATE_TOPK, max(1, len(entity_node_keys) - 1))
        num_entities = len(entity_node_keys)
        ann_query_k = min(num_entities, max(topk * 10, topk + 64))
        raw_similarity_upper_threshold = LEGACY_PAIRWISE_RAW_SIMILARITY_UPPER_THRESHOLD

        # These caches turn repeated dictionary/set work inside the hot candidate loop
        # into O(1) lookups.
        node_filter_cache = {
            node_key: self._should_filter_candidate_entity_text(node_key_to_text[node_key])
            for node_key in entity_node_keys
        }
        node_filter_mask = np.array(
            [node_filter_cache[node_key] for node_key in entity_node_keys],
            dtype=bool,
        )
        original_entity_embeddings = np.ascontiguousarray(graph_data["entity_embeddings"].astype(np.float32))
        original_embedding_norms = np.linalg.norm(original_entity_embeddings, axis=1, keepdims=True)
        original_embedding_norms = np.clip(original_embedding_norms, 1e-12, None)
        normalized_original_entity_embeddings = original_entity_embeddings / original_embedding_norms
        blocked_neighbor_idx_cache = {
            src_key: np.array(
                sorted(
                    node_key_to_idx[dst_key]
                    for dst_key in observed_neighbors.get(src_key, set())
                    if dst_key in node_key_to_idx
                ),
                dtype=np.int64,
            )
            for src_key in entity_node_keys
        }

        build_start_time = time.perf_counter()
        faiss_base = self._build_faiss_index(embeddings)
        build_time_sec = time.perf_counter() - build_start_time

        search_start_time = time.perf_counter()
        ann_scores, ann_indices = faiss_base.search(np.ascontiguousarray(embeddings.astype(np.float32)), ann_query_k)
        search_time_sec = time.perf_counter() - search_start_time

        logger.info(
            "FAISS candidate retrieval completed: num_entities=%d ann_query_k=%d build_time_sec=%.4f "
            "search_time_sec=%.4f total_faiss_time_sec=%.4f",
            num_entities,
            ann_query_k,
            build_time_sec,
            search_time_sec,
            time.perf_counter() - retrieval_start_time,
        )

        filter_start_time = time.perf_counter()
        raw_similarity_filtered_pairs = 0
        logger.info("Starting candidate post-filtering after FAISS retrieval.")
        with logging_redirect_tqdm():
            for src_idx, src_key in tqdm(
                enumerate(entity_node_keys),
                total=len(entity_node_keys),
                desc="Candidate post-filtering",
                position=0,
                dynamic_ncols=True,
            ):
                if node_filter_cache[src_key]:
                    continue

                candidate_dst_indices = ann_indices[src_idx]
                candidate_raw_scores = ann_scores[src_idx]

                # Batch away the cheapest filters first so the expensive Python-side logic
                # only sees a small candidate pool.
                valid_mask = (candidate_dst_indices >= 0)
                valid_mask &= (candidate_dst_indices != src_idx)
                if not np.any(valid_mask):
                    continue

                candidate_dst_indices = candidate_dst_indices[valid_mask]
                candidate_raw_scores = candidate_raw_scores[valid_mask]

                non_filtered_mask = ~node_filter_mask[candidate_dst_indices]
                if not np.any(non_filtered_mask):
                    continue

                candidate_dst_indices = candidate_dst_indices[non_filtered_mask]
                candidate_raw_scores = candidate_raw_scores[non_filtered_mask]

                blocked_neighbor_indices = blocked_neighbor_idx_cache[src_key]
                if blocked_neighbor_indices.size > 0:
                    not_blocked_mask = ~np.isin(candidate_dst_indices, blocked_neighbor_indices, assume_unique=False)
                    if not np.any(not_blocked_mask):
                        continue
                    candidate_dst_indices = candidate_dst_indices[not_blocked_mask]
                    candidate_raw_scores = candidate_raw_scores[not_blocked_mask]

                # Use the original HippoRAG entity embeddings as a lightweight
                # pre-filter. Pairs that are almost identical in the original
                # embedding space are more likely to be aliases/near-duplicates
                # than useful hidden-relation candidates.
                if raw_similarity_upper_threshold < 1.0 and len(candidate_dst_indices) > 0:
                    src_original_embedding = normalized_original_entity_embeddings[src_idx]
                    candidate_original_similarities = (
                        normalized_original_entity_embeddings[candidate_dst_indices] @ src_original_embedding
                    )
                    keep_mask = candidate_original_similarities < raw_similarity_upper_threshold
                    raw_similarity_filtered_pairs += int(candidate_dst_indices.size - np.count_nonzero(keep_mask))
                    if not np.any(keep_mask):
                        continue
                    candidate_dst_indices = candidate_dst_indices[keep_mask]
                    candidate_raw_scores = candidate_raw_scores[keep_mask]

                for dst_idx, raw_score in zip(candidate_dst_indices, candidate_raw_scores):
                    dst_key = entity_node_keys[int(dst_idx)]

                    pair = (src_key, dst_key) if src_key < dst_key else (dst_key, src_key)
                    if pair in observed_pairs or pair in seen_pairs:
                        continue

                    score = 1.0 / (1.0 + math.exp(-float(raw_score)))
                    seen_pairs.add(pair)
                    preliminary_candidates.append(
                        {
                            "pair_ids": pair,
                            "pair_names": (
                                node_key_to_text[pair[0]],
                                node_key_to_text[pair[1]],
                            ),
                            "score": score,
                        }
                    )

        generation_stats = {
            "num_raw_surface_candidate_pairs": len(preliminary_candidates),
        }

        preliminary_candidates.sort(key=lambda item: item["score"], reverse=True)
        num_candidates_before_truncation = len(preliminary_candidates)
        preliminary_candidates = preliminary_candidates[:max_candidate_pairs]
        generation_stats["num_candidates_before_truncation"] = num_candidates_before_truncation
        generation_stats["num_candidates_after_truncation"] = len(preliminary_candidates)

        if len(preliminary_candidates) == 0:
            logger.info(
                "Finished candidate post-filtering in %.4f seconds. "
                "num_candidates_before_truncation=%d max_candidate_pairs=%d "
                "raw_similarity_filtered_pairs=%d raw_similarity_upper_threshold=%.4f",
                time.perf_counter() - filter_start_time,
                num_candidates_before_truncation,
                max_candidate_pairs,
                raw_similarity_filtered_pairs,
                raw_similarity_upper_threshold,
            )
            generation_stats["num_candidate_records_after_evidence_filter"] = 0
            graph_data["hidden_triplet_candidate_generation_stats"] = generation_stats
            return []

        chunk_index_prepare_start_time = time.perf_counter()
        chunk_search_node_keys, global_chunk_faiss_index = (
            self._get_cached_global_chunk_search_data()
        )
        chunk_index_prepare_time_sec = time.perf_counter() - chunk_index_prepare_start_time

        pair_query_texts = [
            f"{candidate['pair_names'][0]} {candidate['pair_names'][1]}".strip()
            for candidate in preliminary_candidates
        ]
        pair_query_embeddings = self.hipporag.embedding_model.batch_encode(pair_query_texts)
        pair_query_embeddings = self._normalize_embedding_matrix(
            np.array(pair_query_embeddings, dtype=np.float32)
        )

        # Retrieve supporting chunks from the full chunk corpus instead of only
        # the direct chunks already attached to the endpoint entities. This lets
        # hidden-triplet mining explore passages that the original HippoRAG
        # OpenIE stage may not have linked to the candidate pair.
        chunk_search_start_time = time.perf_counter()
        top_chunk_count = min(2, len(chunk_search_node_keys))
        _, chunk_indices = global_chunk_faiss_index.search(
            np.ascontiguousarray(pair_query_embeddings.astype(np.float32)),
            top_chunk_count,
        )
        chunk_search_time_sec = time.perf_counter() - chunk_search_start_time

        for candidate_idx, candidate in enumerate(preliminary_candidates):
            ranked_chunk_ids = [
                chunk_search_node_keys[int(chunk_idx)]
                for chunk_idx in chunk_indices[candidate_idx]
                if int(chunk_idx) >= 0
            ]
            chunk_ids_for_pair = ranked_chunk_ids[:2]

            if len(chunk_ids_for_pair) == 0:
                continue

            candidate_records.append(
                CandidatePairRecord(
                    pair_ids=candidate["pair_ids"],
                    pair_names=candidate["pair_names"],
                    score=float(candidate["score"]),
                    chunk_ids=chunk_ids_for_pair,
                )
            )

        generation_stats["num_candidate_records_after_evidence_filter"] = len(candidate_records)
        graph_data["hidden_triplet_candidate_generation_stats"] = generation_stats

        logger.info(
            "Finished candidate post-filtering in %.4f seconds. "
            "num_candidates_before_truncation=%d max_candidate_pairs=%d "
            "raw_similarity_filtered_pairs=%d raw_similarity_upper_threshold=%.4f "
            "global_chunk_faiss_prepare_time_sec=%.4f global_chunk_search_time_sec=%.4f "
            "candidate_records_after_evidence_filter=%d",
            time.perf_counter() - filter_start_time,
            num_candidates_before_truncation,
            max_candidate_pairs,
            raw_similarity_filtered_pairs,
            raw_similarity_upper_threshold,
            chunk_index_prepare_time_sec,
            chunk_search_time_sec,
            len(candidate_records),
        )
        return candidate_records

    @staticmethod
    def _build_faiss_index(embeddings: np.ndarray):
        embedding_matrix = np.ascontiguousarray(embeddings.astype(np.float32))
        embedding_dim = embedding_matrix.shape[1]

        index = faiss.IndexHNSWFlat(embedding_dim, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 128
        index.add(embedding_matrix)
        return index

    @staticmethod
    def _normalize_embedding_matrix(embeddings: np.ndarray) -> np.ndarray:
        embedding_matrix = np.ascontiguousarray(embeddings.astype(np.float32))
        embedding_norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
        embedding_norms = np.clip(embedding_norms, 1e-12, None)
        return embedding_matrix / embedding_norms

    def _get_cached_global_chunk_search_data(
        self,
    ) -> Tuple[List[str], Any]:
        if (
            self._cached_chunk_search_node_keys is None
            or self._cached_normalized_chunk_search_embeddings is None
            or self._cached_global_chunk_faiss_index is None
        ):
            if not hasattr(self.hipporag, "passage_node_keys") or not hasattr(self.hipporag, "passage_embeddings"):
                raise ValueError("HippoRAG passage embeddings are not prepared; call prepare_retrieval_objects() first.")

            self._cached_chunk_search_node_keys = list(self.hipporag.passage_node_keys)
            if len(self._cached_chunk_search_node_keys) == 0:
                raise ValueError("No passage nodes found; cannot retrieve global chunks for candidate pairs.")

            self._cached_normalized_chunk_search_embeddings = self._normalize_embedding_matrix(
                np.array(self.hipporag.passage_embeddings, dtype=np.float32)
            )
            self._cached_global_chunk_faiss_index = self._build_faiss_index(
                self._cached_normalized_chunk_search_embeddings
            )

        return (
            self._cached_chunk_search_node_keys,
            self._cached_global_chunk_faiss_index,
        )

    def _collect_candidate_chunk_ids(
        self,
        pair: Tuple[str, str],
        synonym_neighbors: Dict[str, Set[str]],
    ) -> List[str]:
        # Hidden-relation evidence is intentionally restricted to the two endpoint
        # entities' direct chunks so synonym expansion does not introduce unrelated
        # context into the downstream LLM prompt.
        del synonym_neighbors
        direct_nodes = {pair[0], pair[1]}
        direct_chunk_ids: Set[str] = set()
        for node_id in direct_nodes:
            direct_chunk_ids.update(self.hipporag.ent_node_to_chunk_ids.get(node_id, set()))

        ordered_chunk_ids = sorted(direct_chunk_ids)
        max_chunks = LEGACY_PAIRWISE_MAX_CHUNKS_PER_CALL
        if len(ordered_chunk_ids) > max_chunks:
            ordered_chunk_ids = ordered_chunk_ids[:max_chunks]

        return ordered_chunk_ids

    @staticmethod
    def _has_shared_chunks(left_chunk_ids: Set[str], right_chunk_ids: Set[str]) -> bool:
        if not left_chunk_ids or not right_chunk_ids:
            return False

        # Iterate over the smaller set to reduce repeated intersection cost in the hot loop.
        if len(left_chunk_ids) > len(right_chunk_ids):
            left_chunk_ids, right_chunk_ids = right_chunk_ids, left_chunk_ids
        return any(chunk_id in right_chunk_ids for chunk_id in left_chunk_ids)

    @staticmethod
    def _collect_candidate_chunk_ids_from_cache(
        pair: Tuple[str, str],
        direct_chunk_cache: Dict[str, Set[str]],
        max_chunks: int,
    ) -> List[str]:
        # Use only direct chunks from the two endpoint entities. This keeps the
        # evidence focused on the candidate pair itself instead of broadening the
        # prompt with chunks gathered through synonym expansion.
        left_direct_chunks = direct_chunk_cache.get(pair[0], set())
        right_direct_chunks = direct_chunk_cache.get(pair[1], set())
        direct_chunk_ids = left_direct_chunks.union(right_direct_chunks)

        ordered_chunk_ids = sorted(direct_chunk_ids)
        if len(ordered_chunk_ids) > max_chunks:
            ordered_chunk_ids = ordered_chunk_ids[:max_chunks]

        return ordered_chunk_ids

    def _merge_candidate_records(self, records: List[CandidatePairRecord]) -> List[CandidateBatch]:
        # Greedily merge records with overlapping evidence to reduce duplicate LLM calls.
        merge_start_time = time.perf_counter()
        batches: List[CandidateBatch] = []
        max_chunks = LEGACY_PAIRWISE_MAX_CHUNKS_PER_CALL
        max_pairs = LEGACY_PAIRWISE_MAX_PAIRS_PER_CALL

        for record in records:
            record_chunk_set = set(record.chunk_ids)
            best_batch_idx = None
            best_overlap = -1

            for idx, batch in enumerate(batches):
                if len(batch.pair_records) >= max_pairs:
                    continue
                batch_chunk_set = set(batch.chunk_ids)
                merged_chunk_set = batch_chunk_set | record_chunk_set
                if len(merged_chunk_set) > max_chunks:
                    continue

                overlap = len(batch_chunk_set & record_chunk_set)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_batch_idx = idx

            if best_batch_idx is None:
                batches.append(
                    CandidateBatch(
                        pair_records=[record],
                        chunk_ids=list(record.chunk_ids),
                    )
                )
            else:
                batch = batches[best_batch_idx]
                batch.pair_records.append(record)
                batch.chunk_ids = sorted(set(batch.chunk_ids) | record_chunk_set)

        logger.info(
            "Merged %d candidate records into %d LLM batches in %.4f seconds.",
            len(records),
            len(batches),
            time.perf_counter() - merge_start_time,
        )
        return batches

    def _extract_hidden_triples(
        self,
        candidate_batches: List[CandidateBatch],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        if len(candidate_batches) == 0:
            return [], []

        mined_triples: List[Dict[str, Any]] = []
        batch_logs_by_idx: Dict[int, Dict[str, Any]] = {}

        logger.info(
            "Extracting hidden triples from %d candidate batches with LLM.",
            len(candidate_batches),

        )

        max_workers = max(1, self._cfg_int("llm_concurrency", 32))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_batch_idx = {
                executor.submit(self._process_candidate_batch, batch): batch_idx
                for batch_idx, batch in enumerate(candidate_batches)
            }
            with logging_redirect_tqdm():
                for future in tqdm(
                    as_completed(future_to_batch_idx),
                    total=len(future_to_batch_idx),
                    desc="Extracting hidden triples",
                    position=0,            
                    dynamic_ncols=True
                ):
                    batch_idx = future_to_batch_idx[future]
                    batch = candidate_batches[batch_idx]
                    try:
                        batch_valid_triples, batch_log = future.result()
                    except Exception as exc:
                        logger.exception(
                            "Hidden triplet extraction failed for batch %d after retries; continuing with the remaining batches.",
                            batch_idx,
                        )
                        batch_logs_by_idx[batch_idx] = {
                            "pair_names": [record.pair_names for record in batch.pair_records],
                            "chunk_ids": batch.chunk_ids,
                            "num_attempts": 0,
                            "num_valid_triples": 0,
                            "num_uncovered_pairs_after_retry": len(batch.pair_records),
                            "uncovered_pair_names_after_retry": [record.pair_names for record in batch.pair_records],
                            "failed": True,
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                            "attempt_logs": [],
                        }
                        continue

                    mined_triples.extend(batch_valid_triples)
                    batch_logs_by_idx[batch_idx] = batch_log

        batch_logs = [
            batch_logs_by_idx[idx]
            for idx in range(len(candidate_batches))
            if idx in batch_logs_by_idx
        ]

        unique_triples: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        for triple_item in mined_triples:
            triple = triple_item["triple"]
            if triple not in unique_triples:
                unique_triples[triple] = triple_item
            else:
                unique_triples[triple]["chunk_ids"] = sorted(
                    set(unique_triples[triple]["chunk_ids"]) | set(triple_item["chunk_ids"])
                )
                unique_triples[triple]["score"] = max(
                    unique_triples[triple]["score"], triple_item["score"]
                )

        return list(unique_triples.values()), batch_logs

    def _process_candidate_batch(
        self,
        batch: CandidateBatch,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        batch_valid_triples = []
        attempt_logs = []
        pending_records = list(batch.pair_records)

        # Run the original merged batch once. If the model omits some candidate pairs
        # from its structured response, reorganize only those missing pairs and retry
        # them one more time with a narrower prompt.
        for attempt_idx in range(2):
            if len(pending_records) == 0:
                break

            current_batch = (
                batch
                if attempt_idx == 0
                else self._build_retry_candidate_batch(
                    pending_records=pending_records,
                    original_chunk_ids=batch.chunk_ids,
                )
            )

            messages = self._build_relation_mining_messages(current_batch)
            raw_response, metadata = self._infer(messages)
            parsed_results = self._parse_relation_response(raw_response)
            current_valid_triples, covered_pair_keys = self._collect_valid_triples_from_results(
                parsed_results=parsed_results,
                batch=current_batch,
            )
            batch_valid_triples.extend(current_valid_triples)

            current_allowed_pairs = {
                self._normalize_pair_key(record.pair_names[0], record.pair_names[1]): record
                for record in current_batch.pair_records
            }
            pending_records = [
                record
                for pair_key, record in current_allowed_pairs.items()
                if pair_key not in covered_pair_keys
            ]

            attempt_logs.append(
                {
                    "attempt": attempt_idx + 1,
                    "pair_names": [record.pair_names for record in current_batch.pair_records],
                    "chunk_ids": current_batch.chunk_ids,
                    "messages_used_for_llm": messages,
                    "raw_response": raw_response,
                    "parsed_results": parsed_results,
                    "metadata": metadata,
                    "num_valid_triples": len(current_valid_triples),
                    "num_covered_pairs": len(covered_pair_keys),
                    "num_missing_pairs": len(pending_records),
                    "missing_pair_names": [record.pair_names for record in pending_records],
                }
            )

        batch_log = {
            "pair_names": [record.pair_names for record in batch.pair_records],
            "chunk_ids": batch.chunk_ids,
            "num_attempts": len(attempt_logs),
            "num_valid_triples": len(batch_valid_triples),
            "num_uncovered_pairs_after_retry": len(pending_records),
            "uncovered_pair_names_after_retry": [record.pair_names for record in pending_records],
            "attempt_logs": attempt_logs,
        }
        return batch_valid_triples, batch_log

    def _build_relation_mining_messages(self, batch: CandidateBatch) -> List[Dict[str, str]]:
        candidate_pairs = [
            {"subject": record.pair_names[0], "object": record.pair_names[1]}
            for record in batch.pair_records
        ]

        passages = []
        for chunk_id in batch.chunk_ids:
            passage = self.hipporag.chunk_embedding_store.get_row(chunk_id)["content"]
            passage = passage[: self.global_config.hidden_triplet_max_chars_per_chunk]
            passages.append({"chunk_id": chunk_id, "passage": passage})

        system_prompt = (
            "Your task is to construct an RDF (Resource Description Framework) graph from the given evidence passages and candidate entity pairs. "
            "Respond with a JSON object whose 'results' field contains relation triples for the candidate pairs. "
            "Each triple should represent a relationship in the RDF graph and should contain the candidate pair entities as subject and object. "
            "Only extract relationships that are directly stated or very clearly and unambiguously supported by the evidence passages. "
            "For hidden-triplet mining, prefer high-value, retrieval-useful facts rather than weak narrative associations. "
            "Good relations are stable, reusable facts such as identity, role, occupation, family relation, ownership, location, destination, membership, alias, named object, cause, motivation, or other explicit answer-bearing facts. "
            "For literary passages, prefer durable facts over one-off scene interactions. "
            "Do not extract weak, temporary, scene-specific, or low-value relations such as 'met with', 'talked to', 'was with', 'visited', 'saw', 'looked at', 'compared to', 'associated with', 'present at', or any relation that merely says the two entities co-occur in the same event, sentence, or scene. "
            "Do not turn descriptions, metaphors, comparisons, or rhetorical language into RDF triples. "
            "If the evidence only supports a weak association, an ambiguous interaction, or a relation that would not be useful for downstream retrieval or question answering, return 'triples': [] for that pair. "
            "Every returned triple must be supported by at least one cited chunk that directly states or unambiguously entails the relation; do not stitch together weak hints from multiple chunks to manufacture a relation. "
            "Prefer concise, specific predicates such as 'located in', 'started on', 'launched', or 'is', rather than vague predicates such as 'related to'. "
            "Clearly resolve pronouns or descriptions to the candidate entity strings when the evidence makes the reference clear. "
            "Use the exact entity strings from the candidate pair list for triple subjects and objects. "
            "Do not introduce new subject or object entities outside the candidate pair list. "
            "Return exactly one result object for every candidate pair. "
            "Each result object must have keys 'subject', 'object', and 'triples'. "
            "'subject' and 'object' must exactly match one candidate pair. "
            "'triples' must be a list of objects, and each triple object must have keys 'triple' and 'chunk_ids'. "
            "'triple' must be a [subject, predicate, object] triple for that pair. "
            "'chunk_ids' must be a non-empty list of one or more chunk_id strings copied exactly from the provided evidence passages that support the triple. "
            "Do not paraphrase chunk ids, do not add prefixes or suffixes, and do not invent unseen chunk ids. "
            "If there is no relationship that can be grounded in the evidence for a pair, include that pair with 'triples': []."
        )
        one_shot_candidate_pairs = [
            {"subject": "Radio City", "object": "India"},
            {"subject": "Radio City", "object": "3 July 2001"},
            {"subject": "Radio City", "object": "Hindi"},
            {"subject": "Radio City", "object": "English"},
            {"subject": "Radio City", "object": "New Media"},
            {"subject": "Radio City", "object": "PlanetRadiocity.com"},
            {"subject": "PlanetRadiocity.com", "object": "May 2008"},
            {"subject": "PlanetRadiocity.com", "object": "music portal"},
            {"subject": "PlanetRadiocity.com", "object": "news"},
            {"subject": "PlanetRadiocity.com", "object": "videos"},
            {"subject": "PlanetRadiocity.com", "object": "songs"},
        ]
        one_shot_passages = [
            {
                "chunk_id": "chunk-radio-city-example",
                "passage": one_shot_ner_paragraph,
            }
        ]
        one_shot_response = {
            "results": [
                {
                    "subject": "Radio City",
                    "object": "India",
                    "triples": [
                        {
                            "triple": ["Radio City", "located in", "India"],
                            "chunk_ids": ["chunk-radio-city-example"],
                        }
                    ],
                },
                {
                    "subject": "Radio City",
                    "object": "3 July 2001",
                    "triples": [
                        {
                            "triple": ["Radio City", "started on", "3 July 2001"],
                            "chunk_ids": ["chunk-radio-city-example"],
                        }
                    ],
                },
                {
                    "subject": "Radio City",
                    "object": "Hindi",
                    "triples": [
                        {
                            "triple": ["Radio City", "plays songs in", "Hindi"],
                            "chunk_ids": ["chunk-radio-city-example"],
                        }
                    ],
                },
                {
                    "subject": "Radio City",
                    "object": "English",
                    "triples": [
                        {
                            "triple": ["Radio City", "plays songs in", "English"],
                            "chunk_ids": ["chunk-radio-city-example"],
                        }
                    ],
                },
                {
                    "subject": "Radio City",
                    "object": "New Media",
                    "triples": [
                        {
                            "triple": ["Radio City", "forayed into", "New Media"],
                            "chunk_ids": ["chunk-radio-city-example"],
                        }
                    ],
                },
                {
                    "subject": "Radio City",
                    "object": "PlanetRadiocity.com",
                    "triples": [
                        {
                            "triple": ["Radio City", "launched", "PlanetRadiocity.com"],
                            "chunk_ids": ["chunk-radio-city-example"],
                        }
                    ],
                },
                {
                    "subject": "PlanetRadiocity.com",
                    "object": "May 2008",
                    "triples": [
                        {
                            "triple": ["PlanetRadiocity.com", "launched in", "May 2008"],
                            "chunk_ids": ["chunk-radio-city-example"],
                        }
                    ],
                },
                {
                    "subject": "PlanetRadiocity.com",
                    "object": "music portal",
                    "triples": [
                        {
                            "triple": ["PlanetRadiocity.com", "is", "music portal"],
                            "chunk_ids": ["chunk-radio-city-example"],
                        }
                    ],
                },
                {
                    "subject": "PlanetRadiocity.com",
                    "object": "news",
                    "triples": [
                        {
                            "triple": ["PlanetRadiocity.com", "offers", "news"],
                            "chunk_ids": ["chunk-radio-city-example"],
                        }
                    ],
                },
                {
                    "subject": "PlanetRadiocity.com",
                    "object": "videos",
                    "triples": [
                        {
                            "triple": ["PlanetRadiocity.com", "offers", "videos"],
                            "chunk_ids": ["chunk-radio-city-example"],
                        }
                    ],
                },
                {
                    "subject": "PlanetRadiocity.com",
                    "object": "songs",
                    "triples": [
                        {
                            "triple": ["PlanetRadiocity.com", "offers", "songs"],
                            "chunk_ids": ["chunk-radio-city-example"],
                        }
                    ],
                },
            ]
        }
        one_shot_user_prompt = (
            f"Candidate entity pairs:\n{json.dumps(one_shot_candidate_pairs, ensure_ascii=False, indent=2)}\n\n"
            f"Evidence passages:\n{json.dumps(one_shot_passages, ensure_ascii=False, indent=2)}\n\n"
            "Return JSON only."
        )
        user_prompt = (
            f"Candidate entity pairs:\n{json.dumps(candidate_pairs, ensure_ascii=False, indent=2)}\n\n"
            f"Evidence passages:\n{json.dumps(passages, ensure_ascii=False, indent=2)}\n\n"
            "Required output schema example:\n"
            "{\n"
            '  "results": [\n'
            "    {\n"
            '      "subject": "Entity A",\n'
            '      "object": "Entity B",\n'
            '      "triples": [\n'
            "        {\n"
            '          "triple": ["Entity A", "predicate", "Entity B"],\n'
            '          "chunk_ids": ["chunk-1", "chunk-2"]\n'
            "        }\n"
            "      ]\n"
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Return JSON only."
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": one_shot_user_prompt},
            {"role": "assistant", "content": json.dumps(one_shot_response, ensure_ascii=False, indent=2)},
            {"role": "user", "content": user_prompt},
        ]

    def _infer(self, messages: List[Dict[str, str]]) -> Tuple[str, Dict[str, Any]]:
        try:
            result = self.hipporag.llm_model.infer(
                messages=messages,
                max_completion_tokens=self.global_config.hidden_triplet_llm_max_tokens,
                seed=self.global_config.seed,
                temperature=self.global_config.temperature,
            )
        except TypeError:
            result = self.hipporag.llm_model.infer(
                messages=messages,
                max_tokens=self.global_config.hidden_triplet_llm_max_tokens,
                seed=self.global_config.seed,
                temperature=self.global_config.temperature,
            )

        if isinstance(result, tuple) and len(result) == 3:
            raw_response, metadata, cache_hit = result
            metadata = dict(metadata)
            metadata["cache_hit"] = cache_hit
            return raw_response, metadata

        raw_response, metadata = result
        return raw_response, dict(metadata)

    def _parse_relation_response(self, raw_response: str) -> List[Dict[str, Any]]:
        json_match = re.search(r"\{.*\}", raw_response, re.DOTALL)
        if json_match is None:
            return []

        json_str = fix_broken_generated_json(json_match.group(0))
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError:
            return []

        # Backward-compatible fallback: if the model returns the old flat `triples`
        # schema, regroup them by endpoint pair. These fallback triples will not
        # carry chunk-level provenance and downstream filtering will decide whether
        # they are usable.
        if "results" not in parsed:
            grouped_results: Dict[Tuple[str, str], Dict[str, Any]] = {}
            triples = filter_invalid_triples(parsed.get("triples", []))
            for triple in triples:
                pair_key = self._normalize_pair_key(triple[0], triple[2])
                if pair_key not in grouped_results:
                    grouped_results[pair_key] = {
                        "subject": triple[0],
                        "object": triple[2],
                        "triples": [],
                    }
                grouped_results[pair_key]["triples"].append(
                    {
                        "triple": triple,
                        "chunk_ids": [],
                    }
                )
            return list(grouped_results.values())

        parsed_results = parsed.get("results", [])
        if not isinstance(parsed_results, list):
            return []

        normalized_results = []
        for item in parsed_results:
            if not isinstance(item, dict):
                continue
            subject = item.get("subject")
            obj = item.get("object")
            triples = item.get("triples", [])
            if not isinstance(subject, str) or not isinstance(obj, str):
                continue
            if not isinstance(triples, list):
                triples = []

            normalized_triples = []
            for triple_item in triples:
                if isinstance(triple_item, dict):
                    triple_value = triple_item.get("triple", [])
                    chunk_ids = triple_item.get("chunk_ids", [])
                else:
                    # Backward-compatible fallback for responses that still emit
                    # a bare [subject, predicate, object] list.
                    triple_value = triple_item
                    chunk_ids = []

                filtered_triples = filter_invalid_triples([triple_value])
                if len(filtered_triples) == 0:
                    continue

                if isinstance(chunk_ids, str):
                    chunk_ids = [chunk_ids]
                elif not isinstance(chunk_ids, list):
                    chunk_ids = []

                normalized_triples.append(
                    {
                        "triple": filtered_triples[0],
                        "chunk_ids": [
                            chunk_id for chunk_id in chunk_ids
                            if isinstance(chunk_id, str) and chunk_id.strip() != ""
                        ],
                    }
                )

            normalized_results.append(
                {
                    "subject": subject,
                    "object": obj,
                    "triples": normalized_triples,
                }
            )
        return normalized_results

    @staticmethod
    def _normalize_pair_key(left_text: str, right_text: str) -> Tuple[str, str]:
        return tuple(sorted((text_processing(left_text), text_processing(right_text))))

    @staticmethod
    def _normalize_endpoint_text(text: str) -> str:
        return text_processing(text)

    def _collect_valid_triples_from_results(
        self,
        parsed_results: List[Dict[str, Any]],
        batch: CandidateBatch,
    ) -> Tuple[List[Dict[str, Any]], Set[Tuple[str, str]]]:
        allowed_pairs = {
            self._normalize_pair_key(record.pair_names[0], record.pair_names[1]): record
            for record in batch.pair_records
        }

        batch_valid_triples = []
        covered_pair_keys: Set[Tuple[str, str]] = set()
        allowed_chunk_ids = set(batch.chunk_ids)
        chunk_id_alias_map = self._build_chunk_id_alias_map(batch.chunk_ids)
        for result_item in parsed_results:
            pair_key = self._normalize_pair_key(
                result_item.get("subject", ""),
                result_item.get("object", ""),
            )
            if pair_key not in allowed_pairs:
                continue

            covered_pair_keys.add(pair_key)
            for triple_item in result_item.get("triples", []):
                if not isinstance(triple_item, dict):
                    continue
                triple = triple_item.get("triple", [])
                proc_triple = text_processing(triple)
                triple_pair_key = self._normalize_pair_key(proc_triple[0], proc_triple[2])
                if triple_pair_key != pair_key:
                    continue

                # Keep only chunk ids that are explicitly cited by the model and
                # are actually present in the current evidence batch. This gives
                # the downstream augmented graph a precise chunk provenance signal
                # for each hidden triple.
                cited_chunk_ids = self._resolve_model_chunk_ids(
                    raw_chunk_ids=triple_item.get("chunk_ids", []),
                    allowed_chunk_ids=allowed_chunk_ids,
                    chunk_id_alias_map=chunk_id_alias_map,
                )
                if len(cited_chunk_ids) == 0:
                    # If the model finds a valid triple but fails to return a
                    # usable chunk id, fall back to the pair's own candidate
                    # chunks instead of discarding the triple outright.
                    fallback_chunk_ids = [
                        chunk_id
                        for chunk_id in allowed_pairs[pair_key].chunk_ids
                        if chunk_id in allowed_chunk_ids
                    ]
                    if len(fallback_chunk_ids) == 0:
                        continue
                    cited_chunk_ids = fallback_chunk_ids

                batch_valid_triples.append(
                    {
                        "triple": tuple(proc_triple),
                        "pair_names": allowed_pairs[pair_key].pair_names,
                        "chunk_ids": cited_chunk_ids,
                        "score": allowed_pairs[pair_key].score,
                    }
                )

        return batch_valid_triples, covered_pair_keys

    @staticmethod
    def _build_chunk_id_alias_map(chunk_ids: List[str]) -> Dict[str, str]:
        alias_map: Dict[str, str] = {}
        for chunk_id in chunk_ids:
            if not isinstance(chunk_id, str):
                continue
            for alias in GNNHiddenTripletMiner._generate_chunk_id_aliases(chunk_id):
                alias_map.setdefault(alias, chunk_id)
        return alias_map

    @staticmethod
    def _generate_chunk_id_aliases(chunk_id_text: str) -> List[str]:
        if not isinstance(chunk_id_text, str):
            return []

        aliases = []
        stripped = chunk_id_text.strip()
        quote_stripped = stripped.strip("\"'` ")
        for candidate in [chunk_id_text, stripped, quote_stripped]:
            if candidate == "":
                continue
            aliases.append(candidate)
            aliases.append(candidate.lower())

        for extracted in re.findall(r"chunk-[A-Za-z0-9_-]+", chunk_id_text):
            aliases.append(extracted)
            aliases.append(extracted.lower())

        # Preserve order while removing duplicates.
        return list(dict.fromkeys(aliases))

    def _resolve_model_chunk_ids(
        self,
        raw_chunk_ids: List[Any],
        allowed_chunk_ids: Set[str],
        chunk_id_alias_map: Dict[str, str],
    ) -> List[str]:
        resolved_chunk_ids = []
        seen_chunk_ids = set()

        if isinstance(raw_chunk_ids, str):
            raw_chunk_ids = [raw_chunk_ids]
        elif not isinstance(raw_chunk_ids, list):
            raw_chunk_ids = []

        for raw_chunk_id in raw_chunk_ids:
            if not isinstance(raw_chunk_id, str):
                continue

            canonical_chunk_id = None
            if raw_chunk_id in allowed_chunk_ids:
                canonical_chunk_id = raw_chunk_id
            else:
                for alias in self._generate_chunk_id_aliases(raw_chunk_id):
                    matched_chunk_id = chunk_id_alias_map.get(alias)
                    if matched_chunk_id is not None:
                        canonical_chunk_id = matched_chunk_id
                        break

            if canonical_chunk_id is None or canonical_chunk_id in seen_chunk_ids:
                continue
            resolved_chunk_ids.append(canonical_chunk_id)
            seen_chunk_ids.add(canonical_chunk_id)

        return resolved_chunk_ids

    def _build_retry_candidate_batch(
        self,
        pending_records: List[CandidatePairRecord],
        original_chunk_ids: List[str],
    ) -> CandidateBatch:
        max_chunks = LEGACY_PAIRWISE_MAX_CHUNKS_PER_CALL
        pending_chunk_set = {
            chunk_id
            for record in pending_records
            for chunk_id in record.chunk_ids
        }

        # Preserve the original merged chunk ordering as much as possible so the
        # retry prompt stays focused, then backfill any missing direct chunks.
        ordered_chunk_ids = [
            chunk_id
            for chunk_id in original_chunk_ids
            if chunk_id in pending_chunk_set
        ]
        if len(ordered_chunk_ids) < max_chunks:
            seen_chunk_ids = set(ordered_chunk_ids)
            for record in pending_records:
                for chunk_id in record.chunk_ids:
                    if chunk_id in seen_chunk_ids:
                        continue
                    ordered_chunk_ids.append(chunk_id)
                    seen_chunk_ids.add(chunk_id)
                    if len(ordered_chunk_ids) >= max_chunks:
                        break
                if len(ordered_chunk_ids) >= max_chunks:
                    break

        return CandidateBatch(
            pair_records=list(pending_records),
            chunk_ids=ordered_chunk_ids[:max_chunks],
        )

    def _augment_graph_with_triples(self, mined_triples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if len(mined_triples) == 0:
            return []

        unique_entities = set()
        fact_strings = []
        added_triples = []

        for triple_item in mined_triples:
            triple = tuple(text_processing(list(triple_item["triple"])))
            if len(triple) != 3 or triple[0] == triple[2]:
                continue

            fact_strings.append(str(triple))
            unique_entities.update([triple[0], triple[2]])

            added_triples.append(
                {
                    "triple": triple,
                    "chunk_ids": sorted(set(triple_item["chunk_ids"])),
                    "score": triple_item["score"],
                }
            )

        if len(added_triples) == 0:
            return []

        self.hipporag.hidden_triplet_entity_store.insert_strings(list(unique_entities))
        self.hipporag.hidden_triplet_fact_store.insert_strings(fact_strings)
        self._save_hidden_triplet_results(added_triples)
        all_augmented_triples = self._load_saved_hidden_triplet_results()
        augmented_openie_docs = self._build_augmented_openie_docs(all_augmented_triples)
        self._save_augmented_openie_results(augmented_openie_docs)
        self._rebuild_augmented_graph_from_openie(augmented_openie_docs)
        return added_triples

    def _load_saved_hidden_triplet_results(self) -> List[Dict[str, Any]]:
        if not os.path.isfile(self.hidden_triplet_results_path):
            return []
        with open(self.hidden_triplet_results_path, "r") as f:
            payload = json.load(f)
        return payload.get("triples", [])

    def _build_augmented_openie_docs(self, hidden_triplet_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        chunk_to_rows = self.hipporag.chunk_embedding_store.get_all_id_to_rows()
        base_openie_docs, _ = self.hipporag.load_existing_openie([])

        # Start from the persisted base OpenIE output and explicitly add empty
        # chunk entries so the reconstructed augmented graph follows the same
        # chunk-driven pipeline as the original index() flow.
        docs_by_chunk_id: Dict[str, Dict[str, Any]] = {}
        entity_sets: Dict[str, Set[str]] = {}
        triple_sets: Dict[str, Set[Tuple[str, str, str]]] = {}

        for doc in base_openie_docs:
            chunk_id = doc["idx"]
            normalized_doc = {
                "idx": chunk_id,
                "passage": doc["passage"],
                "extracted_entities": list(doc.get("extracted_entities", [])),
                "extracted_triples": [list(triple) for triple in doc.get("extracted_triples", [])],
            }
            docs_by_chunk_id[chunk_id] = normalized_doc
            entity_sets[chunk_id] = {
                text_processing(entity)
                for entity in normalized_doc["extracted_entities"]
                if text_processing(entity) != ""
            }
            triple_sets[chunk_id] = {
                tuple(text_processing(list(triple)))
                for triple in normalized_doc["extracted_triples"]
                if len(triple) == 3
            }

        for chunk_id, row in chunk_to_rows.items():
            if chunk_id in docs_by_chunk_id:
                continue
            docs_by_chunk_id[chunk_id] = {
                "idx": chunk_id,
                "passage": row["content"],
                "extracted_entities": [],
                "extracted_triples": [],
            }
            entity_sets[chunk_id] = set()
            triple_sets[chunk_id] = set()

        for triple_item in hidden_triplet_results:
            triple = tuple(text_processing(list(triple_item.get("triple", []))))
            if len(triple) != 3 or triple[0] == triple[2]:
                continue
            chunk_ids = sorted(set(triple_item.get("chunk_ids", [])))
            for chunk_id in chunk_ids:
                if chunk_id not in docs_by_chunk_id:
                    continue
                doc = docs_by_chunk_id[chunk_id]
                if triple not in triple_sets[chunk_id]:
                    doc["extracted_triples"].append(list(triple))
                    triple_sets[chunk_id].add(triple)
                for entity in (triple[0], triple[2]):
                    if entity not in entity_sets[chunk_id]:
                        doc["extracted_entities"].append(entity)
                        entity_sets[chunk_id].add(entity)

        return [docs_by_chunk_id[chunk_id] for chunk_id in chunk_to_rows.keys()]

    def _save_augmented_openie_results(self, augmented_openie_docs: List[Dict[str, Any]]):
        sum_phrase_chars = sum(
            len(entity)
            for chunk in augmented_openie_docs
            for entity in chunk.get("extracted_entities", [])
        )
        sum_phrase_words = sum(
            len(entity.split())
            for chunk in augmented_openie_docs
            for entity in chunk.get("extracted_entities", [])
        )
        num_phrases = sum(len(chunk.get("extracted_entities", [])) for chunk in augmented_openie_docs)
        if num_phrases > 0:
            avg_ent_chars = round(sum_phrase_chars / num_phrases, 4)
            avg_ent_words = round(sum_phrase_words / num_phrases, 4)
        else:
            avg_ent_chars = 0
            avg_ent_words = 0

        payload = {
            "docs": augmented_openie_docs,
            "avg_ent_chars": avg_ent_chars,
            "avg_ent_words": avg_ent_words,
        }
        with open(self.hipporag.hidden_triplet_augmented_openie_results_path, "w") as f:
            json.dump(payload, f)
        logger.info(
            "Saved hidden-triplet augmented OpenIE results to %s",
            self.hipporag.hidden_triplet_augmented_openie_results_path,
        )

    def _build_combined_entity_store(self) -> InMemoryEntityEmbeddingStore:
        combined_rows = self.hipporag.entity_embedding_store.get_all_id_to_rows()
        combined_rows.update(self.hipporag.hidden_triplet_entity_store.get_all_id_to_rows())
        combined_ids = list(combined_rows.keys())
        combined_embeddings = self.hipporag.get_entity_embeddings_by_ids(combined_ids)
        return InMemoryEntityEmbeddingStore(combined_rows, combined_embeddings)

    def _rebuild_augmented_graph_from_openie(self, augmented_openie_docs: List[Dict[str, Any]]):
        chunk_to_rows = self.hipporag.chunk_embedding_store.get_all_id_to_rows()
        chunk_ids = list(chunk_to_rows.keys())
        _, triple_results_dict = reformat_openie_results(augmented_openie_docs)
        chunk_triples = [
            [text_processing(triple) for triple in triple_results_dict[chunk_id].triples]
            for chunk_id in chunk_ids
        ]
        _, chunk_triple_entities = extract_entity_nodes(chunk_triples)

        original_graph = self.hipporag.graph
        original_entity_store = self.hipporag.entity_embedding_store
        original_node_to_node_stats = getattr(self.hipporag, "node_to_node_stats", None)
        original_ent_node_to_chunk_ids = self.hipporag.ent_node_to_chunk_ids
        original_entity_id_to_row = getattr(self.hipporag, "entity_id_to_row", None)

        combined_entity_store = self._build_combined_entity_store()

        try:
            # Rebuild the augmented graph through the same chunk -> triples ->
            # fact edges / passage edges / synonymy edges pipeline as the
            # original HippoRAG index() path, but keep everything isolated from
            # the base graph and base files.
            self.hipporag.graph = ig.Graph(directed=self.global_config.is_directed_graph)
            self.hipporag.entity_embedding_store = combined_entity_store
            self.hipporag.node_to_node_stats = {}
            self.hipporag.ent_node_to_chunk_ids = {}

            self.hipporag.add_fact_edges(chunk_ids, chunk_triples)
            self.hipporag.add_passage_edges(chunk_ids, chunk_triple_entities)
            self.hipporag.add_synonymy_edges()
            self.hipporag.add_new_nodes()
            self.hipporag.add_new_edges()

            augmented_graph = self.hipporag.graph
        finally:
            self.hipporag.graph = original_graph
            self.hipporag.entity_embedding_store = original_entity_store
            self.hipporag.node_to_node_stats = original_node_to_node_stats
            self.hipporag.ent_node_to_chunk_ids = original_ent_node_to_chunk_ids
            self.hipporag.entity_id_to_row = original_entity_id_to_row

        augmented_graph.write_pickle(self.augmented_graph_path)
        logger.info(
            "Rebuilt hidden-triplet augmented graph via the original OpenIE graph pipeline and saved it to %s",
            self.augmented_graph_path,
        )

    def _save_hidden_triplet_results(self, added_triples: List[Dict[str, Any]]):
        if os.path.isfile(self.hidden_triplet_results_path):
            with open(self.hidden_triplet_results_path, "r") as f:
                payload = json.load(f)
        else:
            payload = {"triples": []}

        existing = {}
        for triple_item in payload.get("triples", []):
            triple_key = tuple(triple_item.get("triple", []))
            if len(triple_key) != 3:
                continue
            existing[triple_key] = triple_item

        for triple_item in added_triples:
            triple_key = tuple(triple_item["triple"])
            if triple_key not in existing:
                existing[triple_key] = {
                    "triple": list(triple_item["triple"]),
                    "chunk_ids": sorted(set(triple_item["chunk_ids"])),
                    "score": round(float(triple_item["score"]), 6),
                }
            else:
                existing[triple_key]["chunk_ids"] = sorted(
                    set(existing[triple_key].get("chunk_ids", [])) | set(triple_item["chunk_ids"])
                )
                existing[triple_key]["score"] = max(
                    float(existing[triple_key].get("score", 0.0)),
                    float(triple_item["score"]),
                )

        payload["triples"] = list(existing.values())
        with open(self.hidden_triplet_results_path, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info("Saved hidden-triplet augmented facts to %s", self.hidden_triplet_results_path)

    def _save_gnn_triple_dump(
        self,
        summary: Dict[str, Any],
        added_triples: List[Dict[str, Any]],
        batch_logs: List[Dict[str, Any]],
    ):
        if os.path.isfile(self.hidden_triplet_gnn_triples_dump_path):
            with open(self.hidden_triplet_gnn_triples_dump_path, "r") as f:
                payload = json.load(f)
        else:
            payload = {"runs": []}

        payload.setdefault("runs", []).append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "dataset": self.global_config.dataset,
                "summary": dict(summary),
                "triples": [
                    {
                        "triple": list(triple_item["triple"]),
                        "chunk_ids": list(triple_item["chunk_ids"]),
                        "score": round(float(triple_item["score"]), 6),
                    }
                    for triple_item in added_triples
                ],
                "batch_logs": batch_logs,
            }
        )

        with open(self.hidden_triplet_gnn_triples_dump_path, "w") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        logger.info(
            "Saved GNN hidden-triplet dump to %s",
            self.hidden_triplet_gnn_triples_dump_path,
        )

    def _save_audit(
        self,
        graph_data: Dict[str, Any],
        train_output: Dict[str, Any],
        candidate_records: List[CandidatePairRecord],
        candidate_batches: List[CandidateBatch],
        batch_logs: List[Dict[str, Any]],
        added_triples: List[Dict[str, Any]],
        total_mining_time_sec: float,
    ):
        if os.path.isfile(self.audit_path):
            with open(self.audit_path, "r") as f:
                audit = json.load(f)
        else:
            audit = {"runs": []}

        audit["runs"].append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "num_fact_edges": len(graph_data["fact_edges"]),
                "num_synonym_edges": len(graph_data["synonym_edges"]),
                "legacy_pairwise_candidate_cap": graph_data.get("legacy_pairwise_candidate_cap"),
                "hidden_triplet_config_signature": self.hipporag.get_hidden_triplet_mining_config_signature(),
                "hidden_triplet_candidate_generation": graph_data.get(
                    "hidden_triplet_candidate_generation_stats", {}
                ),
                "gnn_training_time_sec": round(float(train_output["training_time_sec"]), 6),
                "gnn_train_loss": round(float(train_output["train_loss"]), 6),
                "gnn_eval_loss": None if train_output["eval_loss"] is None else round(float(train_output["eval_loss"]), 6),
                "gnn_eval_accuracy": None if train_output["eval_accuracy"] is None else round(float(train_output["eval_accuracy"]), 6),
                "gnn_eval_pairwise_accuracy": None if train_output["eval_pairwise_accuracy"] is None else round(float(train_output["eval_pairwise_accuracy"]), 6),
                "gnn_encoder_type": self.global_config.hidden_triplet_gnn_encoder_type,
                "gnn_training_logs": train_output["training_logs"],
                "num_masked_fact_edges": int(train_output["num_masked_fact_edges"]),
                "num_support_fact_edges": int(train_output["num_support_fact_edges"]),
                "num_eval_fact_edges": int(train_output["num_eval_fact_edges"]),
                "total_mining_time_sec": round(float(total_mining_time_sec), 6),
                "candidate_records": [
                    {
                        "pair_ids": record.pair_ids,
                        "pair_names": record.pair_names,
                        "score": round(record.score, 6),
                        "chunk_ids": record.chunk_ids,
                    }
                    for record in candidate_records
                ],
                "candidate_batches": [
                    {
                        "pair_names": [record.pair_names for record in batch.pair_records],
                        "chunk_ids": batch.chunk_ids,
                    }
                    for batch in candidate_batches
                ],
                "batch_logs": batch_logs,
                "added_triples": [
                    {
                        "triple": triple_item["triple"],
                        "chunk_ids": triple_item["chunk_ids"],
                        "score": round(float(triple_item["score"]), 6),
                    }
                    for triple_item in added_triples
                ],
            }
        )

        with open(self.audit_path, "w") as f:
            json.dump(audit, f, indent=2)

    def _build_support_graph(
        self,
        graph_data: Dict[str, Any],
        support_fact_pairs: List[Tuple[str, str]],
    ) -> Data:
        weighted_typed_pairs = [
            (pair, float(graph_data["fact_edges"][pair]), FACT_EDGE_TYPE)
            for pair in support_fact_pairs
        ]
        weighted_typed_pairs.extend(
            (pair, float(weight), SYNONYM_EDGE_TYPE)
            for pair, weight in graph_data["synonym_edges"].items()
        )

        edge_index, edge_weight, edge_type = self._build_weighted_edge_index(
            weighted_typed_pairs=weighted_typed_pairs,
            node_key_to_idx=graph_data["node_key_to_idx"],
        )

        return Data(
            x=torch.tensor(graph_data["entity_embeddings"], dtype=torch.float32, device=self.device),
            edge_index=edge_index.to(self.device),
            edge_weight=edge_weight.to(self.device),
            edge_type=edge_type.to(self.device),
            num_nodes=len(graph_data["entity_node_keys"]),
        )

    def _sample_negative_edges(
        self,
        num_nodes: int,
        num_positive_edges: int,
        negatives_per_positive: int,
        graph_data: Dict[str, Any],
        rng: random.Random,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        target_num_negatives = max(1, num_positive_edges * negatives_per_positive)
        observed_index_pairs = graph_data.get("observed_index_pairs")
        if observed_index_pairs is None:
            observed_index_pairs = self._build_observed_index_pairs(graph_data)
            graph_data["observed_index_pairs"] = observed_index_pairs
        seen_pairs: Set[Tuple[int, int]] = set()
        sampled_pairs = self._sample_random_negative_pairs(
            num_nodes=num_nodes,
            target_count=target_num_negatives,
            observed_index_pairs=observed_index_pairs,
            seen_pairs=seen_pairs,
            rng=rng,
        )

        if len(sampled_pairs) == 0:
            raise RuntimeError("Failed to sample negative edges for contrastive hidden-triplet training.")

        negative_edge_index = self._index_pairs_to_edge_index(sampled_pairs[:target_num_negatives]).to(self.device)
        sampling_stats = {
            "target_num_negatives": int(target_num_negatives),
            "sampled_random_negatives": int(negative_edge_index.size(1)),
        }
        return negative_edge_index, sampling_stats

    @staticmethod
    def _build_observed_index_pairs(graph_data: Dict[str, Any]) -> Set[Tuple[int, int]]:
        node_key_to_idx = graph_data["node_key_to_idx"]
        return {
            (
                min(node_key_to_idx[left_key], node_key_to_idx[right_key]),
                max(node_key_to_idx[left_key], node_key_to_idx[right_key]),
            )
            for left_key, right_key in graph_data["observed_pairs"]
        }

    @staticmethod
    def _index_pairs_to_edge_index(index_pairs: List[Tuple[int, int]]) -> torch.Tensor:
        src_nodes = [pair[0] for pair in index_pairs]
        dst_nodes = [pair[1] for pair in index_pairs]
        return torch.tensor([src_nodes, dst_nodes], dtype=torch.long)

    @staticmethod
    def _sample_random_negative_pairs(
        num_nodes: int,
        target_count: int,
        observed_index_pairs: Set[Tuple[int, int]],
        seen_pairs: Set[Tuple[int, int]],
        rng: random.Random,
    ) -> List[Tuple[int, int]]:
        if target_count <= 0:
            return []

        sampled_pairs: List[Tuple[int, int]] = []
        max_attempts = max(1000, target_count * 100)
        attempts = 0
        while len(sampled_pairs) < target_count and attempts < max_attempts:
            attempts += 1
            src_idx = rng.randrange(num_nodes)
            dst_idx = rng.randrange(num_nodes - 1)
            if dst_idx >= src_idx:
                dst_idx += 1
            pair = (min(src_idx, dst_idx), max(src_idx, dst_idx))
            if pair in observed_index_pairs or pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            sampled_pairs.append(pair)
        return sampled_pairs

    @staticmethod
    def _contrastive_loss(
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor,
        negatives_per_positive: int,
        temperature: float,
    ) -> torch.Tensor:
        pos_logits = pos_scores.view(-1, 1)
        neg_logits = neg_scores.view(-1, negatives_per_positive)
        logits = torch.cat([pos_logits, neg_logits], dim=1) / temperature
        targets = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, targets)

    @staticmethod
    def _evaluate_contrastive_link_prediction(
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor,
        negatives_per_positive: int,
        temperature: float,
    ) -> Dict[str, float]:
        eval_loss = GNNHiddenTripletMiner._contrastive_loss(
            pos_scores=pos_scores,
            neg_scores=neg_scores,
            negatives_per_positive=negatives_per_positive,
            temperature=temperature,
        ).item()

        pos_probs = torch.sigmoid(pos_scores)
        neg_probs = torch.sigmoid(neg_scores)
        binary_correct = (pos_probs >= 0.5).sum().item() + (neg_probs < 0.5).sum().item()
        binary_total = pos_probs.numel() + neg_probs.numel()
        eval_accuracy = binary_correct / max(1, binary_total)

        reshaped_neg_scores = neg_scores.view(-1, negatives_per_positive)
        pairwise_accuracy = (pos_scores.view(-1, 1) > reshaped_neg_scores).all(dim=1).float().mean().item()

        return {
            "eval_loss": eval_loss,
            "eval_accuracy": eval_accuracy,
            "eval_pairwise_accuracy": pairwise_accuracy,
        }

    @staticmethod
    def _build_weighted_edge_index(
        weighted_typed_pairs: List[Tuple[Tuple[str, str], float, int]],
        node_key_to_idx: Dict[str, int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        src_nodes = []
        dst_nodes = []
        weights = []
        edge_types = []

        for pair, weight, edge_type in weighted_typed_pairs:
            src_idx = node_key_to_idx[pair[0]]
            dst_idx = node_key_to_idx[pair[1]]
            src_nodes.extend([src_idx, dst_idx])
            dst_nodes.extend([dst_idx, src_idx])
            weights.extend([weight, weight])
            edge_types.extend([edge_type, edge_type])

        edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
        edge_weight = torch.tensor(weights, dtype=torch.float32)
        edge_type = torch.tensor(edge_types, dtype=torch.long)
        return edge_index, edge_weight, edge_type

    @staticmethod
    def _pairs_to_edge_index(
        pairs: List[Tuple[str, str]],
        node_key_to_idx: Dict[str, int],
        undirected: bool,
    ) -> torch.Tensor:
        src_nodes = []
        dst_nodes = []

        for pair in pairs:
            src_idx = node_key_to_idx[pair[0]]
            dst_idx = node_key_to_idx[pair[1]]
            src_nodes.append(src_idx)
            dst_nodes.append(dst_idx)
            if undirected:
                src_nodes.append(dst_idx)
                dst_nodes.append(src_idx)

        return torch.tensor([src_nodes, dst_nodes], dtype=torch.long)

    def _build_observed_neighbor_tensors(
        self,
        observed_neighbors: Dict[str, Set[str]],
        node_key_to_idx: Dict[str, int],
    ) -> Dict[int, torch.Tensor]:
        neighbor_tensors = {}
        for src_key, dst_keys in observed_neighbors.items():
            src_idx = node_key_to_idx[src_key]
            neighbor_tensors[src_idx] = torch.tensor(
                sorted(node_key_to_idx[dst_key] for dst_key in dst_keys),
                dtype=torch.long,
                device=self.device,
            )
        return neighbor_tensors

    @staticmethod
    def _is_entity_node(node_key: str) -> bool:
        return isinstance(node_key, str) and node_key.startswith("entity-")

    @staticmethod
    def _is_valid_training_entity_text(entity_text: str) -> bool:
        return isinstance(entity_text, str) and entity_text.strip() != ""

    @staticmethod
    def _is_four_digit_year(entity_text: str) -> bool:
        stripped = entity_text.strip()
        return stripped.isdigit() and len(stripped) == 4 and 1000 <= int(stripped) <= 2099

    def _pair_has_shared_chunk(self, pair: Tuple[str, str]) -> bool:
        left_chunk_ids = self.hipporag.ent_node_to_chunk_ids.get(pair[0], set())
        right_chunk_ids = self.hipporag.ent_node_to_chunk_ids.get(pair[1], set())
        return len(left_chunk_ids.intersection(right_chunk_ids)) > 0

    @classmethod
    def _should_filter_candidate_entity_text(cls, entity_text: str) -> bool:
        if not cls._is_valid_training_entity_text(entity_text):
            return True

        normalized = text_processing(entity_text)
        if normalized == "":
            return True
        if normalized.isdigit():
            return True
        if cls._is_four_digit_year(normalized):
            return True
        if len(normalized) <= 1:
            return True

        return False

    @staticmethod
    def _is_fact_edge(weight: float) -> bool:
        return weight >= 1

    @staticmethod
    def _is_synonym_edge(weight: float) -> bool:
        return weight < 1 and not float(weight).is_integer()
