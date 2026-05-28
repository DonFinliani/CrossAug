from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


REPRODUCTION_COMMON_PROFILE: dict[str, Any] = {
    "hidden_dim": 256,
    "encoder_type": "rgcn",
    "epochs": 200,
    "learning_rate": 2e-3,
    "graph_refresh_interval": 10,
    "early_stopping_patience": 50,
    "early_stopping_min_delta": 1e-4,
    "max_chars_per_chunk": 4500,
    "llm_max_tokens": 12000,
    "llm_concurrency": 64,
    "subgraph_walk_length": 3,
    "subgraph_fact_walk_scale": 1.0,
    "subgraph_synonym_walk_scale": 0.5,
    "subgraph_chunk_walk_scale": 1.0,
    "subgraph_walk_weight_clip": 5.0,
    "subgraph_train_roots_per_epoch": 128,
    "subgraph_eval_roots": 64,
    "subgraph_fact_mask_ratio": 0.2,
    "subgraph_entity_delete_ratio": 0.08,
    "subgraph_min_nodes": 4,
    "subgraph_min_fact_edges": 1,
    "subgraph_overlap_threshold": 0.5,
    "subgraph_max_chunks_per_prompt": 15,
    "subgraph_max_known_triples_per_prompt": 120,
    "subgraph_max_entities_per_prompt": 120,
}

REPRODUCTION_DATASET_PROFILES: dict[str, dict[str, Any]] = {
    "literaryqa": {
        "subgraph_walks_per_root": 20,
        "subgraph_negative_walks_per_root": 30,
        "subgraph_inference_root_budget": 512.0,
        "subgraph_missing_score_threshold": 0.5,
        "subgraph_llm_budget": 100.0,
    },
    "musique": {
        "subgraph_walks_per_root": 20,
        "subgraph_negative_walks_per_root": 30,
        "subgraph_inference_root_budget": 0.1,
        "subgraph_missing_score_threshold": 0.5,
        "subgraph_llm_budget": 0.1,
    },
    "hotpotqa": {
        "subgraph_walks_per_root": 20,
        "subgraph_negative_walks_per_root": 30,
        "subgraph_inference_root_budget": 0.1,
        "subgraph_missing_score_threshold": 0.5,
        "subgraph_llm_budget": 0.1,
    },
    "2wikimultihopqa": {
        "subgraph_walks_per_root": 20,
        "subgraph_negative_walks_per_root": 30,
        "subgraph_inference_root_budget": 0.1,
        "subgraph_missing_score_threshold": 0.5,
        "subgraph_llm_budget": 0.1,
    },
}


def normalize_reproduction_dataset_name(dataset_name: str) -> str:
    normalized = str(dataset_name or "").lower()
    if normalized in {"2wiki", "2wikimultihopqa"}:
        return "2wikimultihopqa"
    return normalized


@dataclass(slots=True)
class SubgraphCompletionConfig:
    """Configuration for LightRAG subgraph completion.

    The defaults mirror HippoRAG's hidden-triplet subgraph miner. Differences
    should be limited to LightRAG storage and model-calling boundaries, while
    every algorithmic knob remains overridable through environment variables.
    """

    enabled: bool = True
    artifact_dir: str | None = None
    random_seed: int = 42
    device: str | None = None

    embedding_batch_size: int = 64
    max_embedding_text_chars: int = 1800

    synonymy_edge_topk: int = 2047
    synonymy_edge_query_batch_size: int = 1000
    synonymy_edge_key_batch_size: int = 10000
    synonymy_edge_sim_threshold: float = 0.8

    hidden_dim: int = 256
    encoder_type: str = "rgcn"
    epochs: int = 200
    learning_rate: float = 2e-3
    graph_refresh_interval: int = 10
    early_stopping_patience: int = 50
    early_stopping_min_delta: float = 1e-4

    max_chars_per_chunk: int = 4500
    llm_max_tokens: int = 12000
    llm_concurrency: int = 64

    subgraph_walk_length: int = 3
    subgraph_walks_per_root: int = 50
    subgraph_negative_walks_per_root: int = 60
    subgraph_fact_walk_scale: float = 1.0
    subgraph_synonym_walk_scale: float = 0.5
    subgraph_chunk_walk_scale: float = 1.0
    subgraph_walk_weight_clip: float = 5.0
    subgraph_train_roots_per_epoch: int = 128
    subgraph_eval_roots: int = 64
    subgraph_fact_mask_ratio: float = 0.2
    subgraph_entity_delete_ratio: float = 0.08
    subgraph_min_nodes: int = 4
    subgraph_min_fact_edges: int = 1
    # If >= 1, this is an absolute count. If in (0, 1), it is a fraction of
    # entity nodes, matching HippoRAG's hidden_triplet_subgraph_inference_root_budget.
    subgraph_inference_root_budget: float = 512.0
    subgraph_missing_score_threshold: float = 0.5
    # If >= 1, this is an absolute count. If in (0, 1), it is a fraction of
    # 2 * chunk nodes, matching HippoRAG's hidden_triplet_subgraph_llm_budget.
    subgraph_llm_budget: float = 100.0
    subgraph_overlap_threshold: float = 0.5
    subgraph_max_chunks_per_prompt: int = 15
    subgraph_max_known_triples_per_prompt: int = 120
    subgraph_max_entities_per_prompt: int = 120

    reproduction_profile_name: str | None = None
    reproduction_profile: dict[str, Any] = field(default_factory=dict)

    dry_run: bool = False

    @classmethod
    def from_reproduction_profile(
        cls,
        dataset_name: str,
        artifact_dir: str | None = None,
    ) -> "SubgraphCompletionConfig":
        config = cls(artifact_dir=artifact_dir)
        config.apply_reproduction_profile(dataset_name)
        return config

    def apply_reproduction_profile(self, dataset_name: str) -> None:
        profile_dataset = normalize_reproduction_dataset_name(dataset_name)
        parameters = dict(REPRODUCTION_COMMON_PROFILE)
        parameters.update(
            REPRODUCTION_DATASET_PROFILES.get(
                profile_dataset,
                REPRODUCTION_DATASET_PROFILES["literaryqa"],
            )
        )
        for key, value in parameters.items():
            setattr(self, key, value)
        self.reproduction_profile_name = f"lightrag:{profile_dataset}"
        self.reproduction_profile = {
            "framework": "LightRAG",
            "dataset": profile_dataset,
            "parameters": dict(sorted(parameters.items())),
        }

    @classmethod
    def from_env(cls, artifact_dir: str | None = None) -> "SubgraphCompletionConfig":
        walks_per_root = int(os.getenv("GNN_SUBGRAPH_WALKS_PER_ROOT", 20))
        return cls(
            enabled=_env_bool("ENABLE_GNN_SUBGRAPH_COMPLETION", True),
            artifact_dir=artifact_dir or os.getenv("GNN_SUBGRAPH_ARTIFACT_DIR"),
            random_seed=int(os.getenv("GNN_SUBGRAPH_RANDOM_SEED", 42)),
            device=os.getenv("GNN_SUBGRAPH_DEVICE") or None,
            embedding_batch_size=int(os.getenv("GNN_SUBGRAPH_EMBEDDING_BATCH_SIZE", 64)),
            max_embedding_text_chars=int(
                os.getenv("GNN_SUBGRAPH_MAX_EMBEDDING_TEXT_CHARS", 1800)
            ),
            synonymy_edge_topk=int(
                os.getenv(
                    "GNN_SYNONYMY_EDGE_TOPK",
                    os.getenv("SYNONYMY_EDGE_TOPK", 2047),
                )
            ),
            synonymy_edge_query_batch_size=int(
                os.getenv(
                    "GNN_SYNONYMY_EDGE_QUERY_BATCH_SIZE",
                    os.getenv("SYNONYMY_EDGE_QUERY_BATCH_SIZE", 1000),
                )
            ),
            synonymy_edge_key_batch_size=int(
                os.getenv(
                    "GNN_SYNONYMY_EDGE_KEY_BATCH_SIZE",
                    os.getenv("SYNONYMY_EDGE_KEY_BATCH_SIZE", 10000),
                )
            ),
            synonymy_edge_sim_threshold=float(
                os.getenv(
                    "GNN_SYNONYMY_EDGE_SIM_THRESHOLD",
                    os.getenv("SYNONYMY_EDGE_SIM_THRESHOLD", 0.8),
                )
            ),
            hidden_dim=int(os.getenv("GNN_HIDDEN_DIM", 256)),
            encoder_type=os.getenv("GNN_ENCODER_TYPE", "rgcn"),
            epochs=int(os.getenv("GNN_EPOCHS", 200)),
            learning_rate=float(os.getenv("GNN_LEARNING_RATE", 2e-3)),
            graph_refresh_interval=int(os.getenv("GNN_GRAPH_REFRESH_INTERVAL", 10)),
            early_stopping_patience=int(
                os.getenv("GNN_EARLY_STOPPING_PATIENCE", 50)
            ),
            early_stopping_min_delta=float(
                os.getenv("GNN_EARLY_STOPPING_MIN_DELTA", 1e-4)
            ),
            max_chars_per_chunk=int(os.getenv("GNN_MAX_CHARS_PER_CHUNK", 4500)),
            llm_max_tokens=int(os.getenv("GNN_LLM_MAX_TOKENS", 12000)),
            llm_concurrency=int(os.getenv("GNN_LLM_CONCURRENCY", 64)),
            subgraph_walk_length=int(os.getenv("GNN_SUBGRAPH_WALK_LENGTH", 3)),
            subgraph_walks_per_root=walks_per_root,
            subgraph_negative_walks_per_root=int(
                os.getenv(
                    "GNN_SUBGRAPH_NEGATIVE_WALKS_PER_ROOT",
                    max(walks_per_root, 30),
                )
            ),
            subgraph_fact_walk_scale=float(
                os.getenv("GNN_SUBGRAPH_FACT_WALK_SCALE", 1.0)
            ),
            subgraph_synonym_walk_scale=float(
                os.getenv("GNN_SUBGRAPH_SYNONYM_WALK_SCALE", 0.5)
            ),
            subgraph_chunk_walk_scale=float(
                os.getenv("GNN_SUBGRAPH_CHUNK_WALK_SCALE", 1.0)
            ),
            subgraph_walk_weight_clip=float(
                os.getenv("GNN_SUBGRAPH_WALK_WEIGHT_CLIP", 5.0)
            ),
            subgraph_train_roots_per_epoch=int(
                os.getenv("GNN_SUBGRAPH_TRAIN_ROOTS_PER_EPOCH", 128)
            ),
            subgraph_eval_roots=int(os.getenv("GNN_SUBGRAPH_EVAL_ROOTS", 64)),
            subgraph_fact_mask_ratio=float(
                os.getenv("GNN_SUBGRAPH_FACT_MASK_RATIO", 0.2)
            ),
            subgraph_entity_delete_ratio=float(
                os.getenv("GNN_SUBGRAPH_ENTITY_DELETE_RATIO", 0.08)
            ),
            subgraph_min_nodes=int(os.getenv("GNN_SUBGRAPH_MIN_NODES", 4)),
            subgraph_min_fact_edges=int(
                os.getenv("GNN_SUBGRAPH_MIN_FACT_EDGES", 1)
            ),
            subgraph_inference_root_budget=float(
                os.getenv("GNN_SUBGRAPH_INFERENCE_ROOT_BUDGET", 0.1)
            ),
            subgraph_missing_score_threshold=float(
                os.getenv("GNN_SUBGRAPH_MISSING_SCORE_THRESHOLD", 0.5)
            ),
            subgraph_llm_budget=float(os.getenv("GNN_SUBGRAPH_LLM_BUDGET", 0.1)),
            subgraph_overlap_threshold=float(
                os.getenv("GNN_SUBGRAPH_OVERLAP_THRESHOLD", 0.5)
            ),
            subgraph_max_chunks_per_prompt=int(
                os.getenv("GNN_SUBGRAPH_MAX_CHUNKS_PER_PROMPT", 15)
            ),
            subgraph_max_known_triples_per_prompt=int(
                os.getenv("GNN_SUBGRAPH_MAX_KNOWN_TRIPLES_PER_PROMPT", 120)
            ),
            subgraph_max_entities_per_prompt=int(
                os.getenv("GNN_SUBGRAPH_MAX_ENTITIES_PER_PROMPT", 120)
            ),
            dry_run=_env_bool("GNN_SUBGRAPH_DRY_RUN", False),
        )
