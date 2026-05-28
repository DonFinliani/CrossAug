from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from typing import Any

import numpy as np

from lightrag.utils import logger

from .completion_llm import SubgraphCompletionLLM
from .config import SubgraphCompletionConfig
from .graph_adapter import LightRAGGraphAdapter, summarize_snapshot
from .injector import LightRAGRelationInjector
from .model import SubgraphMissingnessTrainer
from .schemas import SubgraphCompletionBatch
from .subgraph_sampler import SubgraphSampler


def _json_safe(payload: Any) -> Any:
    if hasattr(payload, "__dataclass_fields__"):
        return _json_safe(asdict(payload))
    if isinstance(payload, dict):
        return {str(key): _json_safe(value) for key, value in payload.items()}
    if isinstance(payload, (list, tuple, set)):
        return [_json_safe(value) for value in payload]
    if isinstance(payload, np.generic):
        return payload.item()
    if isinstance(payload, np.ndarray):
        return payload.tolist()
    if isinstance(payload, (str, int, float, bool)) or payload is None:
        return payload
    return str(payload)


def _safe_json_dump(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, ensure_ascii=False, indent=2)


def _artifact_path(artifact_dir: str, stem: str) -> str:
    return os.path.join(artifact_dir, f"{stem}_gnn.json")


def _config_signature(config: SubgraphCompletionConfig) -> dict[str, Any]:
    return {
        key: value
        for key, value in asdict(config).items()
        if key != "artifact_dir"
    }


class LightRAGSubgraphCompletionMiner:
    """End-to-end GNN subgraph completion for an initialized LightRAG instance."""

    def __init__(self, config: SubgraphCompletionConfig | None = None) -> None:
        self.config = config or SubgraphCompletionConfig.from_env()

    async def run(self, rag: Any) -> dict[str, Any]:
        if not self.config.enabled:
            return {"enabled": False, "message": "GNN subgraph completion disabled."}
        mining_start_time = time.perf_counter()

        if self.config.artifact_dir is None:
            self.config.artifact_dir = os.path.join(
                rag.working_dir, "gnn_subgraph_completion"
            )
        os.makedirs(self.config.artifact_dir, exist_ok=True)

        snapshot = await LightRAGGraphAdapter.from_rag(rag, self.config)
        snapshot_summary = summarize_snapshot(snapshot)
        _safe_json_dump(
            _artifact_path(self.config.artifact_dir, "snapshot_summary"),
            snapshot_summary,
        )

        if len(snapshot.entity_data) < self.config.subgraph_min_nodes:
            return {
                "enabled": True,
                "status": "skipped",
                "reason": "not_enough_entities",
                "snapshot": snapshot_summary,
            }

        node_features = await LightRAGGraphAdapter.embed_snapshot_nodes(
            rag,
            snapshot,
            self.config,
        )

        sampler = SubgraphSampler(snapshot, node_features, self.config)
        graph_data = sampler.graph_data
        if int(graph_data.get("num_fact_edges", 0)) == 0:
            total_mining_time_sec = time.perf_counter() - mining_start_time
            return {
                "enabled": True,
                "status": "skipped",
                "reason": "no_fact_edges",
                "snapshot": snapshot_summary,
                "hidden_triplet_mining_strategy": "subgraph_missingness",
                "num_graph_nodes": int(graph_data.get("num_nodes", 0)),
                "num_graph_edges": int(graph_data.get("num_edges", 0)),
                "num_fact_edges": 0,
                "num_synonym_edges": int(graph_data.get("num_synonym_edges", 0)),
                "num_entity_chunk_edges": int(graph_data.get("num_entity_chunk_edges", 0)),
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
                "total_mining_time_sec": round(total_mining_time_sec, 4),
                "hidden_triplet_mining_cache_hit": False,
                "hidden_triplet_mining_skipped": True,
                "hidden_triplet_mining_graph_source": "lightrag_graph",
                "hidden_triplet_config_signature": _config_signature(self.config),
            }

        trainer = SubgraphMissingnessTrainer(graph_data, self.config)
        training_output = trainer.fit(sampler)
        training_summary = {
            key: value
            for key, value in training_output.items()
            if key != "model"
        }
        _safe_json_dump(
            _artifact_path(self.config.artifact_dir, "gnn_training_summary"),
            training_summary,
        )

        inference_records, selection_stats = sampler.sample_inference_records()
        scored_records = trainer.score_records(inference_records)
        selected_records = sampler.select_scored_records(scored_records, selection_stats)
        _safe_json_dump(
            _artifact_path(self.config.artifact_dir, "gnn_selected_subgraphs"),
            [asdict(record) for record in selected_records],
        )
        logger.info(
            "Selected %d/%d subgraphs above missingness threshold",
            len(selected_records),
            len(scored_records),
        )

        completion_batches = [
            SubgraphCompletionBatch(
                subgraph=record,
                chunk_ids=list(record.chunk_ids),
                known_triples=list(record.known_triples),
            )
            for record in selected_records
            if len(record.chunk_ids) > 0
        ]
        completed_relations, batch_logs = await SubgraphCompletionLLM(self.config).complete_many(
            rag,
            snapshot,
            completion_batches,
        )
        _safe_json_dump(
            _artifact_path(self.config.artifact_dir, "gnn_completed_relations"),
            [asdict(relation) for relation in completed_relations],
        )
        _safe_json_dump(
            _artifact_path(self.config.artifact_dir, "gnn_completion_batch_logs"),
            batch_logs,
        )

        injection_report = await LightRAGRelationInjector(self.config).inject(
            rag,
            snapshot,
            completed_relations,
        )
        _safe_json_dump(
            _artifact_path(self.config.artifact_dir, "gnn_injection_report"),
            injection_report,
        )

        successful_injections = [
            item
            for item in injection_report
            if item.get("status") in {"inserted_new_edge", "merged_existing_edge", "dry_run"}
        ]
        total_mining_time_sec = time.perf_counter() - mining_start_time
        summary = {
            "enabled": True,
            "status": "completed",
            "artifact_dir": self.config.artifact_dir,
            "snapshot": snapshot_summary,
            "training": training_summary,
            "inference_subgraphs": len(inference_records),
            "selected_subgraphs": len(selected_records),
            "completed_relations": len(completed_relations),
            "injection_report": injection_report,
            "hidden_triplet_mining_strategy": "subgraph_missingness",
            "num_graph_nodes": int(graph_data["num_nodes"]),
            "num_graph_edges": int(graph_data["num_edges"]),
            "num_fact_edges": int(graph_data["num_fact_edges"]),
            "num_synonym_edges": int(graph_data["num_synonym_edges"]),
            "num_entity_chunk_edges": int(graph_data["num_entity_chunk_edges"]),
            "num_candidate_pairs": 0,
            "num_candidate_subgraphs": len(selected_records),
            "num_llm_batches": len(batch_logs),
            "num_mined_triples": len(completed_relations),
            "num_new_triples": len(successful_injections),
            "hidden_triplet_candidate_generation": graph_data.get(
                "hidden_triplet_subgraph_selection_stats",
                {},
            ),
            "resolved_train_roots_per_epoch": int(
                training_output["resolved_train_roots_per_epoch"]
            ),
            "resolved_eval_roots": int(training_output["resolved_eval_roots"]),
            "gnn_training_time_sec": round(float(training_output["training_time_sec"]), 4),
            "gnn_train_loss": round(float(training_output["train_loss"]), 6),
            "gnn_eval_loss": (
                None
                if training_output["eval_loss"] is None
                else round(float(training_output["eval_loss"]), 6)
            ),
            "gnn_best_eval_loss": (
                None
                if training_output["best_eval_loss"] is None
                else round(float(training_output["best_eval_loss"]), 6)
            ),
            "gnn_eval_accuracy": (
                None
                if training_output["eval_accuracy"] is None
                else round(float(training_output["eval_accuracy"]), 6)
            ),
            "gnn_eval_pairwise_accuracy": None,
            "gnn_eval_ranking_accuracy": (
                None
                if training_output["eval_ranking_accuracy"] is None
                else round(float(training_output["eval_ranking_accuracy"]), 6)
            ),
            "gnn_eval_f1": (
                None
                if training_output["eval_f1"] is None
                else round(float(training_output["eval_f1"]), 6)
            ),
            "gnn_encoder_type": self.config.encoder_type,
            "gnn_completed_epochs": int(training_output["completed_epochs"]),
            "gnn_early_stopped": bool(training_output["early_stopped"]),
            "gnn_best_epoch": training_output["best_epoch"],
            "gnn_best_eval_pairwise_accuracy": None,
            "gnn_best_eval_ranking_accuracy": (
                None
                if training_output["best_eval_ranking_accuracy"] is None
                else round(float(training_output["best_eval_ranking_accuracy"]), 6)
            ),
            "total_mining_time_sec": round(total_mining_time_sec, 4),
            "hidden_triplet_mining_cache_hit": False,
            "hidden_triplet_mining_skipped": False,
            "hidden_triplet_mining_graph_source": "lightrag_graph",
            "gnn_reproduction_profile": self.config.reproduction_profile,
            "hidden_triplet_config_signature": _config_signature(self.config),
        }
        _safe_json_dump(
            _artifact_path(self.config.artifact_dir, "hidden_triplet_mining_audit"),
            {
                "runs": [
                    {
                        **summary,
                        "selected_subgraphs_detail": [
                            asdict(record) for record in selected_records
                        ],
                        "completion_batch_logs": batch_logs,
                        "completed_relations_detail": [
                            asdict(relation) for relation in completed_relations
                        ],
                    }
                ]
            },
        )
        logger.info("GNN subgraph completion summary: %s", summary)
        return summary


async def augment_lightrag_with_subgraph_completion(
    rag: Any,
    config: SubgraphCompletionConfig | None = None,
) -> dict[str, Any]:
    miner = LightRAGSubgraphCompletionMiner(config)
    return await miner.run(rag)
