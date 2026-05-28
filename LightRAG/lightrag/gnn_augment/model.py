from __future__ import annotations

import os
import random
import time
from typing import Any

import numpy as np

from lightrag.utils import logger

from .config import SubgraphCompletionConfig
from .schemas import (
    NUM_SUBGRAPH_EDGE_TYPES,
    NUM_SUBGRAPH_NODE_TYPES,
    SampledSubgraphRecord,
)


def _require_pyg():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch_geometric.data import Batch, Data
        from torch_geometric.nn import GCNConv, SAGEConv, global_max_pool, global_mean_pool
        try:
            from torch_geometric.nn import GATv2Conv
        except ImportError:
            GATv2Conv = None
        try:
            from torch_geometric.nn import RGCNConv
        except ImportError:
            RGCNConv = None
    except ImportError as exc:
        raise ImportError(
            "GNN subgraph completion requires PyTorch and PyG. Install with "
            "`pip install -e .[gnn]` or install torch and torch-geometric."
        ) from exc
    return torch, nn, F, Batch, Data, GCNConv, SAGEConv, global_max_pool, global_mean_pool, GATv2Conv, RGCNConv


def set_gnn_training_seed(seed: int) -> None:
    torch, *_ = _require_pyg()
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


class PyGSubgraphMissingnessClassifier:
    """HippoRAG-compatible graph-level subgraph missingness classifier."""

    def __new__(cls, input_dim: int, hidden_dim: int, encoder_type: str = "rgcn"):
        (
            _torch,
            nn,
            F,
            _Batch,
            _Data,
            GCNConv,
            SAGEConv,
            global_max_pool,
            global_mean_pool,
            GATv2Conv,
            RGCNConv,
        ) = _require_pyg()

        class _Classifier(nn.Module):
            def __init__(self) -> None:
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
                        f"Unsupported GNN encoder type: {encoder_type}. "
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

            def _message_pass(self, hidden, data):
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

            def forward(self, data):
                node_types = getattr(data, "node_type", None)
                if node_types is None:
                    node_types = _torch.zeros(data.x.size(0), dtype=_torch.long, device=data.x.device)
                hidden = self.input_projection(data.x) + self.node_type_embedding(node_types.long())
                hidden = self._message_pass(hidden, data)

                batch = getattr(data, "batch", None)
                if batch is None:
                    batch = _torch.zeros(hidden.size(0), dtype=_torch.long, device=hidden.device)
                pooled = _torch.cat([global_mean_pool(hidden, batch), global_max_pool(hidden, batch)], dim=-1)
                return self.classifier(pooled).view(-1)

        return _Classifier()


def build_subgraph_data(
    graph_data: dict[str, Any],
    node_indices: list[int],
    label: float,
    removed_edge_keys: set[tuple[int, int, int]] | None = None,
    removed_node_indices: set[int] | None = None,
    corruption_type: int = 0,
):
    torch, _nn, _F, _Batch, Data, *_ = _require_pyg()
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


class SubgraphMissingnessTrainer:
    def __init__(
        self,
        graph_data: dict[str, Any],
        config: SubgraphCompletionConfig,
    ) -> None:
        torch, _nn, _F, _Batch, _Data, *_ = _require_pyg()
        self.torch = torch
        self.config = config
        self.device = torch.device(
            config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.graph_data = graph_data
        self.model = PyGSubgraphMissingnessClassifier(
            input_dim=int(graph_data["node_features"].shape[1]),
            hidden_dim=int(config.hidden_dim),
            encoder_type=config.encoder_type,
        ).to(self.device)

    def _evaluate(self, groups: list[dict[str, Any]]) -> dict[str, float | None]:
        torch, _nn, F, Batch, _Data, *_ = _require_pyg()
        if not groups:
            return {
                "eval_loss": None,
                "eval_accuracy": None,
                "eval_f1": None,
                "eval_ranking_accuracy": None,
            }

        from .subgraph_sampler import SubgraphSampler

        data_list, ranking_pairs = SubgraphSampler.flatten_subgraph_groups(groups)
        batch = Batch.from_data_list(data_list).to(self.device)
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            logits = self.model(batch)
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
            self.model.train()
        return {
            "eval_loss": eval_loss,
            "eval_accuracy": accuracy,
            "eval_f1": f1,
            "eval_ranking_accuracy": ranking_accuracy,
        }

    def fit(self, sampler) -> dict[str, Any]:
        torch, _nn, F, Batch, _Data, *_ = _require_pyg()
        train_start_time = time.perf_counter()
        rng_seed = self.config.random_seed
        set_gnn_training_seed(int(rng_seed))
        rng = random.Random(rng_seed)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.config.learning_rate))

        num_epochs = max(1, int(self.config.epochs))
        roots_per_epoch = sampler.resolve_root_sample_count(
            self.config.subgraph_train_roots_per_epoch,
            allow_zero=False,
        )
        eval_roots = sampler.resolve_root_sample_count(
            self.config.subgraph_eval_roots,
            allow_zero=True,
        )
        graph_refresh_interval = max(1, int(self.config.graph_refresh_interval))
        early_stopping_patience = max(0, int(self.config.early_stopping_patience))
        early_stopping_min_delta = max(0.0, float(self.config.early_stopping_min_delta))
        early_stopping_enabled = early_stopping_patience > 0 and eval_roots > 0

        eval_groups = (
            sampler.sample_training_groups(
                rng=random.Random(rng_seed + 104729),
                num_roots=eval_roots,
                prefix="eval",
            )
            if eval_roots > 0
            else []
        )

        training_logs: list[dict[str, Any]] = []
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
            self.model.train()
            train_groups = sampler.sample_training_groups(
                rng=rng,
                num_roots=roots_per_epoch,
                prefix=f"train-e{epoch_idx + 1}",
            )
            data_list, _ranking_pairs = sampler.flatten_subgraph_groups(train_groups)
            batch = Batch.from_data_list(data_list).to(self.device)
            logits = self.model(batch)
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
                last_eval_metrics = self._evaluate(eval_groups)
                current_eval_loss = last_eval_metrics["eval_loss"]
                current_ranking = last_eval_metrics["eval_ranking_accuracy"]
                is_best = False
                if current_eval_loss is not None:
                    improved = best_eval_loss is None or current_eval_loss < best_eval_loss - early_stopping_min_delta
                    if improved:
                        best_eval_loss = current_eval_loss
                        best_eval_ranking_accuracy = current_ranking
                        best_epoch = completed_epochs
                        best_state_dict = {
                            key: value.detach().cpu().clone()
                            for key, value in self.model.state_dict().items()
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
                    "Subgraph GNN epoch %d/%d: train_loss=%.6f bce=%.6f eval_loss=%s eval_accuracy=%s eval_f1=%s eval_ranking_accuracy=%s",
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
            self.model.load_state_dict(best_state_dict)
            last_eval_metrics = self._evaluate(eval_groups)

        training_time_sec = time.perf_counter() - train_start_time
        return {
            "model": self.model,
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

    def score_records(self, records: list[SampledSubgraphRecord]) -> list[SampledSubgraphRecord]:
        torch, _nn, _F, Batch, _Data, *_ = _require_pyg()
        if not records:
            return []
        data_list = [
            build_subgraph_data(
                self.graph_data,
                node_indices=record.node_indices,
                label=0.0,
                corruption_type=0,
            )
            for record in records
        ]
        batch = Batch.from_data_list(data_list).to(self.device)
        self.model.eval()
        with torch.no_grad():
            scores = torch.sigmoid(self.model(batch)).detach().cpu().numpy().tolist()
        for record, score in zip(records, scores):
            record.missing_score = float(score)
        return records
