import hashlib
import json
import logging
import os
import os.path as osp
import sys
import warnings

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch_geometric.data import Data, InMemoryDataset, makedirs
from torch_geometric.data.dataset import _repr, files_exist

from gfmrag.kg_construction.utils import KG_DELIMITER
from gfmrag.text_emb_models import BaseTextEmbModel
from gfmrag.utils import get_rank

logger = logging.getLogger(__name__)


class KGDataset(InMemoryDataset):
    delimiter = KG_DELIMITER

    def __init__(
        self,
        root: str,
        data_name: str,
        text_emb_model_cfgs: DictConfig,
        force_rebuild: bool = False,
        **kwargs: str,
    ) -> None:
        self.name = data_name
        self.force_rebuild = force_rebuild
        self.fingerprint = hashlib.md5(
            json.dumps(
                OmegaConf.to_container(text_emb_model_cfgs, resolve=True)
            ).encode()
        ).hexdigest()
        self.text_emb_model_cfgs = text_emb_model_cfgs
        super().__init__(root, None, None)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        self.feat_dim = self._data.rel_emb.size(1)

    @property
    def raw_file_names(self) -> list:
        return ["kg.txt"]

    @property
    def raw_dir(self) -> str:
        return os.path.join(str(self.root), str(self.name), "processed", "stage1")

    @property
    def processed_dir(self) -> str:
        return os.path.join(
            str(self.root),
            str(self.name),
            "processed",
            "stage2",
            self.fingerprint,
        )

    @property
    def processed_file_names(self) -> str:
        return "data.pt"

    @property
    def num_relations(self) -> int:
        return int(self.data.edge_type.max()) + 1

    def __repr__(self) -> str:
        return f"{self.name}()"

    def load_file(
        self,
        triplet_file: str,
        inv_entity_vocab: dict,
        inv_rel_vocab: dict,
    ) -> dict:
        triplets = []
        entity_cnt, rel_cnt = len(inv_entity_vocab), len(inv_rel_vocab)

        with open(triplet_file, encoding="utf-8") as fin:
            for line in fin:
                try:
                    u, r, v = (
                        line.split()
                        if self.delimiter is None
                        else line.strip().split(self.delimiter)
                    )
                except Exception as e:
                    logger.error("Error in line: %s, %s, skipping", line, e)
                    continue
                if u not in inv_entity_vocab:
                    inv_entity_vocab[u] = entity_cnt
                    entity_cnt += 1
                if v not in inv_entity_vocab:
                    inv_entity_vocab[v] = entity_cnt
                    entity_cnt += 1
                if r not in inv_rel_vocab:
                    inv_rel_vocab[r] = rel_cnt
                    rel_cnt += 1
                triplets.append((inv_entity_vocab[u], inv_entity_vocab[v], inv_rel_vocab[r]))

        return {
            "triplets": triplets,
            "num_node": len(inv_entity_vocab),
            "num_relation": rel_cnt,
            "inv_entity_vocab": inv_entity_vocab,
            "inv_rel_vocab": inv_rel_vocab,
        }

    def _process(self) -> None:
        f = osp.join(self.processed_dir, "pre_transform.pt")
        if osp.exists(f) and torch.load(f, weights_only=False) != _repr(
            self.pre_transform
        ):
            warnings.warn(
                f"The `pre_transform` argument differs from the one used in {self.processed_dir}",
                stacklevel=1,
            )

        f = osp.join(self.processed_dir, "pre_filter.pt")
        if osp.exists(f) and torch.load(f, weights_only=False) != _repr(
            self.pre_filter
        ):
            warnings.warn(
                f"The `pre_filter` argument differs from the one used in {self.processed_dir}",
                stacklevel=1,
            )

        if self.force_rebuild or not files_exist(self.processed_paths):
            logger.warning("Processing KG dataset %s at rank %s", self.name, get_rank())
            if self.log and "pytest" not in sys.modules:
                print("Processing...", file=sys.stderr)

            makedirs(self.processed_dir)
            self.process()
            torch.save(_repr(self.pre_transform), osp.join(self.processed_dir, "pre_transform.pt"))
            torch.save(_repr(self.pre_filter), osp.join(self.processed_dir, "pre_filter.pt"))

            if self.log and "pytest" not in sys.modules:
                print("Done!", file=sys.stderr)

    def process(self) -> None:
        kg_file = self.raw_paths[0]
        kg_result = self.load_file(kg_file, inv_entity_vocab={}, inv_rel_vocab={})

        num_node = kg_result["num_node"]
        num_relations = kg_result["num_relation"]
        kg_triplets = kg_result["triplets"]

        train_target_edges = torch.tensor(
            [[t[0], t[1]] for t in kg_triplets],
            dtype=torch.long,
        ).t()
        train_target_etypes = torch.tensor([t[2] for t in kg_triplets])

        train_edges = torch.cat([train_target_edges, train_target_edges.flip(0)], dim=1)
        train_etypes = torch.cat(
            [train_target_etypes, train_target_etypes + num_relations]
        )

        with open(os.path.join(self.processed_dir, "ent2id.json"), "w") as f:
            json.dump(kg_result["inv_entity_vocab"], f)
        rel2id = kg_result["inv_rel_vocab"]
        id2rel = {v: k for k, v in rel2id.items()}
        for etype in train_etypes:
            if etype.item() >= num_relations:
                raw_etype = etype.item() - num_relations
                rel2id["inverse_" + id2rel[raw_etype]] = etype.item()
        with open(os.path.join(self.processed_dir, "rel2id.json"), "w") as f:
            json.dump(rel2id, f)

        logger.info("Generating relation embeddings")
        text_emb_model: BaseTextEmbModel = instantiate(self.text_emb_model_cfgs)
        rel_emb = text_emb_model.encode(list(rel2id.keys()), is_query=False).cpu()

        kg_data = Data(
            edge_index=train_edges,
            edge_type=train_etypes,
            num_nodes=num_node,
            target_edge_index=train_target_edges,
            target_edge_type=train_target_etypes,
            num_relations=num_relations * 2,
            rel_emb=rel_emb,
        )
        torch.save(self.collate([kg_data]), self.processed_paths[0])

        with open(os.path.join(self.processed_dir, "text_emb_model_cfgs.json"), "w") as f:
            json.dump(OmegaConf.to_container(self.text_emb_model_cfgs), f, indent=4)
