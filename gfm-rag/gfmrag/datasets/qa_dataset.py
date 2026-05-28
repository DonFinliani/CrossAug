import hashlib
import json
import logging
import os
import os.path as osp
import sys
import warnings

import datasets
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils import data as torch_data
from torch_geometric.data import InMemoryDataset, makedirs
from torch_geometric.data.dataset import _repr, files_exist

from gfmrag.datasets.kg_dataset import KGDataset
from gfmrag.text_emb_models import BaseTextEmbModel
from gfmrag.utils import entities_to_mask, get_rank

logger = logging.getLogger(__name__)


class QADataset(InMemoryDataset):
    def __init__(
        self,
        root: str,
        data_name: str,
        text_emb_model_cfgs: DictConfig,
        force_rebuild: bool = False,
    ):
        self.name = data_name
        self.force_rebuild = force_rebuild
        self.text_emb_model_cfgs = text_emb_model_cfgs
        self.fingerprint = hashlib.md5(
            json.dumps(
                OmegaConf.to_container(text_emb_model_cfgs, resolve=True)
            ).encode()
        ).hexdigest()
        kg = KGDataset(root, data_name, text_emb_model_cfgs, force_rebuild)
        self.kg = kg[0]
        self.feat_dim = kg.feat_dim
        super().__init__(root, None, None)
        self.data = torch.load(self.processed_paths[0], weights_only=False)
        self.load_property()

    @property
    def raw_file_names(self) -> list:
        return ["train.json", "test.json"]

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
        return "qa_data.pt"

    def __repr__(self) -> str:
        return f"{self.name}()"

    def load_property(self) -> None:
        with open(os.path.join(self.processed_dir, "ent2id.json")) as fin:
            self.ent2id = json.load(fin)
        with open(os.path.join(self.processed_dir, "rel2id.json")) as fin:
            self.rel2id = json.load(fin)
        with open(
            os.path.join(str(self.root), str(self.name), "raw", "dataset_corpus.json")
        ) as fin:
            self.doc = json.load(fin)
        with open(os.path.join(self.raw_dir, "document2entities.json")) as fin:
            self.doc2entities = json.load(fin)
        self.raw_train_data = self._load_raw_split("train.json")
        self.raw_test_data = self._load_raw_split("test.json")
        self.ent2docs = torch.load(
            os.path.join(self.processed_dir, "ent2doc.pt"),
            weights_only=True,
        )
        self.id2doc = {i: doc for i, doc in enumerate(self.doc2entities)}

    def _load_raw_split(self, file_name: str) -> list:
        path = os.path.join(self.raw_dir, file_name)
        if not os.path.exists(path):
            return []
        with open(path) as fin:
            return json.load(fin)

    def _raw_files_newer_than_processed(self) -> bool:
        if not files_exist(self.processed_paths):
            return False
        processed_mtime = min(os.path.getmtime(path) for path in self.processed_paths)
        for path in self.raw_paths:
            if os.path.exists(path) and os.path.getmtime(path) > processed_mtime:
                return True
        return False

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

        if (
            self.force_rebuild
            or not files_exist(self.processed_paths)
            or self._raw_files_newer_than_processed()
        ):
            logger.warning("Processing QA dataset %s at rank %s", self.name, get_rank())
            if self.log and "pytest" not in sys.modules:
                print("Processing...", file=sys.stderr)

            makedirs(self.processed_dir)
            self.process()
            torch.save(_repr(self.pre_transform), osp.join(self.processed_dir, "pre_transform.pt"))
            torch.save(_repr(self.pre_filter), osp.join(self.processed_dir, "pre_filter.pt"))

            if self.log and "pytest" not in sys.modules:
                print("Done!", file=sys.stderr)

    def process(self) -> None:
        with open(os.path.join(self.processed_dir, "ent2id.json")) as fin:
            self.ent2id = json.load(fin)
        with open(os.path.join(self.raw_dir, "document2entities.json")) as fin:
            self.doc2entities = json.load(fin)

        num_nodes = self.kg.num_nodes
        doc2id = {doc: i for i, doc in enumerate(self.doc2entities)}
        n_docs = len(self.doc2entities)
        doc2ent = torch.zeros((n_docs, num_nodes))
        for doc, entities in self.doc2entities.items():
            entity_ids = [self.ent2id[ent] for ent in entities if ent in self.ent2id]
            if entity_ids:
                doc2ent[doc2id[doc], entity_ids] = 1
        ent2doc = doc2ent.T.to_sparse()
        torch.save(ent2doc, os.path.join(self.processed_dir, "ent2doc.pt"))

        sample_id = []
        questions = []
        question_entities_masks = []
        supporting_entities_masks = []
        supporting_docs_masks = []
        num_samples = []

        for path in self.raw_paths:
            if not os.path.exists(path):
                num_samples.append(0)
                continue
            with open(path) as fin:
                data = json.load(fin)

            is_train = os.path.basename(path) == "train.json"
            num_sample = 0
            for index, item in enumerate(data):
                question_entities = [
                    self.ent2id[x]
                    for x in item.get("question_entities", [])
                    if x in self.ent2id
                ]
                supporting_entities = [
                    self.ent2id[x]
                    for x in item.get("supporting_entities", [])
                    if x in self.ent2id
                ]
                supporting_docs = [
                    doc2id[doc]
                    for doc in item.get("supporting_facts", [])
                    if doc in doc2id
                ]

                if not question_entities:
                    logger.warning(
                        "Sample %s in %s has no linked question entities; using an all-zero seed mask.",
                        item.get("id", index),
                        path,
                    )
                if is_train and (
                    not question_entities or not supporting_entities or not supporting_docs
                ):
                    continue

                num_sample += 1
                sample_id.append(index)
                questions.append(item["question"])
                question_entities_masks.append(entities_to_mask(question_entities, num_nodes))
                supporting_entities_masks.append(entities_to_mask(supporting_entities, num_nodes))
                supporting_docs_masks.append(entities_to_mask(supporting_docs, n_docs))
            num_samples.append(num_sample)

        logger.info("Generating question embeddings")
        text_emb_model: BaseTextEmbModel = instantiate(self.text_emb_model_cfgs)
        if questions:
            question_embeddings = text_emb_model.encode(questions, is_query=True).cpu()
            question_entities_masks_tensor = torch.stack(question_entities_masks)
            supporting_entities_masks_tensor = torch.stack(supporting_entities_masks)
            supporting_docs_masks_tensor = torch.stack(supporting_docs_masks)
            sample_id_tensor = torch.tensor(sample_id, dtype=torch.long)
        else:
            feat_dim = getattr(text_emb_model, "feat_dim", None)
            if feat_dim is None:
                feat_dim = self.kg.rel_emb.size(1)
            question_embeddings = torch.empty((0, feat_dim))
            question_entities_masks_tensor = torch.empty((0, num_nodes))
            supporting_entities_masks_tensor = torch.empty((0, num_nodes))
            supporting_docs_masks_tensor = torch.empty((0, n_docs))
            sample_id_tensor = torch.empty((0,), dtype=torch.long)

        dataset = datasets.Dataset.from_dict(
            {
                "question_embeddings": question_embeddings,
                "question_entities_masks": question_entities_masks_tensor,
                "supporting_entities_masks": supporting_entities_masks_tensor,
                "supporting_docs_masks": supporting_docs_masks_tensor,
                "sample_id": sample_id_tensor,
            }
        ).with_format("torch")

        offset = 0
        splits = []
        for num_sample in num_samples:
            split = torch_data.Subset(dataset, range(offset, offset + num_sample))
            splits.append(split)
            offset += num_sample
        torch.save(splits, self.processed_paths[0])

        with open(os.path.join(self.processed_dir, "text_emb_model_cfgs.json"), "w") as f:
            json.dump(OmegaConf.to_container(self.text_emb_model_cfgs), f, indent=4)
