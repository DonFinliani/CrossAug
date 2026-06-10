#!/usr/bin/env python3
"""Convert HippoRAG graph outputs into GFM-RAG pre-built stage1 layout.

The required GFM-RAG layout is:

    root/data_name/processed/stage1/nodes.csv
    root/data_name/processed/stage1/relations.csv
    root/data_name/processed/stage1/edges.csv

This script also writes compatibility files used by the current GFM-RAG code:

    root/data_name/processed/stage1/kg.txt
    root/data_name/processed/stage1/document2entities.json
    root/data_name/raw/dataset_corpus.json

HippoRAG stores the relation text in the OpenIE JSON files, while the igraph
pickle mainly stores hashed node ids and unlabeled weighted edges. Therefore the
CSV graph is built from the base or GNN-augmented OpenIE results and the pickle
paths are checked and recorded in manifest.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SCRIPT_PATH = Path(__file__).resolve()
GFM_RAG_ROOT = SCRIPT_PATH.parents[1]
CROSSAUG_ROOT = GFM_RAG_ROOT.parent
DEFAULT_HIPPORAG_ROOT = CROSSAUG_ROOT / "HippoRAG"
DEFAULT_HIPPORAG_OUTPUTS = DEFAULT_HIPPORAG_ROOT / "outputs"
DEFAULT_OUTPUT_ROOT = GFM_RAG_ROOT / "data" / "hipporag_stage1_exports"
DEFAULT_QA_SOURCE_DIR = DEFAULT_HIPPORAG_ROOT / "reproduce" / "dataset"
DEFAULT_LITERARYQA_SOURCE_DIR = DEFAULT_QA_SOURCE_DIR / "literaryqa" / "books"
DEFAULT_MODEL_LABEL = "Models_Qwen3-32B"
DEFAULT_MODEL_DIR_GLOB = "Models_Qwen3-32B*"

DATASETS: dict[str, str] = {
    "literaryqa": "literaryqa",
    "musique": "musique",
    "hotpotqa": "hotpotqa",
    "2wikimultihopqa": "2wikimultihopqa",
}


@dataclass(frozen=True)
class FlowSpec:
    name: str
    graph_file: str
    openie_template: str

    def openie_file(self, model_label: str) -> str:
        return self.openie_template.format(model_label=model_label)


FLOWS: dict[str, FlowSpec] = {
    "base": FlowSpec(
        name="base",
        graph_file="graph.pickle",
        openie_template="openie_results_ner_{model_label}.json",
    ),
    "gnn": FlowSpec(
        name="gnn",
        graph_file="graph_hidden_triplet_augmented.pickle",
        openie_template="openie_results_hidden_triplet_augmented_ner_{model_label}.json",
    ),
}

MENTIONED_IN_RELATION = "mentioned_in"
COMPAT_KG_DELIMITER = ","


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export HippoRAG base/GNN graphs to GFM-RAG stage1 CSV layout."
    )
    parser.add_argument(
        "--hipporag-outputs",
        type=Path,
        default=DEFAULT_HIPPORAG_OUTPUTS,
        help=f"HippoRAG outputs root. Default: {DEFAULT_HIPPORAG_OUTPUTS}",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Destination stage1 root. Default: {DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASETS),
        default=sorted(DATASETS),
        help="Datasets to export.",
    )
    parser.add_argument(
        "--flows",
        nargs="+",
        choices=sorted(FLOWS),
        default=sorted(FLOWS),
        help="Graph flows to export.",
    )
    parser.add_argument(
        "--model-label",
        default=DEFAULT_MODEL_LABEL,
        help=f"Model label embedded in OpenIE filenames. Default: {DEFAULT_MODEL_LABEL}",
    )
    parser.add_argument(
        "--model-dir-glob",
        default=DEFAULT_MODEL_DIR_GLOB,
        help=f"Glob used to locate HippoRAG model run dirs. Default: {DEFAULT_MODEL_DIR_GLOB}",
    )
    parser.add_argument(
        "--literaryqa-mode",
        choices=["per-book", "combined"],
        default="combined",
        help=(
            "combined writes one literaryqa_base/literaryqa_gnn data_name; "
            "per-book keeps LiteraryQA graph isolation."
        ),
    )
    parser.add_argument(
        "--max-literary-books",
        type=int,
        default=None,
        help="Optional debug limit for LiteraryQA books.",
    )
    parser.add_argument(
        "--literaryqa-source-dir",
        type=Path,
        default=DEFAULT_LITERARYQA_SOURCE_DIR,
        help=(
            "Directory with per-book LiteraryQA source JSON files used to write "
            f"processed/stage1/test.json. Default: {DEFAULT_LITERARYQA_SOURCE_DIR}"
        ),
    )
    parser.add_argument(
        "--qa-source-dir",
        type=Path,
        default=DEFAULT_QA_SOURCE_DIR,
        help=(
            "Directory with hotpotqa.json/musique.json/2wikimultihopqa.json used "
            f"to write processed/stage1/test.json. Default: {DEFAULT_QA_SOURCE_DIR}"
        ),
    )
    parser.add_argument(
        "--no-compat-files",
        action="store_true",
        help="Only write CSV files, not kg.txt/document2entities.json/raw corpus.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned exports without writing files.",
    )
    return parser.parse_args()


def json_dumps(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def compact_title(text: str, max_len: int = 120) -> str:
    first_line = " ".join(clean_text(text).split())
    if len(first_line) <= max_len:
        return first_line
    return first_line[: max_len - 3].rstrip() + "..."


def compat_phrase(value: str) -> str:
    """Match the current GFM-RAG kg.txt assumptions: comma-free simple text."""

    return re.sub(r"[^A-Za-z0-9 ]", " ", clean_text(value).lower()).strip()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "docs" not in data:
        raise ValueError(f"{path} is not a HippoRAG OpenIE JSON with a docs field")
    return data


def iter_docs(openie_data: dict[str, Any]) -> Iterable[dict[str, Any]]:
    docs = openie_data.get("docs", [])
    if not isinstance(docs, list):
        raise ValueError("OpenIE docs field must be a list")
    for doc in docs:
        if isinstance(doc, dict):
            yield doc


def normalize_triple(triple: Any) -> tuple[str, str, str] | None:
    if not isinstance(triple, (list, tuple)) or len(triple) < 3:
        return None
    source = clean_text(triple[0])
    relation = clean_text(triple[1])
    target = clean_text(triple[2])
    if not source or not relation or not target:
        return None
    return source, relation, target


class Stage1Graph:
    def __init__(self) -> None:
        self.nodes: dict[str, tuple[str, dict[str, Any]]] = {}
        self.relations: dict[str, dict[str, Any]] = {}
        self.edges: dict[tuple[str, str, str, str], tuple[str, str, str, dict[str, Any]]] = {}
        self.document2entities: dict[str, list[str]] = {}
        self.corpus: dict[str, str] = {}
        self.kg_triples: set[tuple[str, str, str]] = set()
        self.source_docs = 0
        self.source_triples = 0
        self.source_entities = 0

    def add_node(
        self,
        name: str,
        node_type: str,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        name = clean_text(name)
        if not name:
            return
        attributes = attributes or {}
        current = self.nodes.get(name)
        if current is None:
            self.nodes[name] = (node_type, dict(attributes))
            return

        current_type, current_attrs = current
        if current_type == "document" or node_type != "document":
            merged_type = current_type
        else:
            merged_type = node_type
        merged_attrs = dict(current_attrs)
        merged_attrs.update(attributes)
        self.nodes[name] = (merged_type, merged_attrs)

    def add_relation(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        name = clean_text(name)
        if not name:
            return
        current = self.relations.get(name, {})
        merged = dict(current)
        merged.update(attributes or {})
        self.relations[name] = merged

    def add_edge(
        self,
        source: str,
        relation: str,
        target: str,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        source = clean_text(source)
        relation = clean_text(relation)
        target = clean_text(target)
        if not source or not relation or not target:
            return
        attributes = attributes or {}
        self.add_node(source, "entity")
        self.add_node(target, "entity")
        self.add_relation(relation)
        attr_text = json_dumps(attributes)
        key = (source, relation, target, attr_text)
        self.edges[key] = (source, relation, target, dict(attributes))

        kg_source = compat_phrase(source)
        kg_relation = compat_phrase(relation)
        kg_target = compat_phrase(target)
        if kg_source and kg_relation and kg_target:
            self.kg_triples.add((kg_source, kg_relation, kg_target))

    def add_openie_doc(self, doc: dict[str, Any], book_id: str | None = None) -> None:
        doc_id = clean_text(doc.get("idx"))
        if not doc_id:
            doc_id = f"document_{self.source_docs}"
        passage = clean_text(doc.get("passage"))
        doc_attrs: dict[str, Any] = {
            "idx": doc_id,
            "title": compact_title(passage),
            "text": passage,
        }
        if book_id:
            doc_attrs["book_id"] = book_id
        self.add_node(doc_id, "document", doc_attrs)
        self.corpus[doc_id] = passage
        self.source_docs += 1

        extracted_entities = [
            clean_text(entity)
            for entity in doc.get("extracted_entities", [])
            if clean_text(entity)
        ]
        unique_entities = sorted(set(extracted_entities))
        self.source_entities += len(unique_entities)
        compat_entities = sorted(
            entity for entity in {compat_phrase(x) for x in unique_entities} if entity
        )
        self.document2entities[doc_id] = sorted(
            set(self.document2entities.get(doc_id, [])) | set(compat_entities)
        )

        self.add_relation(
            MENTIONED_IN_RELATION,
            {"description": "An entity is mentioned in the document."},
        )
        for entity in unique_entities:
            self.add_node(entity, "entity", {})
            edge_attrs: dict[str, Any] = {
                "chunk_id": doc_id,
                "source": "extracted_entities",
            }
            if book_id:
                edge_attrs["book_id"] = book_id
            self.add_edge(entity, MENTIONED_IN_RELATION, doc_id, edge_attrs)

        for raw_triple in doc.get("extracted_triples", []):
            triple = normalize_triple(raw_triple)
            if triple is None:
                continue
            source, relation, target = triple
            edge_attrs = {
                "chunk_id": doc_id,
                "source": "openie_triple",
            }
            if book_id:
                edge_attrs["book_id"] = book_id
            self.add_edge(source, relation, target, edge_attrs)
            self.source_triples += 1

    def merge(self, other: "Stage1Graph") -> None:
        for name, (node_type, attrs) in other.nodes.items():
            self.add_node(name, node_type, attrs)
        for name, attrs in other.relations.items():
            self.add_relation(name, attrs)
        for source, relation, target, attrs in other.edges.values():
            self.add_edge(source, relation, target, attrs)
        for doc_id, entities in other.document2entities.items():
            self.document2entities[doc_id] = sorted(
                set(self.document2entities.get(doc_id, [])) | set(entities)
            )
        self.corpus.update(other.corpus)
        self.kg_triples.update(other.kg_triples)
        self.source_docs += other.source_docs
        self.source_triples += other.source_triples
        self.source_entities += other.source_entities

    def counts(self) -> dict[str, int]:
        return {
            "nodes": len(self.nodes),
            "relations": len(self.relations),
            "edges": len(self.edges),
            "documents": len(self.corpus),
            "document2entities": len(self.document2entities),
            "compat_kg_triples": len(self.kg_triples),
            "source_docs": self.source_docs,
            "source_triples": self.source_triples,
            "source_entities": self.source_entities,
        }


def graph_pickle_stats(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    stats: dict[str, Any] = {"exists": True, "path": str(path)}
    try:
        with path.open("rb") as f:
            graph = pickle.load(f)
        stats["vertices"] = int(graph.vcount())
        stats["edges"] = int(graph.ecount())
        stats["vertex_attributes"] = sorted(graph.vs.attributes())
        stats["edge_attributes"] = sorted(graph.es.attributes())
    except Exception as exc:  # pragma: no cover - depends on optional igraph.
        stats["read_error"] = repr(exc)
    return stats


def find_model_dir(dataset_dir: Path, graph_file: str, model_dir_glob: str) -> Path | None:
    candidates = sorted(
        path
        for path in dataset_dir.glob(model_dir_glob)
        if path.is_dir() and (path / graph_file).exists()
    )
    if candidates:
        return candidates[0]
    fallback = sorted(path.parent for path in dataset_dir.glob(f"*/{graph_file}"))
    return fallback[0] if fallback else None


def build_graph_from_openie(openie_path: Path, book_id: str | None = None) -> Stage1Graph:
    openie_data = load_json(openie_path)
    graph = Stage1Graph()
    for doc in iter_docs(openie_data):
        graph.add_openie_doc(doc, book_id=book_id)
    return graph


def graph_entity_vocab(graph: Stage1Graph) -> list[str]:
    entities: set[str] = set()
    for source, _, target in graph.kg_triples:
        entities.add(source)
        entities.add(target)
    for doc_entities in graph.document2entities.values():
        entities.update(doc_entities)
    return sorted((entity for entity in entities if entity), key=lambda x: (-len(x), x))


def match_entities(text: str, entity_vocab: list[str], limit: int = 20) -> list[str]:
    normalized_text = f" {compat_phrase(text)} "
    matches = []
    seen = set()
    for entity in entity_vocab:
        if len(entity) < 3:
            continue
        if f" {entity} " not in normalized_text:
            continue
        if entity in seen:
            continue
        seen.add(entity)
        matches.append(entity)
        if len(matches) >= limit:
            break
    return matches


def load_literaryqa_samples(source_dir: Path, book_id: str) -> list[dict[str, Any]]:
    source_path = source_dir / f"{book_id}.json"
    if not source_path.exists():
        print(f"[WARN] Missing LiteraryQA source samples: {source_path}")
        return []
    with source_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"LiteraryQA source file must contain a list: {source_path}")
    return [sample for sample in data if isinstance(sample, dict)]


def load_qa_samples(source_dir: Path, dataset_name: str) -> list[dict[str, Any]]:
    source_path = source_dir / f"{dataset_name}.json"
    if not source_path.exists():
        print(f"[WARN] Missing QA source samples: {source_path}")
        return []
    with source_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"QA source file must contain a list: {source_path}")
    return [sample for sample in data if isinstance(sample, dict)]


def passage_title(passage: str) -> str:
    passage = clean_text(passage)
    if not passage:
        return ""
    return clean_text(passage.splitlines()[0])


def build_title_to_doc_ids(graph: Stage1Graph) -> dict[str, list[str]]:
    title_to_doc_ids: dict[str, list[str]] = {}
    for doc_id, passage in graph.corpus.items():
        title = passage_title(passage)
        if title:
            title_to_doc_ids.setdefault(title, []).append(doc_id)
        title_to_doc_ids.setdefault(doc_id, []).append(doc_id)
    return title_to_doc_ids


def extract_support_titles(dataset_name: str, sample: dict[str, Any]) -> list[str]:
    titles: list[str] = []
    if dataset_name in {"hotpotqa", "2wikimultihopqa"}:
        for fact in sample.get("supporting_facts", []):
            if isinstance(fact, (list, tuple)) and fact:
                title = clean_text(fact[0])
            else:
                title = clean_text(fact)
            if title:
                titles.append(title)
    elif dataset_name == "musique":
        paragraphs = sample.get("paragraphs", [])
        for hop in sample.get("question_decomposition", []):
            if not isinstance(hop, dict):
                continue
            support_idx = hop.get("paragraph_support_idx")
            if isinstance(support_idx, int) and 0 <= support_idx < len(paragraphs):
                paragraph = paragraphs[support_idx]
                if isinstance(paragraph, dict):
                    title = clean_text(paragraph.get("title"))
                    if title:
                        titles.append(title)
    return titles


def build_qa_test_data(
    dataset_name: str,
    graph: Stage1Graph,
    samples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    entity_vocab = graph_entity_vocab(graph)
    title_to_doc_ids = build_title_to_doc_ids(graph)
    test_data = []
    for idx, sample in enumerate(samples):
        question = clean_text(sample.get("question"))
        answer = clean_text(sample.get("answer"))
        answer_aliases = [
            clean_text(alias)
            for alias in sample.get("answer_aliases", [])
            if clean_text(alias)
        ]
        support_titles = extract_support_titles(dataset_name, sample)
        supporting_facts = sorted(
            {
                doc_id
                for title in support_titles
                for doc_id in title_to_doc_ids.get(title, [])
            }
        )

        supporting_entities = set()
        for doc_id in supporting_facts:
            supporting_entities.update(graph.document2entities.get(doc_id, []))
        for gold_answer in [answer] + answer_aliases:
            supporting_entities.update(match_entities(gold_answer, entity_vocab))

        item = {
            "id": sample.get("_id", sample.get("id", idx)),
            "question": question,
            "answer": answer,
            "answer_aliases": answer_aliases,
            "question_entities": match_entities(question, entity_vocab),
            "supporting_entities": sorted(supporting_entities),
            "supporting_facts": supporting_facts,
        }
        for key in ("type", "level", "answerable"):
            if key in sample:
                item[key] = sample[key]
        if support_titles:
            item["supporting_fact_titles"] = support_titles
        test_data.append(item)
    return test_data


def build_literaryqa_test_data(
    graph: Stage1Graph,
    samples: list[dict[str, Any]],
    book_id: str | None = None,
) -> list[dict[str, Any]]:
    entity_vocab = graph_entity_vocab(graph)
    test_data = []
    for idx, sample in enumerate(samples):
        question = clean_text(sample.get("question"))
        answer = clean_text(sample.get("answer"))
        answer_aliases = [
            clean_text(alias)
            for alias in sample.get("answer_aliases", [])
            if clean_text(alias)
        ]
        gold_answers = [answer] + answer_aliases
        question_entities = match_entities(question, entity_vocab)
        supporting_entities = []
        for gold_answer in gold_answers:
            supporting_entities.extend(match_entities(gold_answer, entity_vocab))
        supporting_entities = sorted(set(supporting_entities))

        item = {
            "id": sample.get("id", idx),
            "question": question,
            "answer": answer,
            "answer_aliases": answer_aliases,
            "question_entities": question_entities,
            "supporting_entities": supporting_entities,
            "supporting_facts": [],
        }
        if book_id:
            item["book_id"] = book_id
        for key in ("document_id", "gutenberg_id", "title", "split", "metadata"):
            if key in sample:
                item[key] = sample[key]
        test_data.append(item)
    return test_data


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_stage1(
    graph: Stage1Graph,
    out_data_dir: Path,
    manifest: dict[str, Any],
    write_compat_files: bool,
    dry_run: bool,
    test_data: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    stage1_dir = out_data_dir / "processed" / "stage1"
    raw_dir = out_data_dir / "raw"
    counts = graph.counts()
    manifest = dict(manifest)
    manifest["counts"] = counts
    manifest["counts"]["test_samples"] = len(test_data or [])
    manifest["compat_files_written"] = write_compat_files
    if dry_run:
        print(f"[DRY-RUN] {out_data_dir} -> {counts}")
        return manifest

    stage1_dir.mkdir(parents=True, exist_ok=True)
    if write_compat_files:
        raw_dir.mkdir(parents=True, exist_ok=True)

    node_rows = [
        {
            "name": name,
            "type": node_type,
            "attributes": json_dumps(attrs),
        }
        for name, (node_type, attrs) in sorted(graph.nodes.items())
    ]
    relation_rows = [
        {
            "name": name,
            "attributes": json_dumps(attrs),
        }
        for name, attrs in sorted(graph.relations.items())
    ]
    sorted_edges = sorted(
        graph.edges.values(),
        key=lambda item: (item[0], item[1], item[2], json_dumps(item[3])),
    )
    edge_rows = [
        {
            "source": source,
            "relation": relation,
            "target": target,
            "attributes": json_dumps(attrs),
        }
        for source, relation, target, attrs in sorted_edges
    ]

    write_csv(stage1_dir / "nodes.csv", node_rows, ["name", "type", "attributes"])
    write_csv(stage1_dir / "relations.csv", relation_rows, ["name", "attributes"])
    write_csv(
        stage1_dir / "edges.csv",
        edge_rows,
        ["source", "relation", "target", "attributes"],
    )

    if write_compat_files:
        with (stage1_dir / "kg.txt").open("w", encoding="utf-8") as f:
            for source, relation, target in sorted(graph.kg_triples):
                f.write(COMPAT_KG_DELIMITER.join((source, relation, target)) + "\n")
        with (stage1_dir / "document2entities.json").open("w", encoding="utf-8") as f:
            json.dump(graph.document2entities, f, ensure_ascii=False, indent=2)
        with (raw_dir / "dataset_corpus.json").open("w", encoding="utf-8") as f:
            json.dump(graph.corpus, f, ensure_ascii=False, indent=2)
    if test_data is not None:
        with (stage1_dir / "test.json").open("w", encoding="utf-8") as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)

    with (stage1_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[OK] {out_data_dir} -> {counts}")
    return manifest


def export_flat_dataset(
    dataset_name: str,
    dataset_dir: Path,
    flow: FlowSpec,
    output_root: Path,
    model_label: str,
    model_dir_glob: str,
    qa_source_dir: Path,
    write_compat_files: bool,
    dry_run: bool,
) -> dict[str, Any]:
    openie_path = dataset_dir / flow.openie_file(model_label)
    model_dir = find_model_dir(dataset_dir, flow.graph_file, model_dir_glob)
    graph_path = model_dir / flow.graph_file if model_dir else dataset_dir / flow.graph_file
    data_name = f"{dataset_name}_{flow.name}"
    manifest = {
        "dataset": dataset_name,
        "flow": flow.name,
        "source_dataset_dir": str(dataset_dir),
        "source_openie_path": str(openie_path),
        "source_graph_path": str(graph_path),
        "source_qa_path": str(qa_source_dir / f"{dataset_name}.json"),
        "graph_pickle_stats": graph_pickle_stats(graph_path),
        "literaryqa_mode": None,
    }
    if not openie_path.exists():
        print(f"[SKIP] Missing OpenIE JSON: {openie_path}")
        manifest["skipped"] = "missing_openie_json"
        return manifest
    if not graph_path.exists():
        print(f"[SKIP] Missing graph pickle: {graph_path}")
        manifest["skipped"] = "missing_graph_pickle"
        return manifest

    graph = build_graph_from_openie(openie_path)
    test_data = build_qa_test_data(
        dataset_name,
        graph,
        load_qa_samples(qa_source_dir, dataset_name),
    )
    return write_stage1(
        graph,
        output_root / data_name,
        manifest,
        write_compat_files=write_compat_files,
        dry_run=dry_run,
        test_data=test_data,
    )


def iter_literaryqa_books(dataset_dir: Path, max_books: int | None) -> list[Path]:
    books_root = dataset_dir / "books"
    if not books_root.exists():
        return []
    books = sorted(path for path in books_root.iterdir() if path.is_dir())
    if max_books is not None:
        return books[:max_books]
    return books


def export_literaryqa_per_book(
    dataset_dir: Path,
    flow: FlowSpec,
    output_root: Path,
    model_label: str,
    model_dir_glob: str,
    max_books: int | None,
    source_dir: Path,
    write_compat_files: bool,
    dry_run: bool,
) -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    for book_dir in iter_literaryqa_books(dataset_dir, max_books):
        openie_path = book_dir / flow.openie_file(model_label)
        model_dir = find_model_dir(book_dir, flow.graph_file, model_dir_glob)
        graph_path = model_dir / flow.graph_file if model_dir else book_dir / flow.graph_file
        data_name = f"literaryqa_{flow.name}__{book_dir.name}"
        manifest = {
            "dataset": "literaryqa",
            "flow": flow.name,
            "book_id": book_dir.name,
            "source_dataset_dir": str(dataset_dir),
            "source_book_dir": str(book_dir),
            "source_openie_path": str(openie_path),
            "source_graph_path": str(graph_path),
            "graph_pickle_stats": graph_pickle_stats(graph_path),
            "literaryqa_mode": "per-book",
        }
        if not openie_path.exists():
            print(f"[SKIP] Missing LiteraryQA OpenIE JSON: {openie_path}")
            manifest["skipped"] = "missing_openie_json"
            manifests.append(manifest)
            continue
        if not graph_path.exists():
            print(f"[SKIP] Missing LiteraryQA graph pickle: {graph_path}")
            manifest["skipped"] = "missing_graph_pickle"
            manifests.append(manifest)
            continue

        graph = build_graph_from_openie(openie_path, book_id=book_dir.name)
        test_data = build_literaryqa_test_data(
            graph,
            load_literaryqa_samples(source_dir, book_dir.name),
            book_id=book_dir.name,
        )
        manifest = write_stage1(
            graph,
            output_root / data_name,
            manifest,
            write_compat_files=write_compat_files,
            dry_run=dry_run,
            test_data=test_data,
        )
        manifests.append(manifest)
    return manifests


def export_literaryqa_combined(
    dataset_dir: Path,
    flow: FlowSpec,
    output_root: Path,
    model_label: str,
    model_dir_glob: str,
    max_books: int | None,
    source_dir: Path,
    write_compat_files: bool,
    dry_run: bool,
) -> dict[str, Any]:
    combined = Stage1Graph()
    sources: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    combined_test_data: list[dict[str, Any]] = []
    for book_dir in iter_literaryqa_books(dataset_dir, max_books):
        openie_path = book_dir / flow.openie_file(model_label)
        model_dir = find_model_dir(book_dir, flow.graph_file, model_dir_glob)
        graph_path = model_dir / flow.graph_file if model_dir else book_dir / flow.graph_file
        source_info = {
            "book_id": book_dir.name,
            "source_openie_path": str(openie_path),
            "source_graph_path": str(graph_path),
            "graph_pickle_stats": graph_pickle_stats(graph_path),
        }
        if not openie_path.exists():
            skipped.append({"book_id": book_dir.name, "reason": "missing_openie_json"})
            continue
        if not graph_path.exists():
            skipped.append({"book_id": book_dir.name, "reason": "missing_graph_pickle"})
            continue
        book_graph = build_graph_from_openie(openie_path, book_id=book_dir.name)
        combined.merge(book_graph)
        combined_test_data.extend(
            build_literaryqa_test_data(
                book_graph,
                load_literaryqa_samples(source_dir, book_dir.name),
                book_id=book_dir.name,
            )
        )
        sources.append(source_info)

    data_name = f"literaryqa_{flow.name}"
    manifest = {
        "dataset": "literaryqa",
        "flow": flow.name,
        "source_dataset_dir": str(dataset_dir),
        "literaryqa_mode": "combined",
        "sources": sources,
        "skipped_sources": skipped,
    }
    manifest = write_stage1(
        combined,
        output_root / data_name,
        manifest,
        write_compat_files=write_compat_files,
        dry_run=dry_run,
        test_data=combined_test_data,
    )
    if skipped:
        print(f"[WARN] literaryqa {flow.name}: skipped {len(skipped)} books")
    return manifest


def main() -> None:
    args = parse_args()
    write_compat_files = not args.no_compat_files
    all_manifests: list[dict[str, Any]] = []

    for dataset_name in args.datasets:
        dataset_dir = args.hipporag_outputs / DATASETS[dataset_name]
        if not dataset_dir.exists():
            print(f"[SKIP] Missing dataset dir: {dataset_dir}")
            continue
        for flow_name in args.flows:
            flow = FLOWS[flow_name]
            if dataset_name == "literaryqa":
                if args.literaryqa_mode == "per-book":
                    all_manifests.extend(
                        export_literaryqa_per_book(
                            dataset_dir=dataset_dir,
                            flow=flow,
                            output_root=args.output_root,
                            model_label=args.model_label,
                            model_dir_glob=args.model_dir_glob,
                            max_books=args.max_literary_books,
                            source_dir=args.literaryqa_source_dir,
                            write_compat_files=write_compat_files,
                            dry_run=args.dry_run,
                        )
                    )
                else:
                    all_manifests.append(
                        export_literaryqa_combined(
                            dataset_dir=dataset_dir,
                            flow=flow,
                            output_root=args.output_root,
                            model_label=args.model_label,
                            model_dir_glob=args.model_dir_glob,
                            max_books=args.max_literary_books,
                            source_dir=args.literaryqa_source_dir,
                            write_compat_files=write_compat_files,
                            dry_run=args.dry_run,
                        )
                    )
            else:
                all_manifests.append(
                    export_flat_dataset(
                        dataset_name=dataset_name,
                        dataset_dir=dataset_dir,
                        flow=flow,
                        output_root=args.output_root,
                        model_label=args.model_label,
                        model_dir_glob=args.model_dir_glob,
                        qa_source_dir=args.qa_source_dir,
                        write_compat_files=write_compat_files,
                        dry_run=args.dry_run,
                    )
                )

    if not args.dry_run:
        args.output_root.mkdir(parents=True, exist_ok=True)
        summary_path = args.output_root / "export_summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(all_manifests, f, ensure_ascii=False, indent=2)
        print(f"[OK] summary -> {summary_path}")


if __name__ == "__main__":
    main()
