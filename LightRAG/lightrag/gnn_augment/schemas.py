from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


ENTITY_NODE = "entity"
CHUNK_NODE = "chunk"
FACT_EDGE_TYPE = 0
SYNONYM_EDGE_TYPE = 1
ENTITY_CHUNK_EDGE_TYPE = 2
NUM_SUBGRAPH_EDGE_TYPES = 3

ENTITY_NODE_TYPE = 0
CHUNK_NODE_TYPE = 1
OTHER_NODE_TYPE = 2
NUM_SUBGRAPH_NODE_TYPES = 3


def chunk_node_id(chunk_id: str) -> str:
    return f"chunk::{chunk_id}"


def raw_chunk_id(node_id: str) -> str:
    return node_id.removeprefix("chunk::")


@dataclass(slots=True)
class RelationEdge:
    src: str
    tgt: str
    description: str
    keywords: str
    source_ids: list[str]
    file_path: str = "unknown_source"
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, str]:
        return tuple(sorted((self.src, self.tgt)))

    @property
    def triple(self) -> tuple[str, str, str]:
        predicate = self.keywords.strip() or self.description.strip() or "related to"
        return (self.src, predicate, self.tgt)


@dataclass(slots=True)
class LightRAGGraphSnapshot:
    node_ids: list[str]
    node_types: dict[str, str]
    node_texts: dict[str, str]
    entity_data: dict[str, dict[str, Any]]
    chunks: dict[str, dict[str, Any]]
    relations: list[RelationEdge]
    entity_source_ids: dict[str, list[str]]

    @property
    def entity_ids(self) -> list[str]:
        return [node_id for node_id in self.node_ids if self.node_types[node_id] == ENTITY_NODE]

    @property
    def relation_keys(self) -> set[tuple[str, str]]:
        return {relation.key for relation in self.relations}


@dataclass(slots=True)
class SampledSubgraphRecord:
    subgraph_id: str
    root_node_idx: int
    root_node_name: str
    root_node_text: str
    node_indices: list[int]
    missing_score: float = 0.0
    entity_node_names: list[str] = field(default_factory=list)
    entity_texts: list[str] = field(default_factory=list)
    chunk_ids: list[str] = field(default_factory=list)
    known_triples: list[tuple[str, str, str]] = field(default_factory=list)
    unique_fact_edges: int = 0


@dataclass(slots=True)
class SubgraphCompletionBatch:
    subgraph: SampledSubgraphRecord
    chunk_ids: list[str]
    known_triples: list[tuple[str, str, str]]


@dataclass(slots=True)
class CompletedRelation:
    source_entity: str
    target_entity: str
    keywords: str
    description: str
    evidence_chunk_ids: list[str]
    confidence: float = 1.0
    missing_score: float = 0.0
    subgraph_root: str = ""
    raw_response: str = ""
    source_entity_type: str = "UNKNOWN"
    source_entity_description: str = ""
    target_entity_type: str = "UNKNOWN"
    target_entity_description: str = ""
    entity_source_chunk_ids: dict[str, list[str]] = field(default_factory=dict)
    extraction_schema: str = "lightrag_entity_relation"

    @property
    def key(self) -> tuple[str, str]:
        return tuple(sorted((self.source_entity, self.target_entity)))
