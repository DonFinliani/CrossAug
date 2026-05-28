from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from lightrag.utils import logger, remove_think_tags

from .config import SubgraphCompletionConfig
from .schemas import CompletedRelation, LightRAGGraphSnapshot, SubgraphCompletionBatch
from .subgraph_sampler import text_processing


def _fix_broken_generated_json(json_str: str) -> str:
    def find_unclosed(text: str) -> list[str]:
        unclosed = []
        inside_string = False
        escape_next = False

        for char in text:
            if inside_string:
                if escape_next:
                    escape_next = False
                elif char == "\\":
                    escape_next = True
                elif char == '"':
                    inside_string = False
            else:
                if char == '"':
                    inside_string = True
                elif char in "{[":
                    unclosed.append(char)
                elif char in "}]":
                    if unclosed and (
                        (char == "}" and unclosed[-1] == "{")
                        or (char == "]" and unclosed[-1] == "[")
                    ):
                        unclosed.pop()
        return unclosed

    try:
        json.loads(json_str)
        return json_str
    except json.JSONDecodeError:
        pass

    last_comma_index = json_str.rfind(",")
    if last_comma_index != -1:
        json_str = json_str[:last_comma_index]

    closing_map = {"{": "}", "[": "]"}
    for open_char in reversed(find_unclosed(json_str)):
        json_str += closing_map[open_char]
    return json_str


def _filter_invalid_triples(triples: list[Any]) -> list[list[str]]:
    unique_triples = set()
    valid = []
    for triple in triples:
        if not isinstance(triple, (list, tuple)) or len(triple) != 3:
            continue
        valid_triple = [str(item) for item in triple]
        triple_key = tuple(valid_triple)
        if triple_key in unique_triples:
            continue
        unique_triples.add(triple_key)
        valid.append(valid_triple)
    return valid


def _clean_model_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


class SubgraphCompletionLLM:
    def __init__(self, config: SubgraphCompletionConfig) -> None:
        self.config = config

    @staticmethod
    def _generate_chunk_id_aliases(chunk_id_text: str) -> list[str]:
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
        return list(dict.fromkeys(aliases))

    @classmethod
    def _build_chunk_id_alias_map(cls, chunk_ids: list[str]) -> dict[str, str]:
        alias_map = {}
        for chunk_id in chunk_ids:
            if not isinstance(chunk_id, str):
                continue
            for alias in cls._generate_chunk_id_aliases(chunk_id):
                alias_map.setdefault(alias, chunk_id)
        return alias_map

    @classmethod
    def _resolve_model_chunk_ids(
        cls,
        raw_chunk_ids: list[Any],
        allowed_chunk_ids: set[str],
        chunk_id_alias_map: dict[str, str],
    ) -> list[str]:
        resolved = []
        seen = set()
        if isinstance(raw_chunk_ids, str):
            raw_chunk_ids = [raw_chunk_ids]
        elif not isinstance(raw_chunk_ids, list):
            raw_chunk_ids = []

        for raw_chunk_id in raw_chunk_ids:
            if not isinstance(raw_chunk_id, str):
                continue
            canonical = raw_chunk_id if raw_chunk_id in allowed_chunk_ids else None
            if canonical is None:
                for alias in cls._generate_chunk_id_aliases(raw_chunk_id):
                    matched = chunk_id_alias_map.get(alias)
                    if matched is not None:
                        canonical = matched
                        break
            if canonical is None or canonical in seen:
                continue
            resolved.append(canonical)
            seen.add(canonical)
        return resolved

    def _build_messages(
        self,
        snapshot: LightRAGGraphSnapshot,
        batch: SubgraphCompletionBatch,
    ) -> tuple[str, list[dict[str, str]], str]:
        passages = []
        for chunk_id in batch.chunk_ids:
            chunk = snapshot.chunks.get(chunk_id, {})
            passages.append(
                {
                    "chunk_id": chunk_id,
                    "passage": str(chunk.get("content", ""))[: self.config.max_chars_per_chunk],
                }
            )

        known_triples = [list(triple) for triple in batch.known_triples]
        entity_list = batch.subgraph.entity_texts[: self.config.subgraph_max_entities_per_prompt]
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
                "passage": "Lina Chen founded Acme Labs in 2021. Acme Labs developed the Aurora diagnostics platform.",
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
            "entities": [
                {
                    "entity_name": "Acme Labs",
                    "entity_type": "organization",
                    "entity_description": "Acme Labs is a company founded by Lina Chen in 2021 and later acquired by Northwind Health.",
                    "chunk_ids": ["chunk-acme-1", "chunk-acme-2"],
                },
                {
                    "entity_name": "Northwind Health",
                    "entity_type": "organization",
                    "entity_description": "Northwind Health acquired Acme Labs in March 2023 and integrated Aurora into its clinical workflow suite.",
                    "chunk_ids": ["chunk-acme-2"],
                },
                {
                    "entity_name": "Aurora diagnostics platform",
                    "entity_type": "product",
                    "entity_description": "The Aurora diagnostics platform was developed by Acme Labs and later integrated into Northwind Health's clinical workflow suite.",
                    "chunk_ids": ["chunk-acme-1", "chunk-acme-2", "chunk-acme-3"],
                },
                {
                    "entity_name": "March 2023",
                    "entity_type": "time",
                    "entity_description": "March 2023 is when Northwind Health acquired Acme Labs.",
                    "chunk_ids": ["chunk-acme-2"],
                },
                {
                    "entity_name": "2021",
                    "entity_type": "time",
                    "entity_description": "2021 is the year when Lina Chen founded Acme Labs.",
                    "chunk_ids": ["chunk-acme-1"],
                },
                {
                    "entity_name": "Northwind Health clinical workflow suite",
                    "entity_type": "product",
                    "entity_description": "Northwind Health's clinical workflow suite is the system into which Aurora was integrated before hospitals could deploy it.",
                    "chunk_ids": ["chunk-acme-2", "chunk-acme-3"],
                },
            ],
            "relations": [
                {
                    "source_entity": "Acme Labs",
                    "target_entity": "2021",
                    "relationship_keywords": "founding time, company history",
                    "relationship_description": "Acme Labs was founded by Lina Chen in 2021.",
                    "chunk_ids": ["chunk-acme-1"],
                },
                {
                    "source_entity": "Northwind Health",
                    "target_entity": "Acme Labs",
                    "relationship_keywords": "acquisition, corporate ownership",
                    "relationship_description": "Northwind Health acquired Acme Labs in March 2023.",
                    "chunk_ids": ["chunk-acme-2"],
                },
                {
                    "source_entity": "Northwind Health",
                    "target_entity": "March 2023",
                    "relationship_keywords": "acquisition time, chronology",
                    "relationship_description": "Northwind Health's acquisition of Acme Labs occurred in March 2023.",
                    "chunk_ids": ["chunk-acme-2"],
                },
                {
                    "source_entity": "Northwind Health",
                    "target_entity": "Aurora diagnostics platform",
                    "relationship_keywords": "platform integration, clinical workflow",
                    "relationship_description": "Northwind Health integrated the Aurora diagnostics platform into its clinical workflow suite after acquiring Acme Labs.",
                    "chunk_ids": ["chunk-acme-1", "chunk-acme-2"],
                },
                {
                    "source_entity": "Aurora diagnostics platform",
                    "target_entity": "Northwind Health clinical workflow suite",
                    "relationship_keywords": "deployment prerequisite, integration",
                    "relationship_description": "Hospitals could deploy Aurora only after it was integrated into Northwind Health's clinical workflow suite.",
                    "chunk_ids": ["chunk-acme-2", "chunk-acme-3"],
                },
            ],
        }
        system_prompt = (
            "Your task is to complete a local LightRAG knowledge graph using the provided evidence passages. "
            "You are given a sampled subgraph that a GNN marked as likely incomplete. The incompleteness may come "
            "from missing relation edges, missing entity nodes, or both. You are also given the entities already "
            "present in that subgraph and known triples already extracted from the same local evidence. "
            "Before deciding what to add, silently summarize what the current subgraph already says, identify the "
            "most important missing entities and relations, and then recover the highest-value missing facts. "
            "Use the same semantic schema as the base LightRAG extraction pipeline: entity records must contain "
            "entity_name, entity_type, and entity_description; relationship records must contain source_entity, "
            "target_entity, relationship_keywords, and relationship_description. "
            "Do not stop at shallow supplementation. Use the passages and the existing triples together to infer "
            "important but omitted facts that are directly stated or unambiguously supported, including temporal, "
            "conditional, prerequisite, role, ownership, part-whole, sequence, and other structurally important "
            "relations when the evidence clearly supports them. Prefer completions that require connecting multiple "
            "known triples or resolving information across multiple passages when that produces a more important and "
            "better-grounded fact. You should also propose high-value summary triples when they compactly capture the "
            "main role, status, outcome, function, or relationship pattern expressed by the current subgraph and its "
            "evidence passages. These summary triples must still be grounded in the provided evidence and should be "
            "useful for downstream retrieval, not vague restatements. "
            "Return only new, durable, retrieval-useful relationships and the entity records needed to support them. "
            "Do not repeat any known_triples and do not merely rephrase them. Avoid weak co-occurrence, temporary "
            "scene interactions, speculative inferences, or vague relationship_keywords like 'related to'. "
            "relationship_keywords should be concise high-level keywords separated by commas, not a full sentence. "
            "relationship_description should be a clear natural-language explanation of why the two entities are connected. "
            "Each returned entity and relationship must include at least one chunk_id copied exactly from the provided passages. "
            "If a relationship depends on evidence spread across multiple passages, include all relevant chunk_ids. "
            "Prefer existing entity strings when possible, but you may introduce a new entity only if it is explicitly "
            "named in the evidence passages and the relationship meaningfully fills a gap in the subgraph. "
            "Return JSON only with the keys 'entities' and 'relations'."
        )
        one_shot_user_prompt = (
            "Subgraph incompleteness hint:\n"
            f"{json.dumps({'gnn_missing_score': 0.91, 'possible_missingness_types': ['missing_relations', 'missing_entities']}, ensure_ascii=False, indent=2)}\n\n"
            f"Subgraph root entity:\n{json.dumps(one_shot_root_entity, ensure_ascii=False)}\n\n"
            f"Existing entities in the sampled subgraph:\n{json.dumps(one_shot_entities, ensure_ascii=False, indent=2)}\n\n"
            f"Known triples in this sampled subgraph:\n{json.dumps(one_shot_known_triples, ensure_ascii=False, indent=2)}\n\n"
            f"Evidence passages:\n{json.dumps(one_shot_passages, ensure_ascii=False, indent=2)}\n\n"
            "Required output schema:\n"
            "{\n"
            '  "entities": [\n'
            '    {"entity_name": "entity", "entity_type": "type", "entity_description": "description", "chunk_ids": ["chunk-id"]}\n'
            "  ],\n"
            '  "relations": [\n'
            '    {"source_entity": "source", "target_entity": "target", "relationship_keywords": "keyword, keyword", "relationship_description": "description", "chunk_ids": ["chunk-id"]}\n'
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
            "4. For every relationship endpoint, include an entity record with LightRAG-style entity_type and "
            "entity_description. Use existing entity strings when possible.\n"
            "5. For every relationship, write LightRAG-style relationship_keywords and relationship_description, "
            "not a bare predicate.\n"
            "6. When a new relationship depends on multiple passages, attach all relevant chunk_ids instead of citing "
            "only one chunk.\n"
            "7. Keep only facts that are directly stated or unambiguously supported by the evidence passages.\n\n"
            "Required output schema:\n"
            "{\n"
            '  "entities": [\n'
            '    {"entity_name": "entity", "entity_type": "type", "entity_description": "description", "chunk_ids": ["chunk-id"]}\n'
            "  ],\n"
            '  "relations": [\n'
            '    {"source_entity": "source", "target_entity": "target", "relationship_keywords": "keyword, keyword", "relationship_description": "description", "chunk_ids": ["chunk-id"]}\n'
            "  ]\n"
            "}\n\n"
            "Return JSON only."
        )
        history_messages = [
            {"role": "user", "content": one_shot_user_prompt},
            {"role": "assistant", "content": json.dumps(one_shot_response, ensure_ascii=False, indent=2)},
        ]
        return system_prompt, history_messages, user_prompt

    def _parse_response(self, raw_response: str) -> list[dict[str, Any]]:
        raw_response = remove_think_tags(raw_response or "")
        json_match = re.search(r"\{.*\}", raw_response, re.DOTALL)
        if json_match is None:
            return []
        json_str = _fix_broken_generated_json(json_match.group(0))
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError:
            return []
        parsed_entities = parsed.get("entities", [])
        if not isinstance(parsed_entities, list):
            parsed_entities = []
        entity_lookup: dict[str, dict[str, Any]] = {}
        for entity in parsed_entities:
            if not isinstance(entity, dict):
                continue
            entity_name = _clean_model_text(entity.get("entity_name"))
            if not entity_name:
                continue
            chunk_ids = entity.get("chunk_ids", [])
            if isinstance(chunk_ids, str):
                chunk_ids = [chunk_ids]
            elif not isinstance(chunk_ids, list):
                chunk_ids = []
            entity_lookup[text_processing(entity_name)] = {
                "entity_name": entity_name,
                "entity_type": _clean_model_text(entity.get("entity_type")) or "UNKNOWN",
                "entity_description": _clean_model_text(
                    entity.get("entity_description")
                    or entity.get("description")
                ),
                "chunk_ids": [
                    chunk_id
                    for chunk_id in chunk_ids
                    if isinstance(chunk_id, str) and chunk_id.strip() != ""
                ],
            }

        normalized_items = []
        relations = parsed.get("relations", [])
        if not isinstance(relations, list):
            relations = []
        for item in relations:
            if isinstance(item, dict):
                source_entity = _clean_model_text(item.get("source_entity"))
                target_entity = _clean_model_text(item.get("target_entity"))
                relationship_keywords = _clean_model_text(
                    item.get("relationship_keywords") or item.get("keywords")
                )
                relationship_description = _clean_model_text(
                    item.get("relationship_description")
                    or item.get("description")
                )
                chunk_ids = item.get("chunk_ids", [])
            else:
                source_entity = ""
                target_entity = ""
                relationship_keywords = ""
                relationship_description = ""
                chunk_ids = []
            if not source_entity or not target_entity:
                continue
            if not relationship_keywords and not relationship_description:
                continue
            if isinstance(chunk_ids, str):
                chunk_ids = [chunk_ids]
            elif not isinstance(chunk_ids, list):
                chunk_ids = []
            normalized_items.append(
                {
                    "source_entity": source_entity,
                    "target_entity": target_entity,
                    "relationship_keywords": relationship_keywords
                    or relationship_description,
                    "relationship_description": relationship_description
                    or f"{source_entity} {relationship_keywords} {target_entity}",
                    "chunk_ids": [
                        chunk_id
                        for chunk_id in chunk_ids
                        if isinstance(chunk_id, str) and chunk_id.strip() != ""
                    ],
                    "entities": entity_lookup,
                }
            )

        # Backward compatibility for older cached/model responses.
        triples = parsed.get("new_triples", [])
        if isinstance(triples, list):
            for item in triples:
                if isinstance(item, dict):
                    triple_value = item.get("triple", [])
                    chunk_ids = item.get("chunk_ids", [])
                else:
                    triple_value = item
                    chunk_ids = []
                filtered = _filter_invalid_triples([triple_value])
                if not filtered:
                    continue
                source_entity, relationship_keywords, target_entity = [
                    _clean_model_text(value) for value in filtered[0]
                ]
                if isinstance(chunk_ids, str):
                    chunk_ids = [chunk_ids]
                elif not isinstance(chunk_ids, list):
                    chunk_ids = []
                normalized_items.append(
                    {
                        "source_entity": source_entity,
                        "target_entity": target_entity,
                        "relationship_keywords": relationship_keywords,
                        "relationship_description": (
                            f"{source_entity} {relationship_keywords} {target_entity}"
                        ),
                        "chunk_ids": [
                            chunk_id
                            for chunk_id in chunk_ids
                            if isinstance(chunk_id, str) and chunk_id.strip() != ""
                        ],
                        "entities": entity_lookup,
                    }
                )
        return normalized_items

    def _canonical_entity_lookup(self, snapshot: LightRAGGraphSnapshot) -> dict[str, str]:
        lookup = {}
        for entity in snapshot.entity_ids:
            lookup.setdefault(text_processing(entity), entity)
        return lookup

    def _collect_valid_relations(
        self,
        parsed_results: list[dict[str, Any]],
        raw_response: str,
        snapshot: LightRAGGraphSnapshot,
        batch: SubgraphCompletionBatch,
    ) -> list[CompletedRelation]:
        allowed_chunk_ids = set(batch.chunk_ids)
        chunk_id_alias_map = self._build_chunk_id_alias_map(batch.chunk_ids)
        known_triples = {tuple(text_processing(list(triple))) for triple in batch.known_triples}
        canonical_lookup = self._canonical_entity_lookup(snapshot)
        valid = []
        for item in parsed_results:
            raw_source = _clean_model_text(item.get("source_entity"))
            raw_target = _clean_model_text(item.get("target_entity"))
            raw_keywords = _clean_model_text(item.get("relationship_keywords"))
            raw_description = _clean_model_text(item.get("relationship_description"))
            proc_triple = tuple(text_processing([raw_source, raw_keywords, raw_target]))
            if (
                len(proc_triple) != 3
                or proc_triple[0] == ""
                or proc_triple[1] == ""
                or proc_triple[2] == ""
                or proc_triple[0] == proc_triple[2]
            ):
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

            src = canonical_lookup.get(proc_triple[0], raw_source)
            tgt = canonical_lookup.get(proc_triple[2], raw_target)
            entity_lookup = item.get("entities", {})
            if not isinstance(entity_lookup, dict):
                entity_lookup = {}
            source_entity_data = entity_lookup.get(proc_triple[0], {})
            target_entity_data = entity_lookup.get(proc_triple[2], {})
            source_entity_chunk_ids = self._resolve_model_chunk_ids(
                raw_chunk_ids=source_entity_data.get("chunk_ids", []),
                allowed_chunk_ids=allowed_chunk_ids,
                chunk_id_alias_map=chunk_id_alias_map,
            ) or cited_chunk_ids
            target_entity_chunk_ids = self._resolve_model_chunk_ids(
                raw_chunk_ids=target_entity_data.get("chunk_ids", []),
                allowed_chunk_ids=allowed_chunk_ids,
                chunk_id_alias_map=chunk_id_alias_map,
            ) or cited_chunk_ids
            valid.append(
                CompletedRelation(
                    source_entity=src,
                    target_entity=tgt,
                    keywords=raw_keywords,
                    description=raw_description
                    or f"{raw_source} {raw_keywords} {raw_target}",
                    evidence_chunk_ids=cited_chunk_ids,
                    confidence=1.0,
                    missing_score=batch.subgraph.missing_score,
                    subgraph_root=batch.subgraph.root_node_name,
                    raw_response=raw_response,
                    source_entity_type=_clean_model_text(
                        source_entity_data.get("entity_type")
                    )
                    or "UNKNOWN",
                    source_entity_description=_clean_model_text(
                        source_entity_data.get("entity_description")
                    ),
                    target_entity_type=_clean_model_text(
                        target_entity_data.get("entity_type")
                    )
                    or "UNKNOWN",
                    target_entity_description=_clean_model_text(
                        target_entity_data.get("entity_description")
                    ),
                    entity_source_chunk_ids={
                        src: source_entity_chunk_ids,
                        tgt: target_entity_chunk_ids,
                    },
                )
            )
        return valid

    async def complete_one(
        self,
        rag: Any,
        snapshot: LightRAGGraphSnapshot,
        batch: SubgraphCompletionBatch,
    ) -> tuple[list[CompletedRelation], dict[str, Any]]:
        system_prompt, history_messages, user_prompt = self._build_messages(snapshot, batch)
        raw_response = await rag.llm_model_func(
            user_prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            max_tokens=self.config.llm_max_tokens,
            stream=False,
        )
        parsed_results = self._parse_response(raw_response)
        valid_relations = self._collect_valid_relations(
            parsed_results,
            raw_response,
            snapshot,
            batch,
        )
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
            "messages_used_for_llm": [
                {"role": "system", "content": system_prompt},
                *history_messages,
                {"role": "user", "content": user_prompt},
            ],
            "raw_response": raw_response,
            "parsed_results": parsed_results,
            "num_valid_triples": len(valid_relations),
        }
        return valid_relations, batch_log

    async def complete_many(
        self,
        rag: Any,
        snapshot: LightRAGGraphSnapshot,
        batches: list[SubgraphCompletionBatch],
    ) -> tuple[list[CompletedRelation], list[dict[str, Any]]]:
        if not batches:
            return [], []
        semaphore = asyncio.Semaphore(max(1, self.config.llm_concurrency))

        async def _complete(batch_idx: int, batch: SubgraphCompletionBatch):
            async with semaphore:
                try:
                    relations, log = await self.complete_one(rag, snapshot, batch)
                    return batch_idx, relations, log
                except Exception as exc:
                    logger.exception("Subgraph completion failed for batch %d", batch_idx)
                    return batch_idx, [], {
                        "subgraph_id": batch.subgraph.subgraph_id,
                        "root_node_text": batch.subgraph.root_node_text,
                        "missing_score": batch.subgraph.missing_score,
                        "chunk_ids": batch.chunk_ids,
                        "num_valid_triples": 0,
                        "failed": True,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }

        results = await asyncio.gather(
            *[_complete(batch_idx, batch) for batch_idx, batch in enumerate(batches)]
        )
        results.sort(key=lambda item: item[0])
        unique: dict[tuple[str, str, str], CompletedRelation] = {}
        batch_logs = []
        for _idx, relations, log in results:
            batch_logs.append(log)
            for relation in relations:
                triple_key = (
                    text_processing(relation.source_entity),
                    text_processing(relation.keywords),
                    text_processing(relation.target_entity),
                )
                if triple_key not in unique:
                    unique[triple_key] = relation
                else:
                    merged = sorted(
                        set(unique[triple_key].evidence_chunk_ids)
                        | set(relation.evidence_chunk_ids)
                    )
                    unique[triple_key].evidence_chunk_ids = merged
                    for entity_name, chunk_ids in relation.entity_source_chunk_ids.items():
                        current_chunks = set(
                            unique[triple_key].entity_source_chunk_ids.get(
                                entity_name, []
                            )
                        )
                        unique[triple_key].entity_source_chunk_ids[entity_name] = sorted(
                            current_chunks | set(chunk_ids)
                        )
                    if (
                        not unique[triple_key].source_entity_description
                        and relation.source_entity_description
                    ):
                        unique[triple_key].source_entity_description = (
                            relation.source_entity_description
                        )
                    if (
                        not unique[triple_key].target_entity_description
                        and relation.target_entity_description
                    ):
                        unique[triple_key].target_entity_description = (
                            relation.target_entity_description
                        )
                    unique[triple_key].missing_score = max(
                        unique[triple_key].missing_score,
                        relation.missing_score,
                    )
        return list(unique.values()), batch_logs
