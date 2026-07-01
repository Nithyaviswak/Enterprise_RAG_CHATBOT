import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

RELATION_EXTRACTION_PROMPT = """You are a Relationship Extraction system for enterprise knowledge graphs.
Given a text and a list of entities found in it, identify all meaningful relationships between entities.

Entity types: Organization, Person, Product, Location, Event, Document, Concept, Date, Technology

Common relationship types:
- works_for / employs / partner_of (Organization <-> Person)
- owns / acquired_by / subsidiary_of (Organization <-> Organization)
- located_in / headquartered_in (Organization/Location <-> Location)
- developed_by / created_by (Product/Technology <-> Organization/Person)
- uses / integrates / depends_on (Organization/Product <-> Technology/Product)
- participated_in / held_at (Event <-> Organization/Person/Location)
- authored_by / mentioned_in (Document <-> Person/Organization)
- part_of / related_to (general hierarchical or associative relationships)
- collaborated_with / acquired (Organization <-> Organization)
- invested_in / funds (Organization <-> Organization/Product)

Return ONLY a JSON array of objects with fields:
- source_name: exact name from the entities list
- target_name: exact name from the entities list  
- relation_type: one of the types above
- description: brief description of the relationship context
- weight: confidence score 0.0-1.0

Example: [{"source_name": "Google", "target_name": "DeepMind", "relation_type": "acquired_by", "description": "Google acquired DeepMind in 2014", "weight": 0.95}]

If no relationships are found, return an empty array []."""


class RelationshipExtractor:
    """Extracts relationships between entities using Gemini LLM."""

    def __init__(self, gemini_service=None):
        self.gemini_service = gemini_service

    async def extract(
        self,
        text: str,
        entities: list[dict],
        source_document: str = "",
    ) -> list[dict]:
        if not text or not entities:
            return []

        if self.gemini_service:
            try:
                return await self._extract_with_llm(text, entities, source_document)
            except Exception as e:
                logger.warning(f"LLM relationship extraction failed: {e}")

        return self._extract_cooccurrence(text, entities, source_document)

    async def _extract_with_llm(
        self,
        text: str,
        entities: list[dict],
        source_document: str,
    ) -> list[dict]:
        entity_names = [e["name"] for e in entities]
        entity_list_str = "\n".join(f"- {e['name']} ({e.get('entity_type', 'Unknown')})" for e in entities)

        prompt = f"""Text:
{text[:8000]}

Entities found in text:
{entity_list_str}

Extract relationships between these entities."""

        try:
            from google import genai
            from google.genai import types
            from app.config import get_settings

            settings = get_settings()
            client = genai.Client(api_key=settings.gemini_api_key)

            response = client.models.generate_content(
                model=settings.gemini_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=RELATION_EXTRACTION_PROMPT,
                    temperature=0.1,
                    max_output_tokens=2048,
                ),
            )

            result_text = response.text.strip()
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0]

            relationships = json.loads(result_text)
            if not isinstance(relationships, list):
                return []

            entity_names_set = {e["name"].lower() for e in entities}
            validated = []
            for rel in relationships:
                if (
                    rel.get("source_name", "").lower() in entity_names_set
                    and rel.get("target_name", "").lower() in entity_names_set
                ):
                    rel["source_document"] = source_document
                    validated.append(rel)

            return validated

        except Exception as e:
            logger.warning(f"LLM relationship extraction parse error: {e}")
            return []

    def _extract_cooccurrence(
        self,
        text: str,
        entities: list[dict],
        source_document: str,
    ) -> list[dict]:
        relationships = []
        seen_pairs = set()
        text_lower = text.lower()

        entity_positions = []
        for entity in entities:
            name = entity["name"]
            pos = text_lower.find(name.lower())
            if pos >= 0:
                entity_positions.append((pos, name, entity))

        entity_positions.sort()

        for i in range(len(entity_positions)):
            for j in range(i + 1, len(entity_positions)):
                pos1, name1, ent1 = entity_positions[i]
                pos2, name2, ent2 = entity_positions[j]

                distance = pos2 - (pos1 + len(name1))
                if 0 < distance < 500:
                    pair_key = tuple(sorted([name1.lower(), name2.lower()]))
                    if pair_key not in seen_pairs:
                        seen_pairs.add(pair_key)
                        window_text = text[pos1:pos2 + len(name2)]

                        relation_type = self._infer_relation_type(ent1, ent2, window_text)

                        relationships.append({
                            "source_name": name1,
                            "target_name": name2,
                            "relation_type": relation_type,
                            "description": f"Co-occurring entities within proximity",
                            "weight": max(0.1, 1.0 - (distance / 500)),
                            "source_document": source_document,
                        })

        return relationships

    def _infer_relation_type(self, ent1: dict, ent2: dict, window_text: str) -> str:
        t1 = ent1.get("entity_type", "")
        t2 = ent2.get("entity_type", "")
        text_lower = window_text.lower()

        type_pairs = {
            ("Organization", "Person"): "works_for",
            ("Person", "Organization"): "works_for",
            ("Organization", "Location"): "located_in",
            ("Location", "Organization"): "located_in",
            ("Organization", "Organization"): "partner_of",
            ("Organization", "Product"): "develops",
            ("Product", "Organization"): "developed_by",
            ("Organization", "Technology"): "uses",
            ("Technology", "Organization"): "used_by",
            ("Person", "Product"): "created_by",
            ("Product", "Person"): "creator",
            ("Document", "Person"): "authored_by",
            ("Person", "Document"): "author_of",
            ("Event", "Location"): "held_at",
            ("Location", "Event"): "hosted",
        }

        pair_key = (t1, t2)
        if pair_key in type_pairs:
            return type_pairs[pair_key]

        if "acquired" in text_lower or "acquisition" in text_lower:
            return "acquired"
        if "partner" in text_lower or "collaborat" in text_lower:
            return "partner_of"
        if "invest" in text_lower:
            return "invested_in"
        if "subsidiary" in text_lower or "owned by" in text_lower:
            return "subsidiary_of"

        return "related_to"
