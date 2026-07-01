import json
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

NER_SYSTEM_PROMPT = """You are a Named Entity Recognition (NER) system for enterprise knowledge graphs.
Extract all named entities from the text and classify them into these types:
- Organization: Companies, institutions, agencies, departments
- Person: Individual people, roles
- Product: Products, services, offerings
- Location: Physical places, regions, addresses
- Event: Events, conferences, meetings
- Document: Documents, reports, files, articles
- Concept: Abstract concepts, methodologies, frameworks
- Date: Dates, time periods
- Technology: Technologies, tools, platforms, frameworks

For each entity, provide:
1. The canonical name
2. Entity type (from the list above)
3. A brief description
4. Aliases (alternative names, acronyms, abbreviations)

Return ONLY a JSON array of objects with fields: name, entity_type, description, aliases
Example: [{"name": "Google", "entity_type": "Organization", "description": "Technology company", "aliases": ["GOOG", "Alphabet"]}]

If no entities are found, return an empty array []."""


class EntityExtractor:
    """Extracts named entities from text using Gemini LLM."""

    def __init__(self, gemini_service=None):
        self.gemini_service = gemini_service

    async def extract(self, text: str, chunk_id: str = "", source_document: str = "") -> list[dict]:
        if not text or not text.strip():
            return []

        if self.gemini_service:
            try:
                return await self._extract_with_llm(text, chunk_id, source_document)
            except Exception as e:
                logger.warning(f"LLM entity extraction failed, using pattern fallback: {e}")

        return self._extract_with_patterns(text, chunk_id, source_document)

    async def _extract_with_llm(self, text: str, chunk_id: str, source_document: str) -> list[dict]:
        prompt = f"Extract entities from the following text:\n\n{text[:8000]}"
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
                    system_instruction=NER_SYSTEM_PROMPT,
                    temperature=0.1,
                    max_output_tokens=2048,
                ),
            )

            result_text = response.text.strip()
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0]

            entities = json.loads(result_text)
            if not isinstance(entities, list):
                return []

            for entity in entities:
                entity["chunk_ids"] = [chunk_id] if chunk_id else []
                entity["source_document"] = source_document
                entity["aliases"] = entity.get("aliases", [])

            return entities

        except Exception as e:
            logger.warning(f"LLM extraction parse error: {e}")
            return []

    def _extract_with_patterns(self, text: str, chunk_id: str, source_document: str) -> list[dict]:
        entities = []
        seen = set()

        patterns = {
            "Organization": [
                r'(?i)\b([A-Z][a-z]+(?:Inc|Corp|Ltd|LLC|GmbH|SA|PLC|Co|Group|Technologies|Systems|Solutions|Industries|Enterprises?|Global|International))',
                r'(?i)\b(Google|Microsoft|Amazon|Apple|Meta|OpenAI|Anthropic|Intel|IBM|Oracle|SAP|Salesforce|Adobe|Uber|Airbnb|Twitter|LinkedIn|Netflix|Spotify)\b',
            ],
            "Person": [
                r'(?i)\b(?:CEO|CTO|CFO|President|Director|Manager|Chief|Founder)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)\b',
                r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b(?:,\s+(?:CEO|CTO|CFO|President|Director))',
            ],
            "Location": [
                r'(?i)\b(New\s+York|San\s+Francisco|London|Tokyo|Berlin|Paris|Sydney|Toronto|Seattle|Austin|Boston|Chicago|Los\s+Angeles|Washington|Singapore|Hong\s+Kong|Dubai|Shanghai)\b',
            ],
            "Technology": [
                r'(?i)\b(Python|JavaScript|TypeScript|React|Node\.js|Docker|Kubernetes|AWS|Azure|GCP|TensorFlow|PyTorch|LangChain|GraphQL|REST|API|PostgreSQL|MongoDB|Redis|Kafka)\b',
            ],
            "Date": [
                r'\b(20\d{2})\b',
                r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+20\d{2}\b',
            ],
        }

        for entity_type, regex_list in patterns.items():
            for pattern in regex_list:
                for match in re.finditer(pattern, text):
                    name = match.group(1) if match.lastindex and match.groups() else match.group(0)
                    name = name.strip()
                    if name.lower() not in seen and len(name) > 1:
                        seen.add(name.lower())
                        entities.append({
                            "name": name,
                            "entity_type": entity_type,
                            "description": f"{entity_type} mentioned in text",
                            "aliases": [],
                            "chunk_ids": [chunk_id] if chunk_id else [],
                            "source_document": source_document,
                        })

        return entities
