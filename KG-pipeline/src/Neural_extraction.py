#!/usr/bin/env python3
"""
Neural Extraction Module for Knowledge Graph Generation

This module performs AI-driven extraction of entities and relations from parsed documents,
generating three sub-KGs (product_technical, operation_modes, troubleshooting).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
import unicodedata
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import jsonschema
except ImportError:
    jsonschema = None


# === Configuration Loading ===

def load_config(config_path: Path) -> dict:
    """Load YAML configuration file."""
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_schema(schema_path: Path) -> dict:
    """Load JSON schema for validation."""
    with schema_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# === Utility Functions ===

def slugify(text: str) -> str:
    """
    Convert text to a URL-safe slug.

    Examples:
        "Descaling Mode" -> "descaling_mode"
        "NTC Temperature Sensor" -> "ntc_temperature_sensor"
    """
    # Normalize unicode characters
    text = unicodedata.normalize('NFKD', text)
    text = text.encode('ascii', 'ignore').decode('ascii')

    # Convert to lowercase and replace spaces/special chars with underscore
    text = re.sub(r'[^\w\s-]', '', text.lower())
    text = re.sub(r'[-\s]+', '_', text)

    return text.strip('_')


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison (lowercase, strip, normalize spaces).
    """
    return re.sub(r'\s+', ' ', text.strip().lower())


def safe_json_parse(response: str, logger: Optional[logging.Logger] = None) -> Optional[dict]:
    """
    Extract and parse JSON from AI response with repair attempts.

    Handles:
    - Markdown code blocks
    - Typographic quotes
    - Trailing commas
    - Missing commas
    - Extra text around JSON
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Try to find JSON in markdown code block
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # Try to find JSON directly
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            logger.warning("No JSON structure found in response")
            return None

    # Repair common JSON issues
    # 1. Replace typographic quotes with standard quotes
    json_str = json_str.replace('"', '"').replace('"', '"')
    json_str = json_str.replace(''', "'").replace(''', "'")

    # 2. Remove trailing commas before closing brackets
    json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)

    # 3. Try to parse
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as exc:
        logger.warning(f"Initial JSON parse failed: {exc}")

        # Attempt 2: Try to fix missing commas between objects in arrays
        try:
            # Add comma between }{ patterns (with potential newlines)
            json_str_fixed = re.sub(r'\}[\s\n]*\{', '},{', json_str)
            return json.loads(json_str_fixed)
        except json.JSONDecodeError:
            pass

        # Attempt 3: Fix missing commas after closing braces in arrays
        try:
            # Pattern: } followed by newline and then { without comma
            json_str_fixed = re.sub(r'\}(\s*\n\s*)(\{)', r'},\1\2', json_str)
            return json.loads(json_str_fixed)
        except json.JSONDecodeError:
            pass

        # Attempt 4: Fix missing commas between property values
        try:
            # Add comma between " and { if missing
            json_str_fixed = re.sub(r'"\s*\n\s*"', '",\n"', json_str)
            return json.loads(json_str_fixed)
        except json.JSONDecodeError:
            pass

        # Attempt 5: Try to extract and fix entities array first
        try:
            entities_match = re.search(r'"entities"\s*:\s*\[(.*?)\]', json_str, re.DOTALL)
            if entities_match:
                entities_str = '[' + entities_match.group(1) + ']'
                # Fix missing commas in entities array
                entities_str = re.sub(r'\}[\s\n]*\{', '},{', entities_str)
                entities = json.loads(entities_str)

                # Try to extract relations
                relations_match = re.search(r'"relations"\s*:\s*\[(.*?)\]', json_str, re.DOTALL)
                relations = []
                if relations_match:
                    relations_str = '[' + relations_match.group(1) + ']'
                    relations_str = re.sub(r'\}[\s\n]*\{', '},{', relations_str)
                    try:
                        relations = json.loads(relations_str)
                    except json.JSONDecodeError:
                        logger.warning("Could not parse relations, using entities only")

                return {"entities": entities, "relations": relations}
        except (json.JSONDecodeError, AttributeError) as e:
            logger.warning(f"Attempt 5 failed: {e}")

        # Attempt 6: Parse objects one by one
        try:
            # Extract individual entity objects
            entity_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\}'
            entity_matches = re.findall(entity_pattern, json_str)

            entities = []
            for entity_str in entity_matches:
                try:
                    # Try to repair and parse individual entity
                    entity_str = re.sub(r',(\s*[}\]])', r'\1', entity_str)
                    entity = json.loads(entity_str)
                    if 'id' in entity and 'type' in entity:  # Validate it's an entity
                        entities.append(entity)
                except json.JSONDecodeError:
                    continue

            if entities:
                logger.info(f"Recovered {len(entities)} entities by parsing one-by-one")
                return {"entities": entities, "relations": []}
        except Exception as e:
            logger.warning(f"Attempt 6 failed: {e}")

        logger.error(f"Failed to parse JSON after all repair attempts. First 200 chars: {json_str[:200]}")
        return None


# === Text Processing ===

def extract_text_with_provenance(parsed_data: dict) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Extract text from parsed document with detailed provenance tracking.

    Returns:
        (concatenated_text, provenance_map)

        provenance_map: List of dicts with:
        - text: text chunk
        - section_id: section identifier
        - section_title: section title
        - page_start: starting page
        - page_end: ending page
        - char_start: start position in concatenated text
        - char_end: end position in concatenated text
    """
    text_parts = []
    provenance_map = []
    current_pos = 0

    # Extract from sections
    sections = parsed_data.get("sections", [])
    for section in sections:
        section_text_parts = []

        # Main section text
        if section.get("text"):
            section_text_parts.append(section["text"])

        # Extract from tables
        for table in section.get("tables", []):
            if table.get("caption"):
                section_text_parts.append(f"Table: {table['caption']}")
            if table.get("text_content"):
                section_text_parts.append(table["text_content"])

        # Extract from figures
        for figure in section.get("figures", []):
            if figure.get("caption"):
                section_text_parts.append(f"Figure: {figure['caption']}")

            # Extract legend items
            for legend_item in figure.get("legend_items", []):
                if legend_item.get("description"):
                    section_text_parts.append(
                        f"{legend_item.get('number', '')}) {legend_item['description']}"
                    )

        # Concatenate section text
        if section_text_parts:
            section_text = "\n\n".join(section_text_parts)
            text_parts.append(section_text)

            # Record provenance
            char_start = current_pos
            char_end = current_pos + len(section_text)

            provenance_map.append({
                "text": section_text,
                "section_id": section.get("section_id", "unknown"),
                "section_title": section.get("title", ""),
                "page_start": section.get("page_start", 0),
                "page_end": section.get("page_end", 0),
                "char_start": char_start,
                "char_end": char_end,
            })

            # Update position (adding 2 for \n\n separator)
            current_pos = char_end + 2

    concatenated_text = "\n\n".join(text_parts)
    return concatenated_text, provenance_map


def chunk_text_with_provenance(
    text: str,
    provenance_map: List[Dict[str, Any]],
    max_tokens: int = 1000
) -> List[Tuple[str, List[Dict[str, Any]]]]:
    """
    Split text into chunks while maintaining provenance information.

    Returns:
        List of (chunk_text, chunk_provenance) tuples

        chunk_provenance: List of sections that overlap with this chunk
    """
    max_chars = max_tokens * 4
    chunks = []

    # Split by paragraphs first
    paragraphs = text.split("\n\n")
    current_chunk = ""
    current_chunk_start = 0

    for para in paragraphs:
        if len(current_chunk) + len(para) + 2 <= max_chars:
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para
                current_chunk_start = text.find(para)
        else:
            if current_chunk:
                # Find provenance for this chunk
                chunk_start = current_chunk_start
                chunk_end = chunk_start + len(current_chunk)
                chunk_prov = find_provenance_for_range(chunk_start, chunk_end, provenance_map)
                chunks.append((current_chunk, chunk_prov))

            current_chunk = para
            current_chunk_start = text.find(para, current_chunk_start + len(current_chunk) if current_chunk else 0)

    # Add last chunk
    if current_chunk:
        chunk_start = current_chunk_start
        chunk_end = chunk_start + len(current_chunk)
        chunk_prov = find_provenance_for_range(chunk_start, chunk_end, provenance_map)
        chunks.append((current_chunk, chunk_prov))

    return chunks


def find_provenance_for_range(
    start: int,
    end: int,
    provenance_map: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Find all provenance entries that overlap with the given character range.
    """
    overlapping = []
    for prov in provenance_map:
        # Check if ranges overlap
        if not (prov["char_end"] <= start or prov["char_start"] >= end):
            overlapping.append(prov)
    return overlapping


def inject_provenance_into_entities(
    entities: List[dict],
    chunk_provenance: List[Dict[str, Any]]
) -> List[dict]:
    """
    Inject provenance information (section_id, pages, title) into extracted entities.

    Each entity gets detailed span information showing where it was found.
    """
    if not chunk_provenance:
        return entities

    # Inject into entities
    for entity in entities:
        if "spans" not in entity or not entity["spans"]:
            entity["spans"] = []

        # Add span information from chunk provenance
        for prov in chunk_provenance:
            # Create detailed span with all available information
            span = {
                "section_id": prov["section_id"],
                "section_title": prov["section_title"],
                "page_start": prov["page_start"],
                "page_end": prov["page_end"],
                "source_text": f"Section: {prov['section_title'][:80]}... (p.{prov['page_start']}-{prov['page_end']})"
            }

            # Avoid duplicate spans for same section
            # Check if we already have a span for this section
            already_exists = any(
                s.get("section_id") == span["section_id"]
                for s in entity["spans"]
            )

            if not already_exists:
                entity["spans"].append(span)

    return entities


# === Text Processing (continued) ===

def chunk_text_legacy(text: str, max_tokens: int = 1000) -> List[str]:
    """
    Legacy chunk_text function for backwards compatibility.

    Split text into chunks of approximately max_tokens.
    Uses simple heuristic: ~4 characters per token.
    """
    max_chars = max_tokens * 4
    chunks = []

    # Split by paragraphs first
    paragraphs = text.split("\n\n")
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) + 2 <= max_chars:
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = para

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


# === Prompt Engineering ===

def build_extraction_prompt(
    text_chunk: str,
    profile_name: str,
    allowed_types: List[str],
    allowed_relations: List[str],
    schema_types: List[str],
    schema_relations: List[str],
    simplified: bool = False,
) -> str:
    """
    Build a structured prompt for entity and relation extraction.

    The prompt is dynamically constructed based on the profile and schema.

    Args:
        simplified: If True, use a more constrained prompt for retry attempts
    """
    allowed_types_str = ", ".join(allowed_types)
    allowed_relations_str = ", ".join(allowed_relations)

    profile_rules: List[str] = []
    profile_examples = ""

    if profile_name == "troubleshooting":
        profile_rules.append(
            "Se il testo contiene parole chiave come 'failure', 'fault', 'error', 'malfunction', assegna sempre type=FailureMode (mai MachineMode)."
        )
        profile_rules.append(
            f"Limita le entità ai soli tipi consentiti: {allowed_types_str}."
        )
    elif profile_name == "product_technical":
        profile_rules.append(
            "Per ogni specifica numerica crea un'entità ParameterSpec collegata al Product/Component con hasSpec e collega un'entità Unit tramite hasUnit."
        )
        profile_rules.append(
            "Non collegare mai direttamente un'Unit a Product o Component: l'arco hasUnit deve partire dal ParameterSpec."
        )
        profile_rules.append(
            "Se presente un valore numerico, usa i campi nominal_value (numero) e unit_raw (stringa UCUM quando possibile)."
        )
        profile_examples = """Esempio:
```json
{
  "entities": [
    {"id": "PR_01", "type": "Product", "name": "Espresso Machine", "confidence": 0.93},
    {"id": "PS_01", "type": "ParameterSpec", "name": "Steam pressure", "nominal_value": 1.2, "unit_raw": "bar", "confidence": 0.9},
    {"id": "UN_01", "type": "Unit", "name": "bar", "confidence": 0.9}
  ],
  "relations": [
    {"type": "hasSpec", "from_ref": "PR_01", "to_ref": "PS_01", "confidence": 0.9},
    {"type": "hasUnit", "from_ref": "PS_01", "to_ref": "UN_01", "confidence": 0.88}
  ]
}
```"""
    elif profile_name == "testing":
        profile_rules.append(
            "Crea entità TestCase, SafetyNotice, Measurement, AcceptanceCriterion, Equipment, ApplicabilityClause, NormativeReference, CrossReference quando il testo lo indica."
        )
        profile_rules.append(
            "Collega i TestCase con le altre entità usando relazioni disponibili (es. requires→Equipment/ApplicabilityClause, validatedBy→Measurement, constrainedBy→AcceptanceCriterion, refersTo→CrossReference, appliesTo→ApplicabilityClause)."
        )
        profile_rules.append(
            "Ogni relazione deve riferirsi a ID di entità esistenti; se non sei sicuro dei riferimenti esatti, ometti la relazione."
        )
        profile_rules.append(
            "Includi nel Measurement eventuali valori, intervalli o tolleranze; usa i campi nominal_value/min_value/max_value/tolerance/unit_raw quando presenti nel testo."
        )

    common_rules = [
        "Rispondere solo con JSON valido, senza testo extra o commenti.",
        f"Usa esclusivamente questi tipi di entità: {allowed_types_str}.",
        f"Limita le relazioni a: {allowed_relations_str}.",
        "Deduplica entità con la stessa coppia (type, name) mantenendo quella con confidence più alta.",
        "Mantieni gli ID coerenti nel formato <TIPO_ABBR>_<NUM> (es. CT_01, FM_02).",
        "Ogni relazione DEVE avere type, from_ref, to_ref, confidence e deve riferirsi a ID di entità presenti; se non puoi determinare entrambi i riferimenti, non emettere la relazione.",
    ]

    all_rules = common_rules + profile_rules
    rules_block = "\n".join(f"{idx + 1}. {rule}" for idx, rule in enumerate(all_rules))

    if simplified:
        prompt = f"""Extract entities and relations from the following technical text. Return ONLY valid JSON, no comments or extra text.

Profile: {profile_name}
Allowed entity types: {allowed_types_str}
Allowed relations: {allowed_relations_str}
Rules:
- No duplicate entities with the same (type, name).
- Apply profile-specific constraints: {"; ".join(profile_rules) if profile_rules else "respect allowed types and relations only."}
- Emit relations only when from_ref/to_ref point to existing entity IDs; otherwise skip them.

Required JSON format:
{{
  "entities": [{{"id": "TYPE_01", "type": "Type", "name": "Name", "confidence": 0.9}}],
  "relations": [{{"type": "relationType", "from_ref": "ID1", "to_ref": "ID2", "confidence": 0.9}}]
}}

Text:
{text_chunk}

JSON output:"""
    else:
        prompt = f"""Analizza il seguente testo estratto da un manuale tecnico e restituisci SOLO UN JSON VALIDO conforme a questo schema:

{{ "entities": [...], "relations": [...] }}

⚠️ IMPORTANTE: Rispondere SOLO con JSON valido, senza testo aggiuntivo, commenti o spiegazioni.

**Profilo attivo**: {profile_name}

**Tipi di entità ammessi per questo profilo**:
{json.dumps(allowed_types, indent=2)}

**Relazioni ammesse per questo profilo**:
{json.dumps(allowed_relations, indent=2)}

**Regole obbligatorie**:
{rules_block}

{profile_examples}

**Testo da analizzare**:

{text_chunk}

**JSON output**:"""

    return prompt


# === AI Integration ===

def call_openai_api(
    client: OpenAI,
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
) -> Optional[str]:
    """
    Call OpenAI API for entity extraction.

    Returns the response text or None on failure.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert in technical knowledge extraction. You extract entities and relations from technical manuals and return them as structured JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        return response.choices[0].message.content
    except Exception as exc:
        logging.error(f"OpenAI API call failed: {exc}")
        return None


def extract_with_retry(
    client: OpenAI,
    text_chunk: str,
    profile_name: str,
    allowed_types: List[str],
    allowed_relations: List[str],
    schema_types: List[str],
    schema_relations: List[str],
    extractor_config: dict,
    logger: logging.Logger,
    max_retries: int = 2,
) -> Optional[dict]:
    """
    Extract entities and relations with automatic retry on failure.

    Retry strategy:
    1. First attempt: Full detailed prompt with temperature from config
    2. Second attempt: Simplified prompt with temperature 0.1
    3. Third attempt: Simplified prompt with temperature 0.0
    """
    model = extractor_config.get("model", "gpt-4o-mini")
    timeout = extractor_config.get("request_timeout", 60)
    max_tokens = extractor_config.get("max_output_tokens", 1500)

    for attempt in range(max_retries + 1):
        # Adjust parameters based on attempt
        if attempt == 0:
            # First attempt: normal parameters
            temperature = extractor_config.get("temperature", 0.2)
            simplified = False
        elif attempt == 1:
            # Second attempt: simplified prompt, lower temperature
            temperature = 0.1
            simplified = True
            logger.info("Retry attempt 1: using simplified prompt with temperature 0.1")
        else:
            # Third attempt: simplified prompt, temperature 0
            temperature = 0.0
            simplified = True
            logger.info("Retry attempt 2: using simplified prompt with temperature 0.0")

        # Build prompt
        prompt = build_extraction_prompt(
            text_chunk,
            profile_name,
            allowed_types,
            allowed_relations,
            schema_types,
            schema_relations,
            simplified=simplified,
        )

        # Call API
        response = call_openai_api(
            client,
            prompt,
            model,
            temperature,
            max_tokens,
            timeout,
        )

        if not response:
            logger.warning(f"No response from AI (attempt {attempt + 1}/{max_retries + 1})")
            if attempt < max_retries:
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            else:
                return None

        # Parse response with safe parser
        extraction = safe_json_parse(response, logger)

        if extraction and "entities" in extraction and "relations" in extraction:
            return extraction
        else:
            logger.warning(f"Failed to extract valid JSON (attempt {attempt + 1}/{max_retries + 1})")
            if attempt < max_retries:
                time.sleep(2 ** attempt)
                continue

    return None


# === Validation ===

def validate_extraction(
    data: dict,
    allowed_types: List[str],
    allowed_relations: List[str],
) -> Tuple[bool, List[str]]:
    """
    Validate extracted entities and relations.

    Returns (is_valid, warnings).
    """
    warnings = []

    # Check structure
    if "entities" not in data:
        warnings.append("Missing 'entities' field")
        return False, warnings

    if "relations" not in data:
        warnings.append("Missing 'relations' field")
        return False, warnings

    # Validate entities
    entity_ids = set()
    for idx, entity in enumerate(data.get("entities", [])):
        if "id" not in entity:
            warnings.append(f"Entity {idx} missing 'id'")
        else:
            entity_ids.add(entity["id"])

        if "type" not in entity:
            warnings.append(f"Entity {idx} missing 'type'")
        elif entity["type"] not in allowed_types:
            warnings.append(f"Entity {idx} has invalid type: {entity['type']}")

        if "name" not in entity:
            warnings.append(f"Entity {idx} missing 'name'")

        if "confidence" not in entity:
            warnings.append(f"Entity {idx} missing 'confidence'")
        elif not isinstance(entity["confidence"], (int, float)) or not 0 <= entity["confidence"] <= 1:
            warnings.append(f"Entity {idx} has invalid confidence: {entity['confidence']}")

    # Validate relations
    for idx, relation in enumerate(data.get("relations", [])):
        if "type" not in relation:
            warnings.append(f"Relation {idx} missing 'type'")
        elif relation["type"] not in allowed_relations:
            warnings.append(f"Relation {idx} has invalid type: {relation['type']}")

        if "from_ref" not in relation:
            warnings.append(f"Relation {idx} missing 'from_ref'")
        elif relation["from_ref"] not in entity_ids:
            warnings.append(f"Relation {idx} references unknown entity: {relation['from_ref']}")

        if "to_ref" not in relation:
            warnings.append(f"Relation {idx} missing 'to_ref'")
        elif relation["to_ref"] not in entity_ids:
            warnings.append(f"Relation {idx} references unknown entity: {relation['to_ref']}")

        if "confidence" not in relation:
            warnings.append(f"Relation {idx} missing 'confidence'")
        elif not isinstance(relation["confidence"], (int, float)) or not 0 <= relation["confidence"] <= 1:
            warnings.append(f"Relation {idx} has invalid confidence: {relation['confidence']}")

    is_valid = len(warnings) == 0
    return is_valid, warnings


# === Normalization ===

def generate_deterministic_id(entity_type: str, entity_name: str, profile_name: str = "") -> str:
    """
    Generate deterministic ID with namespace pattern.

    Examples:
        Product, "Citiz" -> "ns:Product/citiz"
        ComponentType, "NTC Temperature Sensor" -> "ns:ComponentType/ntc_temperature_sensor"
        MachineMode, "Descaling Mode" -> "ns:Mode/descaling_mode"
        FailureMode, "No water flow" -> "ns:FM/no_water_flow"
    """
    # Type-specific namespace prefixes
    namespace_map = {
        "Product": "Product",
        "ComponentType": "ComponentType",
        "Component": "Component",
        "ParameterSpec": "PS",
        "Unit": "Unit",
        "MachineMode": "Mode",
        "State": "State",
        "ProcessStep": "Step",
        "TestSpec": "Test",
        "FailureMode": "FM",
        "RepairAction": "RA",
        "MaintenanceTask": "MT",
        "Tool": "Tool",
        "Consumable": "Consumable",
        "RatingPlate": "RatingPlate",
    }

    prefix = namespace_map.get(entity_type, entity_type)
    slug = slugify(entity_name)

    return f"ns:{prefix}/{slug}"


def deduplicate_entities(entities: List[dict], profile_name: str) -> Tuple[List[dict], Dict[str, str]]:
    """
    Deduplicate entities by normalized name and type.

    Returns:
        (unique_entities, id_mapping)
    """
    # Map from (type, normalized_name) to entity
    entity_map: Dict[Tuple[str, str], dict] = {}
    id_mapping: Dict[str, str] = {}

    for entity in entities:
        entity_type = entity.get("type", "")
        entity_name = entity.get("name", "")
        normalized_name = normalize_text(entity_name)

        key = (entity_type, normalized_name)

        # Generate deterministic ID
        new_id = generate_deterministic_id(entity_type, entity_name, profile_name)
        old_id = entity.get("id", new_id)
        id_mapping[old_id] = new_id
        entity["id"] = new_id

        if key in entity_map:
            # Duplicate found - merge spans and keep entity with higher confidence
            existing = entity_map[key]

            # Merge spans from both entities
            existing_spans = existing.get("spans", [])
            new_spans = entity.get("spans", [])
            merged_spans = existing_spans + new_spans

            # Deduplicate spans by section_id
            seen_sections = set()
            unique_spans = []
            for span in merged_spans:
                section_id = span.get("section_id", "")
                if section_id and section_id not in seen_sections:
                    seen_sections.add(section_id)
                    unique_spans.append(span)

            # Keep the entity with higher confidence, but merge spans
            if entity.get("confidence", 0) > existing.get("confidence", 0):
                entity["spans"] = unique_spans
                entity_map[key] = entity
                # Update mapping to point to this entity
                id_mapping[existing["id"]] = new_id
            else:
                existing["spans"] = unique_spans
                entity_map[key] = existing
        else:
            entity_map[key] = entity

    unique_entities = list(entity_map.values())
    return unique_entities, id_mapping


def deduplicate_relations(relations: List[dict]) -> List[dict]:
    """
    Deduplicate relations by (type, from_ref, to_ref) key.

    Keeps the relation with highest confidence.
    """
    relation_map: Dict[Tuple[str, str, str], dict] = {}

    for relation in relations:
        key = (
            relation.get("type", ""),
            relation.get("from_ref", ""),
            relation.get("to_ref", ""),
        )

        if key in relation_map:
            # Keep the one with higher confidence
            existing = relation_map[key]
            if relation.get("confidence", 0) > existing.get("confidence", 0):
                relation_map[key] = relation
        else:
            relation_map[key] = relation

    return list(relation_map.values())


def fix_troubleshooting_semantics(
    entities: List[dict],
    relations: List[dict],
) -> Tuple[List[dict], List[dict]]:
    """
    Fix semantic issues in troubleshooting profile:

    1. FailureMode --mitigatedBy--> RepairAction (correct)
    2. FailureMode --mitigatedBy--> Tool (wrong, should be via RepairAction)
    3. RepairAction --requiresTool--> Tool
    4. RepairAction --requiresConsumable--> Consumable
    5. FailureMode --affects--> Component
    """
    # Build entity lookup
    entity_by_id = {e["id"]: e for e in entities}

    fixed_relations = []
    new_entities = []
    repair_action_counter = 0

    for relation in relations:
        rel_type = relation.get("type", "")
        from_id = relation.get("from_ref", "")
        to_id = relation.get("to_ref", "")

        from_entity = entity_by_id.get(from_id)
        to_entity = entity_by_id.get(to_id)

        if not from_entity or not to_entity:
            # Invalid reference, skip
            continue

        # Check for FailureMode --mitigatedBy--> Tool/Consumable
        if (
            rel_type == "mitigatedBy"
            and from_entity.get("type") == "FailureMode"
            and to_entity.get("type") in ["Tool", "Consumable"]
        ):
            # Create intermediate RepairAction
            repair_action_counter += 1
            ra_name = f"Use {to_entity.get('name', 'tool')}"
            ra_id = f"ns:RA/use_{slugify(to_entity.get('name', 'tool'))}_{repair_action_counter}"

            repair_action = {
                "id": ra_id,
                "type": "RepairAction",
                "name": ra_name,
                "confidence": relation.get("confidence", 0.8),
            }

            new_entities.append(repair_action)
            entity_by_id[ra_id] = repair_action

            # Add FailureMode --mitigatedBy--> RepairAction
            fixed_relations.append({
                "type": "mitigatedBy",
                "from_ref": from_id,
                "to_ref": ra_id,
                "confidence": relation.get("confidence", 0.8),
            })

            # Add RepairAction --requiresTool/Consumable--> Tool/Consumable
            req_type = "requiresTool" if to_entity.get("type") == "Tool" else "requiresConsumable"
            fixed_relations.append({
                "type": req_type,
                "from_ref": ra_id,
                "to_ref": to_id,
                "confidence": relation.get("confidence", 0.8),
            })
        else:
            # Keep relation as-is
            fixed_relations.append(relation)

    # Merge new entities
    all_entities = entities + new_entities

    return all_entities, fixed_relations


def normalize_extraction(data: dict, profile_name: str) -> dict:
    """
    Normalize the entire extraction result.

    Includes:
    - Deterministic ID generation with namespace
    - Entity deduplication
    - Relation deduplication
    - Semantic fixes for troubleshooting profile
    """
    entities = data.get("entities", [])
    relations = data.get("relations", [])

    # Step 1: Deduplicate and normalize entities
    entities, id_mapping = deduplicate_entities(entities, profile_name)

    # Step 2: Update relation references with new IDs, skipping malformed relations
    normalized_relations = []
    logger = logging.getLogger(f"neural_extraction.normalize.{profile_name}")

    for idx, relation in enumerate(relations):
        from_ref = relation.get("from_ref")
        to_ref = relation.get("to_ref")
        rel_type = relation.get("type")

        if not from_ref or not to_ref or not rel_type:
            logger.warning(
                "Skipping relation %s missing required fields (type=%s, from_ref=%s, to_ref=%s)",
                idx,
                rel_type,
                from_ref,
                to_ref,
            )
            continue

        relation["from_ref"] = id_mapping.get(from_ref, from_ref)
        relation["to_ref"] = id_mapping.get(to_ref, to_ref)
        normalized_relations.append(relation)

    # Step 3: Fix troubleshooting semantics if needed
    if profile_name == "troubleshooting":
        entities, normalized_relations = fix_troubleshooting_semantics(entities, normalized_relations)

    # Step 4: Deduplicate relations
    normalized_relations = deduplicate_relations(normalized_relations)

    return {
        "entities": entities,
        "relations": normalized_relations,
    }


# === Quality Metrics ===

def generate_quality_report(
    kg_data: dict,
    warnings: List[str],
    profile_name: str,
) -> dict:
    """
    Generate quality metrics for the extracted knowledge graph.
    """
    entities = kg_data.get("entities", [])
    relations = kg_data.get("relations", [])

    # Calculate average confidence
    entity_confidences = [e.get("confidence", 0) for e in entities]
    relation_confidences = [r.get("confidence", 0) for r in relations]

    avg_entity_conf = sum(entity_confidences) / len(entity_confidences) if entity_confidences else 0
    avg_relation_conf = sum(relation_confidences) / len(relation_confidences) if relation_confidences else 0
    overall_conf = (avg_entity_conf + avg_relation_conf) / 2 if (entities or relations) else 0

    # Entity type distribution
    entity_types = {}
    for entity in entities:
        etype = entity.get("type", "unknown")
        entity_types[etype] = entity_types.get(etype, 0) + 1

    # Relation type distribution
    relation_types = {}
    for relation in relations:
        rtype = relation.get("type", "unknown")
        relation_types[rtype] = relation_types.get(rtype, 0) + 1

    return {
        "json_valid": len(warnings) == 0,
        "entities_count": len(entities),
        "relations_count": len(relations),
        "warnings": warnings,
        "overall_confidence": round(overall_conf, 3),
        "avg_entity_confidence": round(avg_entity_conf, 3),
        "avg_relation_confidence": round(avg_relation_conf, 3),
        "entity_type_distribution": entity_types,
        "relation_type_distribution": relation_types,
        "profile": profile_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


# === Main Processing ===

def process_profile(
    profile_name: str,
    profile_config: dict,
    extractor_config: dict,
    schema_data: dict,
    root_dir: Path,
    client: Optional[OpenAI],
    dry_run: bool = False,
) -> bool:
    """
    Process a single neural extraction profile.

    Returns True on success, False on failure.
    """
    logger = logging.getLogger(f"neural_extraction.{profile_name}")
    logger.info(f"Processing profile: {profile_name}")

    # Get configuration
    input_glob = profile_config.get("input_glob")
    output_dir = root_dir / profile_config.get("output_dir")
    allowed_types = profile_config.get("class_map", [])
    allowed_relations = profile_config.get("relation_map", [])

    schema_types = schema_data.get("allowed_types", [])
    schema_relations = schema_data.get("allowed_relations", [])

    # Find input files
    input_pattern = root_dir / input_glob
    input_files = sorted(Path(root_dir).glob(input_glob))

    if not input_files:
        logger.warning(f"No input files found for pattern: {input_glob}")
        return False

    logger.info(f"Found {len(input_files)} input file(s)")

    # Prepare output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate all extractions
    all_entities = []
    all_relations = []
    all_warnings = []
    all_provenance_sections = {}  # Map section_id -> section details

    # Process each input file
    for input_file in input_files:
        logger.info(f"Processing: {input_file.name}")

        # Load parsed document
        with input_file.open("r", encoding="utf-8") as fh:
            parsed_data = json.load(fh)

        # Extract text with provenance tracking
        text, provenance_map = extract_text_with_provenance(parsed_data)
        if not text:
            logger.warning(f"No text extracted from {input_file.name}")
            continue

        # Record sections for final provenance
        for prov in provenance_map:
            section_id = prov["section_id"]
            if section_id not in all_provenance_sections:
                all_provenance_sections[section_id] = {
                    "section_id": section_id,
                    "title": prov["section_title"],
                    "page_start": prov["page_start"],
                    "page_end": prov["page_end"],
                    "document": input_file.name,
                }

        # Split into chunks with provenance
        chunks_with_prov = chunk_text_with_provenance(
            text,
            provenance_map,
            extractor_config.get("max_tokens_per_chunk", 1000)
        )
        logger.info(f"Split text into {len(chunks_with_prov)} chunk(s)")

        # Process each chunk
        for chunk_idx, (chunk, chunk_provenance) in enumerate(chunks_with_prov):
            logger.debug(f"Processing chunk {chunk_idx + 1}/{len(chunks_with_prov)}")

            # Log provenance for this chunk
            if chunk_provenance:
                sections_info = ", ".join([
                    f"{p['section_id']} (p.{p['page_start']}-{p['page_end']})"
                    for p in chunk_provenance[:3]  # Show first 3 sections
                ])
                logger.debug(f"  Chunk sources: {sections_info}")

            if dry_run:
                # Build a dummy prompt just to show length
                prompt = build_extraction_prompt(
                    chunk,
                    profile_name,
                    allowed_types,
                    allowed_relations,
                    schema_types,
                    schema_relations,
                )
                logger.info(f"[DRY RUN] Would call AI with prompt length: {len(prompt)}")
                continue

            # Call AI with retry
            if not client:
                logger.error("OpenAI client not initialized")
                return False

            extraction = extract_with_retry(
                client,
                chunk,
                profile_name,
                allowed_types,
                allowed_relations,
                schema_types,
                schema_relations,
                extractor_config,
                logger,
                max_retries=2,
            )

            if not extraction:
                logger.warning(f"Failed to extract data for chunk {chunk_idx + 1} after retries")
                continue

            # Inject provenance into extracted entities
            if "entities" in extraction and chunk_provenance:
                extraction["entities"] = inject_provenance_into_entities(
                    extraction["entities"],
                    chunk_provenance
                )

            # Save raw output if enabled (save the final successful extraction with provenance)
            if extractor_config.get("save_raw_outputs", True):
                raw_dir = output_dir / "raw"
                raw_dir.mkdir(exist_ok=True)
                raw_file = raw_dir / f"{input_file.stem}_chunk_{chunk_idx:03d}_extraction.json"
                with raw_file.open("w", encoding="utf-8") as fh:
                    json.dump({
                        "extraction": extraction,
                        "chunk_provenance": chunk_provenance,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }, fh, indent=2, ensure_ascii=False)

            # Validate
            is_valid, warnings = validate_extraction(
                extraction,
                allowed_types,
                allowed_relations,
            )

            if not is_valid:
                logger.warning(f"Validation failed for chunk {chunk_idx + 1}: {warnings[:5]}")  # Show first 5 warnings
                all_warnings.extend(warnings)

            # Normalize
            normalized = normalize_extraction(extraction, profile_name)

            # Aggregate
            all_entities.extend(normalized.get("entities", []))
            all_relations.extend(normalized.get("relations", []))

    if dry_run:
        logger.info("[DRY RUN] Completed")
        return True

    # Final normalization and deduplication of aggregated data
    logger.info(f"Applying final normalization and deduplication...")
    final_normalized = normalize_extraction(
        {"entities": all_entities, "relations": all_relations},
        profile_name
    )

    final_entities = final_normalized["entities"]
    final_relations = final_normalized["relations"]

    logger.info(f"After deduplication: {len(final_entities)} entities, {len(final_relations)} relations")

    # Build final KG
    kg_data = {
        "document_code": f"KG_{profile_name.upper()}",
        "ingestion_id": str(uuid.uuid4()),
        "extraction_version": "neural_v1.2",
        "datasource_code": "NEURAL_EXTRACTION",
        "extractor": {
            "model": extractor_config.get("model", "gpt-4o-mini"),
            "prompt_id": "neural_extraction_v1_hardened",
            "temperature": extractor_config.get("temperature", 0.2),
            "max_tokens": extractor_config.get("max_output_tokens", 1500),
        },
        "allowed_types": allowed_types,
        "allowed_relations": allowed_relations,
        "entities": final_entities,
        "relations": final_relations,
        "provenance": {
            "overall_confidence": 0.0,  # Will be calculated in quality report
            "documents_processed": [f.name for f in input_files],
            "sections_used": list(all_provenance_sections.values()),
            "total_sections": len(all_provenance_sections),
            "notes": f"Neural extraction for profile {profile_name} with provenance tracking and semantic normalization",
        },
    }

    # Generate quality report
    quality = generate_quality_report(kg_data, all_warnings, profile_name)
    kg_data["quality"] = quality
    kg_data["provenance"]["overall_confidence"] = quality["overall_confidence"]

    # Save KG
    partial_kg_dir = root_dir / "output" / "neural_extraction" / "Partial_KG"
    partial_kg_dir.mkdir(parents=True, exist_ok=True)
    kg_filename = f"{profile_name}_kg.json"
    kg_file = partial_kg_dir / kg_filename

    with kg_file.open("w", encoding="utf-8") as fh:
        json.dump(kg_data, fh, indent=2, ensure_ascii=False)

    logger.info(f"Saved KG to: {kg_file}")
    logger.info(f"Entities: {len(final_entities)}, Relations: {len(final_relations)}")

    # Save quality report separately
    quality_file = output_dir / "quality.json"
    with quality_file.open("w", encoding="utf-8") as fh:
        json.dump(quality, fh, indent=2, ensure_ascii=False)

    logger.info(f"Saved quality report to: {quality_file}")

    return True


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Neural Knowledge Extraction Module"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="all",
        help="Profile to process (product_technical, operation_modes, troubleshooting, or all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without calling AI (for testing)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args(argv)

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Determine paths
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    if args.config is None:
        config_path = project_root / "config.yaml"
    else:
        config_path = args.config
        if not config_path.is_absolute():
            config_path = (Path.cwd() / config_path).resolve()

    if not config_path.exists():
        logging.error(f"Configuration file not found: {config_path}")
        return 1

    # Load configuration
    config = load_config(config_path)
    extractor_config = config.get("neural_extractor", {})
    profiles_config = config.get("neural_extraction_profiles", {})

    model_name = extractor_config.get("model", "gpt-4o-mini")
    provider_name = extractor_config.get("provider", "openai")
    logging.info(f"Neural extractor configured provider/model: {provider_name}/{model_name}")

    if not profiles_config:
        logging.error("No neural_extraction_profiles found in config.yaml")
        return 1

    # Load schema
    schema_path = project_root / "schemas" / "neural_extraction.json"
    if not schema_path.exists():
        logging.error(f"Schema file not found: {schema_path}")
        return 1

    schema_data = load_schema(schema_path)

    # Initialize OpenAI client
    client = None
    if not args.dry_run:
        if OpenAI is None:
            logging.error("OpenAI library not installed. Run: pip install openai")
            return 1

        api_key_env = extractor_config.get("api_key_env", "OPENAI_API_KEY")
        api_key = os.getenv(api_key_env)

        if not api_key:
            logging.error(f"OpenAI API key not found in environment variable: {api_key_env}")
            return 1

        client = OpenAI(api_key=api_key)

    # Determine which profiles to process
    if args.profile == "all":
        profiles_to_run = list(profiles_config.keys())
    else:
        if args.profile not in profiles_config:
            logging.error(f"Profile not found: {args.profile}")
            return 1
        profiles_to_run = [args.profile]

    # Process each profile
    overall_status = 0

    for profile_name in profiles_to_run:
        profile_config = profiles_config[profile_name]

        try:
            success = process_profile(
                profile_name,
                profile_config,
                extractor_config,
                schema_data,
                project_root,
                client,
                dry_run=args.dry_run,
            )

            if not success:
                overall_status = 2
        except Exception as exc:
            logging.exception(f"Failed to process profile {profile_name}: {exc}")
            overall_status = 2

    return overall_status


if __name__ == "__main__":
    sys.exit(main())
