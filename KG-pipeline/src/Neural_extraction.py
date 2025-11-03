#!/usr/bin/env python3
"""
Neural Extraction Module for Knowledge Graph Generation

This module performs AI-driven extraction of entities and relations from parsed documents,
generating three sub-KGs (product_technical, operation_modes, troubleshooting).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


# === Text Processing ===

def extract_text_from_parsed(parsed_data: dict) -> str:
    """
    Extract all relevant text from a parsed document JSON.

    Concatenates text from sections, tables, and figures to create
    a comprehensive text corpus for neural extraction.
    """
    text_parts = []

    # Extract from sections
    sections = parsed_data.get("sections", [])
    for section in sections:
        if section.get("text"):
            text_parts.append(section["text"])

        # Extract from tables
        for table in section.get("tables", []):
            if table.get("caption"):
                text_parts.append(f"Table: {table['caption']}")
            if table.get("text_content"):
                text_parts.append(table["text_content"])

        # Extract from figures
        for figure in section.get("figures", []):
            if figure.get("caption"):
                text_parts.append(f"Figure: {figure['caption']}")

            # Extract legend items
            for legend_item in figure.get("legend_items", []):
                if legend_item.get("description"):
                    text_parts.append(
                        f"{legend_item.get('number', '')}) {legend_item['description']}"
                    )

    return "\n\n".join(text_parts)


def chunk_text(text: str, max_tokens: int = 1000) -> List[str]:
    """
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
) -> str:
    """
    Build a structured prompt for entity and relation extraction.

    The prompt is dynamically constructed based on the profile and schema.
    """
    prompt = f"""Analizza il seguente testo estratto da un manuale tecnico e restituisci un JSON conforme a questo schema:

{{ "entities": [...], "relations": [...] }}

**Profilo attivo**: {profile_name}

**Tipi di entità ammessi per questo profilo**:
{json.dumps(allowed_types, indent=2)}

**Tipi di entità disponibili nello schema completo**:
{json.dumps(schema_types, indent=2)}

**Relazioni ammesse per questo profilo**:
{json.dumps(allowed_relations, indent=2)}

**Relazioni disponibili nello schema completo**:
{json.dumps(schema_relations, indent=2)}

**Regole**:
1. Estrarre SOLO entità e relazioni pertinenti al profilo attivo "{profile_name}".
2. Ogni entità deve avere: id (stringa univoca), type (uno dei tipi ammessi), name (nome dell'entità), confidence (numero tra 0 e 1).
3. Le relazioni devono indicare: type (una delle relazioni ammesse), from_ref (id entità sorgente), to_ref (id entità destinazione), confidence (numero tra 0 e 1).
4. Gli ID devono essere nel formato: <TIPO_ABBREVIATO>_<NUMERO> (es: CT_01 per ComponentType, C_01 per Component).
5. Mantenere la struttura JSON valida e conforme allo schema.
6. Non aggiungere testo fuori dal JSON.
7. Se un'entità ha proprietà opzionali rilevanti (es: ofType_ref, nominal_value, unit_raw), includerle.

**Esempio di output**:

{{
  "entities": [
    {{"id": "CT_01", "type": "ComponentType", "name": "Thermoblock", "confidence": 0.94}},
    {{"id": "C_01", "type": "Component", "name": "Pump", "ofType_ref": "CT_Pump", "confidence": 0.91}}
  ],
  "relations": [
    {{"type": "hasPart", "from_ref": "C_01", "to_ref": "CT_01", "confidence": 0.88}}
  ]
}}

**Testo da analizzare**:

{text_chunk}

**Output JSON**:"""

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


def extract_json_from_response(response: str) -> Optional[dict]:
    """
    Extract JSON from AI response, handling markdown code blocks.
    """
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
            return None

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as exc:
        logging.error(f"Failed to parse JSON: {exc}")
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

def normalize_entity_id(entity_id: str, entity_type: str) -> str:
    """
    Normalize entity ID to ensure consistent format.
    """
    # Extract type abbreviation
    type_abbrev = "".join([c for c in entity_type if c.isupper()])
    if not type_abbrev:
        type_abbrev = entity_type[:3].upper()

    # If ID already has correct format, keep it
    if entity_id.startswith(f"{type_abbrev}_"):
        return entity_id

    # Otherwise, generate new ID
    import hashlib
    hash_val = hashlib.md5(entity_id.encode()).hexdigest()[:6]
    return f"{type_abbrev}_{hash_val}"


def normalize_entities(entities: List[dict]) -> List[dict]:
    """
    Normalize entity IDs and names.
    """
    id_mapping = {}
    normalized = []

    for entity in entities:
        old_id = entity["id"]
        new_id = normalize_entity_id(old_id, entity["type"])
        id_mapping[old_id] = new_id

        entity["id"] = new_id
        entity["name"] = entity["name"].strip()

        normalized.append(entity)

    return normalized, id_mapping


def normalize_relations(relations: List[dict], id_mapping: Dict[str, str]) -> List[dict]:
    """
    Normalize relation references using the ID mapping.
    """
    normalized = []

    for relation in relations:
        relation["from_ref"] = id_mapping.get(relation["from_ref"], relation["from_ref"])
        relation["to_ref"] = id_mapping.get(relation["to_ref"], relation["to_ref"])
        normalized.append(relation)

    return normalized


def normalize_extraction(data: dict) -> dict:
    """
    Normalize the entire extraction result.
    """
    entities, id_mapping = normalize_entities(data.get("entities", []))
    relations = normalize_relations(data.get("relations", []), id_mapping)

    return {
        "entities": entities,
        "relations": relations,
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
        "generated_at": datetime.utcnow().isoformat() + "Z",
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

    # Process each input file
    for input_file in input_files:
        logger.info(f"Processing: {input_file.name}")

        # Load parsed document
        with input_file.open("r", encoding="utf-8") as fh:
            parsed_data = json.load(fh)

        # Extract text
        text = extract_text_from_parsed(parsed_data)
        if not text:
            logger.warning(f"No text extracted from {input_file.name}")
            continue

        # Split into chunks
        chunks = chunk_text(text, extractor_config.get("max_tokens_per_chunk", 1000))
        logger.info(f"Split text into {len(chunks)} chunk(s)")

        # Process each chunk
        for chunk_idx, chunk in enumerate(chunks):
            logger.debug(f"Processing chunk {chunk_idx + 1}/{len(chunks)}")

            # Build prompt
            prompt = build_extraction_prompt(
                chunk,
                profile_name,
                allowed_types,
                allowed_relations,
                schema_types,
                schema_relations,
            )

            if dry_run:
                logger.info(f"[DRY RUN] Would call AI with prompt length: {len(prompt)}")
                continue

            # Call AI
            if not client:
                logger.error("OpenAI client not initialized")
                return False

            response = call_openai_api(
                client,
                prompt,
                extractor_config.get("model", "gpt-4o-mini"),
                extractor_config.get("temperature", 0.2),
                extractor_config.get("max_output_tokens", 1500),
                extractor_config.get("request_timeout", 60),
            )

            if not response:
                logger.warning(f"No response from AI for chunk {chunk_idx + 1}")

                # Retry with simplified prompt if enabled
                if extractor_config.get("retry_on_failure", True):
                    logger.info("Retrying with simplified prompt...")
                    time.sleep(2)
                    # TODO: Implement simplified prompt

                continue

            # Save raw output if enabled
            if extractor_config.get("save_raw_outputs", True):
                raw_dir = output_dir / "raw"
                raw_dir.mkdir(exist_ok=True)
                raw_file = raw_dir / f"{input_file.stem}_chunk_{chunk_idx:03d}_raw.json"
                with raw_file.open("w", encoding="utf-8") as fh:
                    json.dump({
                        "prompt": prompt,
                        "response": response,
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                    }, fh, indent=2, ensure_ascii=False)

            # Parse response
            extraction = extract_json_from_response(response)
            if not extraction:
                logger.warning(f"Failed to extract JSON from response for chunk {chunk_idx + 1}")
                continue

            # Validate
            is_valid, warnings = validate_extraction(
                extraction,
                allowed_types,
                allowed_relations,
            )

            if not is_valid:
                logger.warning(f"Validation failed for chunk {chunk_idx + 1}: {warnings}")
                all_warnings.extend(warnings)

            # Normalize
            normalized = normalize_extraction(extraction)

            # Aggregate
            all_entities.extend(normalized.get("entities", []))
            all_relations.extend(normalized.get("relations", []))

    if dry_run:
        logger.info("[DRY RUN] Completed")
        return True

    # Deduplicate entities by ID
    unique_entities = {}
    for entity in all_entities:
        eid = entity["id"]
        if eid not in unique_entities:
            unique_entities[eid] = entity
        else:
            # Keep the one with higher confidence
            if entity.get("confidence", 0) > unique_entities[eid].get("confidence", 0):
                unique_entities[eid] = entity

    final_entities = list(unique_entities.values())

    # Build final KG
    kg_data = {
        "document_code": f"KG_{profile_name.upper()}",
        "ingestion_id": str(uuid.uuid4()),
        "extraction_version": "neural_v1.0",
        "datasource_code": "NEURAL_EXTRACTION",
        "extractor": {
            "model": extractor_config.get("model", "gpt-4o-mini"),
            "prompt_id": "neural_extraction_v1",
            "temperature": extractor_config.get("temperature", 0.2),
            "max_tokens": extractor_config.get("max_output_tokens", 1500),
        },
        "allowed_types": allowed_types,
        "allowed_relations": allowed_relations,
        "entities": final_entities,
        "relations": all_relations,
        "provenance": {
            "overall_confidence": 0.0,  # Will be calculated in quality report
            "sections_used": [f.name for f in input_files],
            "notes": f"Neural extraction for profile {profile_name}",
        },
    }

    # Generate quality report
    quality = generate_quality_report(kg_data, all_warnings, profile_name)
    kg_data["quality"] = quality
    kg_data["provenance"]["overall_confidence"] = quality["overall_confidence"]

    # Save KG
    kg_file = output_dir / "kg.json"
    with kg_file.open("w", encoding="utf-8") as fh:
        json.dump(kg_data, fh, indent=2, ensure_ascii=False)

    logger.info(f"Saved KG to: {kg_file}")
    logger.info(f"Entities: {len(final_entities)}, Relations: {len(all_relations)}")

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
