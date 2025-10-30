"""
Neural Extractor Module
Transforms parsed PDF documents into structured knowledge triples/entities
using an LLM (e.g. OpenAI GPT models).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from jsonschema import validate, ValidationError

    _JSONSCHEMA_AVAILABLE = True
except ModuleNotFoundError:
    validate = None  # type: ignore[assignment]
    _JSONSCHEMA_AVAILABLE = False

try:
    from openai import APIError as OpenAIAPIError
    from openai import OpenAI

    _OPENAI_AVAILABLE = True
except Exception:  # pragma: no cover - defensive import
    OpenAI = None  # type: ignore[assignment]
    OpenAIAPIError = Exception  # type: ignore[assignment]
    _OPENAI_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ExtractorSettings:
    model: str
    temperature: float
    max_output_tokens: int
    request_timeout: int
    prompt_id: str
    api_key_env: str
    version: str
    datasource_code: str
    max_sections: int = 20
    max_chars_per_section: int = 1200
    max_total_chars: int = 12000


class NeuralExtractor:
    """
    Runs the neural extraction step over parsed PDF documents.
    """

    def __init__(
        self,
        schema_path: str,
        config: Optional[Dict[str, Any]] = None,
        settings_override: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.schema_path = Path(schema_path)
        if not self.schema_path.exists():
            raise FileNotFoundError(f"Neural extraction schema not found: {self.schema_path}")

        self.config = config or {}
        self.module_config = self.config.get("neural_extractor", {}) or {}

        datasource_code = (
            self.config.get("pdf_processing", {}).get("datasource_code", "UNKNOWN_DATASOURCE")
        )

        self.settings = self._build_settings(
            datasource_code=datasource_code,
            override=settings_override or {},
        )

        logger.info(
            "NeuralExtractor configured with model %s (prompt_id=%s)",
            self.settings.model,
            self.settings.prompt_id,
        )

        self.schema_template = self._load_schema_template()
        self.allowed_types = self.schema_template.get("allowed_types", [])

        self._client = self._initialise_client()

    # --------------------------------------------------------------------- #
    # Initialisation helpers
    # --------------------------------------------------------------------- #

    def _build_settings(
        self,
        datasource_code: str,
        override: Dict[str, Any],
    ) -> ExtractorSettings:
        prompt_id = self.module_config.get("prompt_id", "neural_extractor_prompt_v1")
        version = self.module_config.get("version", "neural_extractor_v1")
        settings = ExtractorSettings(
            model=self.module_config.get("model", "gpt-4o-mini"),
            temperature=float(self.module_config.get("temperature", 0.0)),
            max_output_tokens=int(self.module_config.get("max_output_tokens", 1024)),
            request_timeout=int(self.module_config.get("request_timeout", 60)),
            prompt_id=prompt_id,
            api_key_env=self.module_config.get("api_key_env", "OPENAI_API_KEY"),
            version=version,
            datasource_code=datasource_code,
            max_sections=int(self.module_config.get("max_sections", 20)),
            max_chars_per_section=int(self.module_config.get("max_chars_per_section", 1200)),
            max_total_chars=int(self.module_config.get("max_total_chars", 12000)),
        )

        for key, value in override.items():
            if hasattr(settings, key):
                setattr(settings, key, value)

        return settings

    def _load_schema_template(self) -> Dict[str, Any]:
        with open(self.schema_path, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                raise ValueError(f"Invalid JSON in schema file {self.schema_path}: {exc}") from exc

    def _initialise_client(self):
        if not self.module_config.get("enabled", False):
            logger.info("Neural extractor disabled via configuration.")
            return None

        if not _OPENAI_AVAILABLE:
            logger.warning("openai package not installed. Neural extraction will be skipped.")
            return None

        api_key = os.getenv(self.settings.api_key_env)
        if not api_key:
            logger.warning(
                "API key environment variable %s is not set. Neural extraction will be skipped.",
                self.settings.api_key_env,
            )
            return None

        client = OpenAI(api_key=api_key, timeout=self.settings.request_timeout)
        return client

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    def run_batch(
        self,
        parsed_documents: Iterable[Dict[str, Any]],
        output_dir: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        output_path = Path(output_dir) if output_dir else None

        if output_path:
            output_path.mkdir(parents=True, exist_ok=True)

        if self._client is None:
            logger.info("Neural extractor client unavailable. Skipping extraction.")
            return results

        for parsed_doc in parsed_documents:
            try:
                payload = self.extract_document(parsed_doc)
            except Exception as exc:
                document_code = parsed_doc.get("document_code", "UNKNOWN")
                logger.error("Neural extraction failed for %s: %s", document_code, exc)
                continue

            results.append(payload)

            if output_path:
                document_code = payload.get("document_code", "doc")
                filename = output_path / f"{document_code}_neural.json"
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2, ensure_ascii=False)
                logger.info("Saved neural extraction to %s", filename)

        return results

    def save_batch(self, extractions: List[Dict[str, Any]], output_file: str) -> None:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        batch = {
            "batch_id": f"neural_batch_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "processed_utc": datetime.utcnow().isoformat() + "Z",
            "document_count": len(extractions),
            "documents": extractions,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(batch, f, indent=2, ensure_ascii=False)

        logger.info("Saved neural extraction batch to %s", output_path)

    # --------------------------------------------------------------------- #
    # Core extraction logic
    # --------------------------------------------------------------------- #

    def extract_document(self, parsed_document: Dict[str, Any]) -> Dict[str, Any]:
        base_payload = self._build_base_payload(parsed_document)
        prompt, section_ids = self._build_prompt(parsed_document)

        llm_data: Dict[str, Any] = {}
        warnings: List[str] = []
        json_valid = False

        if self._client is None:
            warnings.append("LLM client not initialised; skipping extraction.")
        else:
            try:
                llm_data = self._invoke_model(prompt)
                json_valid = True
            except json.JSONDecodeError as exc:
                warnings.append(f"Model response was not valid JSON: {exc}")
                logger.warning("JSON decoding failed for neural extraction: %s", exc)
            except OpenAIAPIError as exc:
                warnings.append(f"OpenAI API error: {exc}")
                logger.error("OpenAI API error during neural extraction: %s", exc)
            except Exception as exc:  # pragma: no cover - defensive
                warnings.append(f"Unexpected error during neural extraction: {exc}")
                logger.error("Unexpected neural extraction error: %s", exc)

        entities = llm_data.get("entities", []) if isinstance(llm_data, dict) else []
        relations = llm_data.get("relations", []) if isinstance(llm_data, dict) else []
        provenance = llm_data.get("provenance", {}) if isinstance(llm_data, dict) else {}

        if provenance:
            base_payload["provenance"].update(provenance)
        else:
            base_payload["provenance"]["sections_used"] = section_ids

        base_payload["entities"] = self._ensure_entity_list(entities)
        base_payload["relations"] = self._ensure_relation_list(relations)

        warnings.extend(self._validate_payload(base_payload))

        base_payload["quality"] = {
            "json_valid": json_valid,
            "entities_count": len(base_payload["entities"]),
            "relations_count": len(base_payload["relations"]),
            "warnings": warnings,
        }

        if _JSONSCHEMA_AVAILABLE and isinstance(self.schema_template, dict):
            try:
                validate(instance=base_payload, schema=self._build_json_schema())
            except ValidationError as exc:
                logger.warning("Schema validation failed: %s", exc)
                base_payload["quality"]["warnings"].append(
                    f"Schema validation warning: {getattr(exc, 'message', str(exc))}"
                )

        return base_payload

    def _invoke_model(self, prompt: str) -> Dict[str, Any]:
        assert self._client is not None  # for mypy / type checkers

        response = self._client.responses.create(
            model=self.settings.model,
            input=prompt,
            temperature=self.settings.temperature,
            max_output_tokens=self.settings.max_output_tokens,
            response_format={"type": "json_object"},
        )

        response_text = getattr(response, "output_text", None)
        if response_text is None:
            response_text = self._concatenate_response_content(response)

        if not response_text:
            raise json.JSONDecodeError("empty response", "", 0)

        return json.loads(response_text)

    @staticmethod
    def _concatenate_response_content(response: Any) -> str:
        """
        Extract text from an OpenAI Responses API payload.
        """
        chunks: List[str] = []
        output = getattr(response, "output", None)
        if isinstance(output, list):
            for item in output:
                content = getattr(item, "content", None)
                if isinstance(content, list):
                    for part in content:
                        if getattr(part, "type", None) in ("text", "message"):
                            text = getattr(part, "text", None)
                            if isinstance(text, str):
                                chunks.append(text)
                        elif getattr(part, "type", None) == "tool_call":
                            args = getattr(part, "input", None)
                            if isinstance(args, str):
                                chunks.append(args)
        return "".join(chunks)

    # --------------------------------------------------------------------- #
    # Payload helpers
    # --------------------------------------------------------------------- #

    def _build_base_payload(self, parsed_document: Dict[str, Any]) -> Dict[str, Any]:
        document_code = parsed_document.get("document_code") or self._derive_document_code(parsed_document)
        ingestion_id = parsed_document.get("ingestion_id") or f"{document_code}-ingestion"
        language = parsed_document.get("language", "und")

        base_payload = {
            "document_code": document_code,
            "ingestion_id": ingestion_id,
            "extraction_version": self.settings.version,
            "datasource_code": self.settings.datasource_code,
            "extractor": {
                "model": self.settings.model,
                "prompt_id": self.settings.prompt_id,
                "temperature": self.settings.temperature,
                "max_tokens": self.settings.max_output_tokens,
            },
            "allowed_types": self.allowed_types,
            "entities": [],
            "relations": [],
            "provenance": {
                "overall_confidence": 0.0,
                "sections_used": [],
                "notes": f"Extraction run on {datetime.utcnow().isoformat()}Z (doc language={language})",
            },
            "quality": {
                "json_valid": False,
                "entities_count": 0,
                "relations_count": 0,
                "warnings": [],
            },
        }

        return base_payload

    def _build_prompt(self, parsed_document: Dict[str, Any]) -> Tuple[str, List[str]]:
        metadata_lines = [
            f"Document code: {parsed_document.get('document_code', 'N/A')}",
            f"Title: {parsed_document.get('title', parsed_document.get('document_title', 'N/A'))}",
            f"Language: {parsed_document.get('language', 'und')}",
            f"Page count: {parsed_document.get('page_count', 'N/A')}",
        ]

        sections = parsed_document.get("sections", []) or []
        section_snippets, section_ids = self._collect_section_snippets(sections)

        allowed_types_str = ", ".join(self.allowed_types)

        instructions = (
            "You are a knowledge extraction system. Extract entities and relations that match "
            "the provided allowed types. Only create entries when the information is explicitly "
            "supported by the context. Follow these rules:\n"
            "1. Output MUST be valid JSON matching the specified schema (no extra commentary).\n"
            "2. Use section_id references from the provided context.\n"
            "3. For numerical values, prefer floats. Omit fields when information is missing.\n"
            "4. Provide confidence scores between 0 and 1.\n"
            "5. Include provenance.sections_used with section ids you actually used.\n"
        )

        schema_expectations = (
            "JSON schema outline:\n"
            "{\n"
            '  "document_code": string,\n'
            '  "ingestion_id": string,\n'
            '  "extraction_version": string,\n'
            '  "datasource_code": string,\n'
            '  "extractor": { "model": string, "prompt_id": string, "temperature": number, "max_tokens": integer },\n'
            f'  "allowed_types": [{allowed_types_str}],\n'
            '  "entities": [ { "id": string, "type": string, "name": string, ... } ],\n'
            '  "relations": [ { "type": string, "from_ref": string, "to_ref": string, ... } ],\n'
            '  "provenance": { "overall_confidence": number, "sections_used": [string], "notes": string },\n'
            '  "quality": { "json_valid": boolean, "entities_count": integer, "relations_count": integer, "warnings": [string] }\n'
            "} (quality will be recalculated automatically, you may omit it)\n"
        )

        prompt = (
            f"{instructions}\n"
            f"{schema_expectations}\n"
            f"Allowed entity types: {allowed_types_str}\n\n"
            "Document metadata:\n"
            + "\n".join(f"- {line}" for line in metadata_lines)
            + "\n\nContext sections:\n"
            + "\n".join(section_snippets)
            + "\n\nRespond with JSON only."
        )

        return prompt, section_ids

    def _collect_section_snippets(self, sections: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
        snippets: List[str] = []
        section_ids: List[str] = []
        total_chars = 0

        for section in sections:
            text = (section.get("text") or "").strip()
            if not text:
                continue

            section_id = section.get("section_id") or section.get("id") or f"sec-{len(section_ids)+1}"
            snippet = text[: self.settings.max_chars_per_section]

            total_chars += len(snippet)
            if total_chars > self.settings.max_total_chars:
                logger.info("Reached prompt character budget; truncating remaining sections.")
                break

            title = section.get("title") or section.get("title_normalized") or "Untitled section"
            page_span = f"pages {section.get('page_start', '?')}-{section.get('page_end', '?')}"
            snippets.append(
                f"[Section {section_id} | {title} | {page_span}]:\n{snippet}\n"
            )
            section_ids.append(section_id)

            if len(section_ids) >= self.settings.max_sections:
                logger.info("Reached maximum number of sections (%s) for prompt.", self.settings.max_sections)
                break

        if not snippets:
            snippets.append("[No usable sections found in parsed document]\n")

        return snippets, section_ids

    @staticmethod
    def _ensure_entity_list(entities: Any) -> List[Dict[str, Any]]:
        if not isinstance(entities, list):
            return []

        cleaned: List[Dict[str, Any]] = []
        for idx, entity in enumerate(entities, start=1):
            if not isinstance(entity, dict):
                continue
            entity_id = entity.get("id") or f"ent-{idx:04d}"
            entity_type = entity.get("type") or "Unknown"
            name = entity.get("name") or entity_type
            spans = entity.get("spans") if isinstance(entity.get("spans"), list) else []
            cleaned.append(
                {
                    "id": entity_id,
                    "type": entity_type,
                    "name": name,
                    "aliases": entity.get("aliases", []),
                    "notes": entity.get("notes", ""),
                    "spans": spans,
                    "confidence": float(entity.get("confidence", 0.0)),
                    "ofType_ref": entity.get("ofType_ref"),
                    "unit_raw": entity.get("unit_raw"),
                    "nominal_value": entity.get("nominal_value"),
                    "min_value": entity.get("min_value"),
                    "max_value": entity.get("max_value"),
                    "tolerance": entity.get("tolerance"),
                    "machine_mode_ref": entity.get("machine_mode_ref"),
                }
            )
        return cleaned

    @staticmethod
    def _ensure_relation_list(relations: Any) -> List[Dict[str, Any]]:
        if not isinstance(relations, list):
            return []

        cleaned: List[Dict[str, Any]] = []
        for relation in relations:
            if not isinstance(relation, dict):
                continue
            relation_type = relation.get("type") or "relatedTo"
            from_ref = relation.get("from_ref")
            to_ref = relation.get("to_ref")
            if not (from_ref and to_ref):
                continue

            spans = relation.get("spans") if isinstance(relation.get("spans"), list) else []
            cleaned.append(
                {
                    "type": relation_type,
                    "from_ref": from_ref,
                    "to_ref": to_ref,
                    "spans": spans,
                    "confidence": float(relation.get("confidence", 0.0)),
                }
            )
        return cleaned

    def _validate_payload(self, payload: Dict[str, Any]) -> List[str]:
        warnings: List[str] = []

        allowed_types = payload.get("allowed_types", [])
        if not allowed_types:
            warnings.append("allowed_types is empty.")

        for entity in payload.get("entities", []):
            entity_type = entity.get("type")
            if allowed_types and entity_type not in allowed_types:
                warnings.append(f"Entity {entity.get('id')} has unexpected type '{entity_type}'.")

        relation_refs = {entity["id"] for entity in payload.get("entities", [])}
        for relation in payload.get("relations", []):
            if relation.get("from_ref") not in relation_refs:
                warnings.append(
                    f"Relation '{relation.get('type')}' references missing entity {relation.get('from_ref')}."
                )
            if relation.get("to_ref") not in relation_refs:
                warnings.append(
                    f"Relation '{relation.get('type')}' references missing entity {relation.get('to_ref')}."
                )

        return warnings

    def _build_json_schema(self) -> Dict[str, Any]:
        """
        Builds a lightweight JSON schema to validate essential fields.
        """
        return {
            "type": "object",
            "required": [
                "document_code",
                "ingestion_id",
                "extraction_version",
                "datasource_code",
                "extractor",
                "allowed_types",
                "entities",
                "relations",
                "provenance",
                "quality",
            ],
            "properties": {
                "document_code": {"type": "string"},
                "ingestion_id": {"type": "string"},
                "extraction_version": {"type": "string"},
                "datasource_code": {"type": "string"},
                "extractor": {"type": "object"},
                "allowed_types": {"type": "array"},
                "entities": {"type": "array"},
                "relations": {"type": "array"},
                "provenance": {"type": "object"},
                "quality": {"type": "object"},
            },
        }

    @staticmethod
    def _derive_document_code(parsed_document: Dict[str, Any]) -> str:
        filename = parsed_document.get("source", {}).get("filename")
        if filename:
            return Path(filename).stem
        return f"doc-{datetime.now().strftime('%Y%m%d%H%M%S')}"


def main() -> None:  # pragma: no cover - convenience CLI
    base_dir = Path(__file__).resolve().parent.parent
    schema_path = base_dir / "schemas" / "neural_extraction.json"
    parsed_dir = base_dir / "output" / "test_parsed"
    output_dir = base_dir / "output" / "neural_extraction"

    if not parsed_dir.exists():
        raise FileNotFoundError(f"Parsed documents directory not found: {parsed_dir}")

    parsed_documents: List[Dict[str, Any]] = []
    for json_file in parsed_dir.glob("*.json"):
        with open(json_file, "r", encoding="utf-8") as f:
            parsed_documents.append(json.load(f))

    extractor = NeuralExtractor(schema_path=str(schema_path))
    results = extractor.run_batch(parsed_documents=parsed_documents, output_dir=str(output_dir))

    if results:
        extractor.save_batch(results, str(output_dir / "neural_batch.json"))
        print(f"✅ Neural extraction complete for {len(results)} document(s).")
    else:
        print("⚠️  No neural extraction results produced.")


if __name__ == "__main__":  # pragma: no cover
    main()
