"""
Neural Extractor Module (hardened)
- Enforces JSON-only outputs with the OpenAI Responses API
- Validates/repairs per-chunk JSON
- Normalizes/deduplicates entities and renumbers IDs
- Adds conservative autolinking for core relations
- Computes provenance confidence and quality metrics
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
import re
from textwrap import dedent
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

try:
    from jsonschema import validate, ValidationError

    _JSONSCHEMA_AVAILABLE = True
except ModuleNotFoundError:
    validate = None  # type: ignore[assignment]
    ValidationError = Exception  # type: ignore[assignment]
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


# ----------------------------- Settings ------------------------------------ #
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
    max_retries: int = 2
    use_json_mode: bool = True  # request JSON object from API


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

    # ------------------------- Initialisation helpers ---------------------- #
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
            max_retries=int(self.module_config.get("max_retries", 2)),
            use_json_mode=bool(self.module_config.get("use_json_mode", True)),
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

    # ------------------------------ Public API ----------------------------- #
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

    # --------------------------- Core extraction --------------------------- #
    def extract_document(self, parsed_document: Dict[str, Any]) -> Dict[str, Any]:
        base_payload = self._build_base_payload(parsed_document)
        prompt_batches = self._build_prompt_batches(parsed_document)

        warnings: List[str] = []
        entities: List[Dict[str, Any]] = []
        relations: List[Dict[str, Any]] = []
        sections_used: Set[str] = set()
        overall_confidences: List[float] = []
        json_valid = True

        if not prompt_batches:
            warnings.append("No sections available for neural extraction.")
            json_valid = False

        if self._client is None:
            warnings.append("LLM client not initialised; skipping extraction.")
            json_valid = False

        for batch_index, batch in enumerate(prompt_batches, start=1):
            sections_used.update(batch["section_ids"])

            if self._client is None:
                break

            try:
                chunk_data = self._invoke_model_with_retries(batch["prompt"], batch_index)
            except json.JSONDecodeError as exc:
                json_valid = False
                warnings.append(f"Chunk {batch_index}: response was not valid JSON ({exc}).")
                logger.warning("Chunk %s JSON decoding failed: %s", batch_index, exc)
                continue
            except OpenAIAPIError as exc:
                json_valid = False
                warnings.append(f"Chunk {batch_index}: OpenAI API error: {exc}")
                logger.error("OpenAI API error during neural extraction chunk %s: %s", batch_index, exc)
                continue
            except Exception as exc:  # pragma: no cover - defensive
                json_valid = False
                warnings.append(f"Chunk {batch_index}: unexpected error: {exc}")
                logger.error("Unexpected neural extraction error for chunk %s: %s", batch_index, exc)
                continue

            if not isinstance(chunk_data, dict):
                json_valid = False
                warnings.append(f"Chunk {batch_index}: response not a JSON object.")
                continue

            # Lightweight per-chunk schema
            chunk_warnings = self._validate_chunk(chunk_data)
            warnings.extend([f"Chunk {batch_index}: {w}" for w in chunk_warnings])

            chunk_entities = self._ensure_entity_list(chunk_data.get("entities"))
            chunk_relations = self._ensure_relation_list(chunk_data.get("relations"))
            chunk_provenance = chunk_data.get("provenance", {})

            conf = chunk_provenance.get("overall_confidence") if isinstance(chunk_provenance, dict) else None
            if conf is not None:
                try:
                    overall_confidences.append(float(conf))
                except (TypeError, ValueError):
                    warnings.append(
                        f"Chunk {batch_index}: could not parse overall_confidence '{conf}'."
                    )
            if isinstance(chunk_provenance, dict):
                sections_used.update(chunk_provenance.get("sections_used", []))

            entities.extend(chunk_entities)
            relations.extend(chunk_relations)

        # --- Post-processing: dedup, normalize, autolink, renumber IDs --- #
        entities, dedup_warnings, id_map = self._deduplicate_and_normalize_entities(entities)
        warnings.extend(dedup_warnings)

        relations, rel_warnings = self._normalize_relations(relations, id_map)
        warnings.extend(rel_warnings)

        auto_rel, auto_warn = self._autolink_relations(entities)
        relations.extend(auto_rel)
        warnings.extend(auto_warn)

        # Renumber relations with REL-xxxx
        relations = self._renumber_relations(relations)

        # Provenance confidence: fallback to mean entity conf if chunk confs missing
        if overall_confidences:
            base_payload["provenance"]["overall_confidence"] = sum(overall_confidences) / len(overall_confidences)
        else:
            ent_confs = [float(e.get("confidence", 0.0)) for e in entities if isinstance(e.get("confidence"), (int, float, str))]
            try:
                ent_confs = [float(x) for x in ent_confs]
            except Exception:
                ent_confs = []
            base_payload["provenance"]["overall_confidence"] = sum(ent_confs) / len(ent_confs) if ent_confs else 0.0

        base_payload["entities"] = entities
        base_payload["relations"] = relations
        base_payload["provenance"]["sections_used"] = sorted(sections_used)

        # Final schema validation (non-blocking)
        if _JSONSCHEMA_AVAILABLE and isinstance(self.schema_template, dict):
            try:
                validate(instance=base_payload, schema=self._build_json_schema())
            except ValidationError as exc:
                logger.warning("Schema validation failed: %s", exc)
                warnings.append(
                    f"Schema validation warning: {getattr(exc, 'message', str(exc))}"
                )

        base_payload["quality"] = {
            "json_valid": json_valid,
            "entities_count": len(base_payload["entities"]),
            "relations_count": len(base_payload["relations"]),
            "warnings": warnings,
        }

        return base_payload

    # --------------------------- Model invocation -------------------------- #
    def _invoke_model_with_retries(self, prompt: str, batch_index: int) -> Dict[str, Any]:
        assert self._client is not None
        last_err: Optional[Exception] = None
        suffix_retry = dedent(
            """
            IMPORTANT: Respond with a single JSON object only. No prose, no markdown fences, no explanations.
            Root keys must be exactly: entities (array), relations (array), provenance (object).
            """
        ).strip()

        for attempt in range(1, self.settings.max_retries + 2):  # first try + retries
            try:
                response = self._client.responses.create(
                    model=self.settings.model,
                    input=(prompt if attempt == 1 else f"{prompt}\n\n{suffix_retry}"),
                    temperature=self.settings.temperature,
                    max_output_tokens=self.settings.max_output_tokens,
                    **({"response_format": {"type": "json_object"}} if self.settings.use_json_mode else {}),
                )

                response_text = getattr(response, "output_text", None)
                if response_text is None:
                    response_text = self._concatenate_response_content(response)

                if not response_text:
                    raise json.JSONDecodeError("empty response", "", 0)

                return self._parse_json_response(response_text)
            except Exception as exc:  # keep trying
                last_err = exc
                logger.debug("Chunk %s attempt %s failed: %s", batch_index, attempt, exc)

        # After retries
        if isinstance(last_err, json.JSONDecodeError):
            raise last_err
        raise json.JSONDecodeError(f"Unable to parse LLM response as JSON after retries: {last_err}", "", 0)

    @staticmethod
    def _concatenate_response_content(response: Any) -> str:
        """Extract text from an OpenAI Responses API payload."""
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

    @staticmethod
    def _yield_json_candidates(text: str) -> Iterable[str]:
        stripped = text.strip()
        if stripped:
            yield stripped
        # Code fences ```json ... ```
        fence_pattern = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
        for match in fence_pattern.finditer(text):
            block = match.group(1).strip()
            if block:
                yield block
        # Balanced braces
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            yield text[first_brace : last_brace + 1].strip()

    @staticmethod
    def _sanitize_json_candidate(candidate: str) -> str:
        cleaned = candidate.strip()
        cleaned = cleaned.lstrip("\ufeff")
        cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
        cleaned = re.sub(r"//.*", "", cleaned)
        cleaned = re.sub(r"/\*.*?\*/", "", cleaned, flags=re.DOTALL)
        cleaned = re.sub(r"\bTrue\b", "true", cleaned)
        cleaned = re.sub(r"\bFalse\b", "false", cleaned)
        cleaned = re.sub(r"\bNone\b", "null", cleaned)
        cleaned = re.sub(r",(\s*[}\]])", r"\1", cleaned)
        cleaned = re.sub(r",\s*,+", ",", cleaned)
        return cleaned

    @staticmethod
    def _parse_json_response(text: str) -> Dict[str, Any]:
        for candidate in NeuralExtractor._yield_json_candidates(text):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass
            sanitized = NeuralExtractor._sanitize_json_candidate(candidate)
            if sanitized != candidate:
                try:
                    return json.loads(sanitized)
                except json.JSONDecodeError:
                    continue
        first_candidate = next(iter(NeuralExtractor._yield_json_candidates(text)), text.strip())
        raise json.JSONDecodeError("Unable to parse LLM response as JSON", first_candidate, 0)

    # ------------------------- Chunk & payload QA -------------------------- #
    def _validate_chunk(self, chunk: Dict[str, Any]) -> List[str]:
        warnings: List[str] = []
        if not isinstance(chunk.get("entities"), list):
            warnings.append("entities missing or not a list; coerced to empty.")
            chunk["entities"] = []
        if not isinstance(chunk.get("relations"), list):
            warnings.append("relations missing or not a list; coerced to empty.")
            chunk["relations"] = []
        prov = chunk.get("provenance", {})
        if not isinstance(prov, dict):
            warnings.append("provenance missing or not an object; coerced.")
            chunk["provenance"] = {}
        return warnings

    @staticmethod
    def _ensure_entity_list(entities: Any) -> List[Dict[str, Any]]:
        if not isinstance(entities, list):
            return []
        cleaned: List[Dict[str, Any]] = []
        for idx, entity in enumerate(entities, start=1):
            if not isinstance(entity, dict):
                continue
            entity_id = str(entity.get("id") or f"ENT-TEMP-{idx:04d}")
            entity_type = str(entity.get("type") or "Unknown")
            name = str(entity.get("name") or entity_type)
            spans = entity.get("spans") if isinstance(entity.get("spans"), list) else []
            try:
                conf = float(entity.get("confidence", 0.0))
            except Exception:
                conf = 0.0
            cleaned.append(
                {
                    "id": entity_id,
                    "type": entity_type,
                    "name": name,
                    "aliases": entity.get("aliases", []),
                    "notes": entity.get("notes", ""),
                    "spans": spans,
                    "confidence": conf,
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
            relation_type = str(relation.get("type") or "relatedTo")
            from_ref = relation.get("from_ref")
            to_ref = relation.get("to_ref")
            if not (from_ref and to_ref):
                continue
            spans = relation.get("spans") if isinstance(relation.get("spans"), list) else []
            try:
                conf = float(relation.get("confidence", 0.0))
            except Exception:
                conf = 0.0
            cleaned.append(
                {
                    "type": relation_type,
                    "from_ref": str(from_ref),
                    "to_ref": str(to_ref),
                    "spans": spans,
                    "confidence": conf,
                }
            )
        return cleaned

    def _validate_payload(self, payload: Dict[str, Any]) -> List[str]:
        warnings: List[str] = []
        allowed_types = payload.get("allowed_types", [])
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

    # ------------------------ Post-processing utils ------------------------ #
    @staticmethod
    def _norm_name(text: str) -> str:
        t = text.strip().lower()
        t = re.sub(r"[^a-z0-9]+", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    def _deduplicate_and_normalize_entities(self, entities: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str], Dict[str, str]]:
        warnings: List[str] = []
        id_map: Dict[str, str] = {}
        bucket: Dict[Tuple[str, str], Dict[str, Any]] = {}
        order: List[Tuple[str, str]] = []

        for e in entities:
            orig_id = str(e.get("id") or "")
            e_type = str(e.get("type") or "Unknown")
            name = str(e.get("name") or e_type)
            key = (e_type, self._norm_name(name))
            if key not in bucket:
                bucket[key] = {
                    **e,
                    "aliases": list({*([name] + e.get("aliases", []))}),
                    "spans": e.get("spans", [])[:],
                }
                order.append(key)
                # Assign canonical ID placeholder; real IDs assigned later
                id_map[orig_id] = orig_id  # temporary
            else:
                # merge
                existing = bucket[key]
                # keep higher confidence
                try:
                    if float(e.get("confidence", 0.0)) > float(existing.get("confidence", 0.0)):
                        existing["confidence"] = float(e.get("confidence", 0.0))
                except Exception:
                    pass
                # merge aliases
                existing_aliases = set(existing.get("aliases", []))
                for a in e.get("aliases", []):
                    existing_aliases.add(a)
                existing["aliases"] = sorted(existing_aliases)
                # merge spans
                ex_spans = existing.get("spans", []) or []
                new_spans = e.get("spans", []) or []
                existing["spans"] = ex_spans + new_spans
                warnings.append(f"Deduplicated entity '{name}' of type '{e_type}'.")
                id_map[orig_id] = existing.get("id", orig_id)

        # Assign final sequential IDs ENT-0001 ...
        normalized_entities: List[Dict[str, Any]] = []
        final_id_map: Dict[str, str] = {}
        for idx, key in enumerate(order, start=1):
            ent = bucket[key]
            new_id = f"ENT-{idx:04d}"
            final_id_map[ent.get("id", new_id)] = new_id
            ent["id"] = new_id
            normalized_entities.append(ent)

        # remap temporary ids to final ids
        for k, v in list(id_map.items()):
            final = final_id_map.get(v, v)
            id_map[k] = final

        # enforce allowed types only
        filtered_entities: List[Dict[str, Any]] = []
        for ent in normalized_entities:
            if self.allowed_types and ent.get("type") not in self.allowed_types:
                warnings.append(f"Dropped entity '{ent.get('name')}' with unsupported type '{ent.get('type')}'.")
                continue
            filtered_entities.append(ent)

        return filtered_entities, warnings, id_map

    def _normalize_relations(self, relations: List[Dict[str, Any]], id_map: Dict[str, str]) -> Tuple[List[Dict[str, Any]], List[str]]:
        warnings: List[str] = []
        norm: List[Dict[str, Any]] = []
        for r in relations:
            fr = id_map.get(r.get("from_ref"), r.get("from_ref"))
            to = id_map.get(r.get("to_ref"), r.get("to_ref"))
            if not fr or not to:
                warnings.append("Skipped relation with missing endpoints after ID remap.")
                continue
            norm.append({**r, "from_ref": fr, "to_ref": to})
        return norm, warnings

    def _renumber_relations(self, relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for idx, r in enumerate(relations, start=1):
            rr = dict(r)
            rr["id"] = f"REL-{idx:04d}"
            out.append(rr)
        return out

    # ---------------------------- Autolinking ------------------------------ #
    def _autolink_relations(self, entities: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
        warnings: List[str] = []
        rels: List[Dict[str, Any]] = []

        # Index by type
        by_type: Dict[str, List[Dict[str, Any]]] = {}
        for e in entities:
            by_type.setdefault(e.get("type", "Unknown"), []).append(e)

        products = by_type.get("Product", [])
        params = by_type.get("ParameterSpec", [])
        units = by_type.get("Unit", [])
        failures = by_type.get("FailureMode", [])
        maint = by_type.get("MaintenanceTask", [])
        repairs = by_type.get("RepairAction", [])
        tests = by_type.get("TestSpec", [])
        components = by_type.get("Component", [])
        comptypes = by_type.get("ComponentType", [])

        # Helper: find unit by keyword
        def guess_unit_for_param(name: str) -> Optional[Dict[str, Any]]:
            n = self._norm_name(name)
            mapping = {
                "temperature": ["c", "f", "°c", "°f"],
                "volume": ["ml", "l"],
                "torque": ["n m", "nm", "n·m"],
            }
            for unit in units:
                u = self._norm_name(unit.get("name", ""))
                if any(tag in n for tag in mapping.get("temperature", [])) and ("c" in u or "f" in u):
                    return unit
                if "volume" in n and ("ml" in u or "l" in u):
                    return unit
                if "torque" in n and ("nm" in u or "n m" in u or "n·m" in u):
                    return unit
            # no direct unit found; return None
            return None

        # Product → ParameterSpec
        for p in products:
            for ps in params:
                rels.append({"type": "HAS_PARAMETER", "from_ref": p["id"], "to_ref": ps["id"], "confidence": 0.8})

        # ParameterSpec → Unit (if obvious)
        for ps in params:
            # prefer explicit unit_raw first
            unit_explicit = None
            if ps.get("unit_raw"):
                for u in units:
                    if self._norm_name(u.get("name", "")) in self._norm_name(ps.get("unit_raw", "")):
                        unit_explicit = u
                        break
            unit_guess = unit_explicit or guess_unit_for_param(ps.get("name", ""))
            if unit_guess:
                rels.append({"type": "MEASURED_IN_UNIT", "from_ref": ps["id"], "to_ref": unit_guess["id"], "confidence": 0.9})

        # Product → FailureMode
        for p in products:
            for f in failures:
                rels.append({"type": "HAS_FAILURE_MODE", "from_ref": p["id"], "to_ref": f["id"], "confidence": 0.75})

        # FailureMode → RepairAction or MaintenanceTask (choose best available)
        def looks_like_repair(name: str) -> bool:
            return bool(re.search(r"\b(replace|repair|solder|swap|fix)\b", name.lower()))

        for f in failures:
            linked = False
            for r in repairs:
                if looks_like_repair(r.get("name", "")):
                    rels.append({"type": "MITIGATED_BY", "from_ref": f["id"], "to_ref": r["id"], "confidence": 0.8})
                    linked = True
                    break
            if not linked:
                for m in maint:
                    if looks_like_repair(m.get("name", "")) or re.search(r"\bclean|descal|inspect\b", m.get("name", "").lower()):
                        rels.append({"type": "MITIGATED_BY", "from_ref": f["id"], "to_ref": m["id"], "confidence": 0.65})
                        break

        # Component → ComponentType (OF_TYPE) if names overlap
        def name_overlap(a: str, b: str) -> bool:
            A = set(self._norm_name(a).split())
            B = set(self._norm_name(b).split())
            return len(A & B) >= 1

        for c in components:
            for ct in comptypes:
                if name_overlap(c.get("name", ""), ct.get("name", "")):
                    rels.append({"type": "OF_TYPE", "from_ref": c["id"], "to_ref": ct["id"], "confidence": 0.8})
                    break

        # De-duplicate relations (type, from, to)
        seen: Set[Tuple[str, str, str]] = set()
        uniq: List[Dict[str, Any]] = []
        for r in rels:
            key = (r["type"], r["from_ref"], r["to_ref"])
            if key in seen:
                continue
            seen.add(key)
            uniq.append(r)

        if not uniq:
            warnings.append("Autolinking produced no relations (check entity coverage).")

        return uniq, warnings

    # --------------------------- Payload builders -------------------------- #
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

    def _build_prompt_batches(self, parsed_document: Dict[str, Any]) -> List[Dict[str, Any]]:
        sections = parsed_document.get("sections", []) or []
        section_batches = self._chunk_sections(sections)

        prompts: List[Dict[str, Any]] = []
        total_batches = max(len(section_batches), 1)

        for idx, batch in enumerate(section_batches, start=1):
            prompt = self._render_prompt(parsed_document, batch, idx, total_batches)
            section_ids = [item["section_id"] for item in batch]
            prompts.append({"prompt": prompt, "section_ids": section_ids})

        return prompts

    def _chunk_sections(self, sections: List[Dict[str, Any]]) -> List[List[Dict[str, str]]]:
        batches: List[List[Dict[str, str]]] = []
        current_batch: List[Dict[str, str]] = []
        current_chars = 0

        for section in sections:
            text = (section.get("text") or "").strip()
            if not text:
                continue

            snippet = text[: self.settings.max_chars_per_section]
            if not snippet:
                continue

            section_id = section.get("section_id") or section.get("id") or f"sec-{len(current_batch)+1}"
            title = section.get("title") or section.get("title_normalized") or "Untitled section"
            page_span = f"pages {section.get('page_start', '?')}-{section.get('page_end', '?')}"
            snippet_len = len(snippet)

            exceed_char_limit = current_batch and (current_chars + snippet_len > self.settings.max_total_chars)
            exceed_section_limit = current_batch and len(current_batch) >= self.settings.max_sections

            if exceed_char_limit or exceed_section_limit:
                batches.append(current_batch)
                current_batch = []
                current_chars = 0

            current_batch.append(
                {
                    "section_id": section_id,
                    "title": title,
                    "page_span": page_span,
                    "snippet": snippet,
                }
            )
            current_chars += snippet_len

        if current_batch:
            batches.append(current_batch)

        if not batches:
            batches.append([])

        return batches

    def _render_prompt(
        self,
        parsed_document: Dict[str, Any],
        batch: List[Dict[str, str]],
        chunk_index: int,
        chunk_total: int,
    ) -> str:
        metadata_lines = [
            f"Document code: {parsed_document.get('document_code', 'N/A')}",
            f"Title: {parsed_document.get('title', parsed_document.get('document_title', 'N/A'))}",
            f"Language: {parsed_document.get('language', 'und')}",
            f"Page count: {parsed_document.get('page_count', 'N/A')}",
        ]

        allowed_types_str = ", ".join(self.allowed_types)

        instructions = dedent(
            """
            You are a knowledge extraction assistant specialized in technical documentation.
            Analyze only the sections provided in this chunk and extract structured entities and relations that conform to the schema.

            RESPONSE REQUIREMENTS:
              1. Output strict JSON without commentary.
              2. The root object must contain 'entities' (array), 'relations' (array), and 'provenance' (object).
              3. Use provided section_id values in spans where possible.
              4. Provide confidence scores between 0 and 1.
              5. Do not reference sections outside this chunk.
            """
        ).strip()

        sections_text: List[str] = []
        for item in batch:
            sections_text.append(
                f"[Chunk {chunk_index}/{chunk_total} | Section {item['section_id']} | {item['title']} | {item['page_span']}]\n{item['snippet']}\n"
            )

        if not sections_text:
            sections_text.append("[No usable sections provided in this chunk]\n")

        prompt = (
            f"{instructions}\n\n"
            f"Allowed entity types: {allowed_types_str}\n"
            f"This is chunk {chunk_index} of {chunk_total}.\n\n"
            "Document metadata:\n"
            + "\n".join(f"- {line}" for line in metadata_lines)
            + "\n\nChunk context:\n"
            + "\n".join(sections_text)
            + "\nRespond with JSON."
        )
        return prompt

    # ------------------------------ Utilities ------------------------------ #
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
