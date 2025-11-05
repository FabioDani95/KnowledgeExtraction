from __future__ import annotations

import csv
import hashlib
import logging
import re
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from uuid import uuid4

from PyPDF2 import PdfReader

from ..base_pipeline import BaseSymbolicParser


class TestingSymbolicParser(BaseSymbolicParser):
    """
    Symbolic parser for QA / testing manuals.

    Produces a parsed-document JSON that complies with schemas/parsed_document.json.
    Sections are derived from heading patterns or page boundaries, and basic heuristics
    tag blocks that mention test outcomes (pass/fail) or requirements.
    """

    def __init__(
        self,
        root_dir: Path,
        pipeline_config: dict,
        global_config: dict,
    ) -> None:
        super().__init__(root_dir, "testing", pipeline_config, global_config)

        self.symbolic_config = pipeline_config.get("symbolic", {}) or {}
        pdf_processing = global_config.get("pdf_processing", {}) or {}
        self.default_language = pdf_processing.get("default_language", "en")
        self.datasource_code = (
            pdf_processing.get("datasource_code")
            or self.symbolic_config.get("datasource_code")
            or "SRC_TESTING"
        )

        self.skip_patterns = self._compile_skip_patterns(
            self.symbolic_config.get("skip_header_footer_patterns", {})
        )
        self.section_patterns = self._compile_section_patterns(
            self.symbolic_config.get("section_heading_patterns", [])
        )
        self.min_section_chars = int(self.symbolic_config.get("min_section_chars", 25))
        self.numeric_heading_pattern = re.compile(r"^\d+(?:\.\d+){0,3}\s+.+")

        keyword_cfg = self.symbolic_config.get("test_case_keywords", {}) or {}
        self.testcase_keywords = {
            kw.lower().strip()
            for kws in keyword_cfg.values()
            for kw in (kws or [])
            if kw
        } or {"test", "pass", "fail", "requirement"}

        self.default_role = self.symbolic_config.get("default_role", "main")

        block_cfg = self.symbolic_config.get("block_markers", {}) or {}
        self.block_marker_map = self._build_block_marker_map(block_cfg)
        self.safety_markers = [
            marker.lower()
            for marker in self.symbolic_config.get(
                "safety_markers", ["danger", "warning", "caution", "note", "notice"]
            )
        ]
        self.measurement_markers = [
            marker.lower()
            for marker in block_cfg.get(
                "measurement_markers",
                ["measurement table", "measurement", "diagram", "graph", "ntc"],
            )
        ]
        self.applicability_markers = [
            marker.lower()
            for marker in block_cfg.get(
                "applicability_markers",
                ["only necessary", "for class", "for models", "after a repair"],
            )
        ]

        self.step_patterns = [
            re.compile(pattern)
            for pattern in self.symbolic_config.get(
                "step_markers_regex", [r"^\d+\)", r"^\d+\.\s", r"^[•\-–]\s"]
            )
        ]
        acceptance_phrases = self.symbolic_config.get(
            "acceptance_patterns",
            [
                "must be",
                "should be",
                "shall be",
                "at least",
                "lower than",
                "less than",
                "greater than",
                "higher than",
                "between",
                "±",
            ],
        )
        self.acceptance_phrases = [phrase.lower() for phrase in acceptance_phrases]

        unit_tokens = self.symbolic_config.get(
            "unit_patterns",
            [
                "bar",
                "ml",
                "°c",
                "c",
                "°f",
                "f",
                "ohm",
                "kohm",
                "kω",
                "ω",
                "v dc",
                "vdc",
                "ma",
                "sec",
                "s",
            ],
        )
        unit_regex = "|".join(sorted({re.escape(token) for token in unit_tokens}, key=len, reverse=True))
        self.measurement_value_pattern = re.compile(
            rf"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>{unit_regex})\b", re.IGNORECASE
        )
        self.measurement_range_pattern = re.compile(
            rf"(?P<min>\d+(?:\.\d+)?)\s*(?:–|-|to)\s*(?P<max>\d+(?:\.\d+)?)\s*(?P<unit>{unit_regex})\b",
            re.IGNORECASE,
        )
        self.measurement_tolerance_pattern = re.compile(
            rf"(?P<center>\d+(?:\.\d+)?)\s*(?P<unit>{unit_regex})\s*[±\+/-]+\s*(?P<tolerance>\d+(?:\.\d+)?)",
            re.IGNORECASE,
        )
        self.measurement_comparison_pattern = re.compile(
            rf"(?P<operator>>=|<=|≥|≤|>|<)\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>{unit_regex})\b",
            re.IGNORECASE,
        )

        self.cross_ref_pattern = re.compile(r"refer to page\s+(?P<page>\d+)", re.IGNORECASE)
        self.figure_pattern = re.compile(r"^(figure|graph|diagram)\s+\d+", re.IGNORECASE)
        self.multi_column_pattern = re.compile(r"\S+\s{2,}\S+")

        self.logger = logging.getLogger("pipeline.testing.symbolic")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def _build_block_marker_map(self, block_cfg: Dict[str, Sequence[str]]) -> Dict[str, dict]:
        defaults = [
            {
                "config_key": "precondition_markers",
                "markers": ["prerequisite", "preparations"],
                "role": "main",
                "tags": {"precondition"},
                "level": 2,
            },
            {
                "config_key": "procedure_markers",
                "markers": ["test run", "test sequence", "procedure"],
                "role": "main",
                "tags": {"procedure"},
                "level": 2,
            },
            {
                "config_key": "equipment_markers",
                "markers": ["test equipment", "tools", "equipment"],
                "role": "main",
                "tags": {"equipment"},
                "level": 2,
            },
            {
                "config_key": "measurement_markers",
                "markers": ["measurement table", "measurement", "diagram", "graph"],
                "role": "table",
                "tags": {"measurement"},
                "level": 2,
            },
            {
                "config_key": "applicability_markers",
                "markers": ["additional tests", "applicability", "variants"],
                "role": "main",
                "tags": {"variant"},
                "level": 2,
            },
            {
                "config_key": "general_markers",
                "markers": ["description", "general"],
                "role": "main",
                "tags": set(),
                "level": 2,
            },
            {
                "config_key": "safety_markers",
                "markers": ["danger", "warning", "caution", "note", "notice"],
                "role": "warning",
                "tags": {"safety"},
                "level": 2,
            },
        ]

        mapping: Dict[str, dict] = {}
        for spec in defaults:
            markers = block_cfg.get(spec["config_key"], spec["markers"])
            role = block_cfg.get(f"{spec['config_key']}_role", spec["role"])
            tags = set(block_cfg.get(f"{spec['config_key']}_tags", spec["tags"]))
            level = int(block_cfg.get(f"{spec['config_key']}_level", spec["level"]))
            for marker in markers or []:
                marker_key = marker.lower().strip().rstrip(":")
                if not marker_key:
                    continue
                mapping[marker_key] = {
                    "role": role,
                    "tags": tags.copy(),
                    "level": max(1, min(level, 6)),
                }
        return mapping

    def parse_document(self, document_path: Path) -> dict:
        reader = PdfReader(str(document_path))
        page_count = len(reader.pages)
        pdf_cfg = self.global_config.get("pdf_processing", {}) or {}
        skip_start = max(int(pdf_cfg.get("skip_start_pages", 0)), 0)
        skip_end = max(int(pdf_cfg.get("skip_end_pages", 0)), 0)
        processed_end = max(page_count - skip_end, skip_start)

        sections: List[dict] = []
        discarded_blocks: List[dict] = []
        total_chars = 0
        discarded_chars = 0
        covered_chars = 0
        running_offset = 0

        section_counter = 0
        current_section: Optional[dict] = None

        def flush_current_section() -> None:
            nonlocal current_section, covered_chars, sections
            if not current_section:
                return

            text_lines: List[str] = current_section.pop("_text_lines", [])
            text = "\n".join(text_lines).strip()
            if not text:
                current_section = None
                return

            self._finalize_section(current_section, text)
            covered_chars += len(text)
            sections.append(current_section)
            current_section = None

        for page_idx, page in enumerate(reader.pages):
            if page_idx < skip_start or page_idx >= processed_end:
                continue

            raw_text = page.extract_text() or ""
            raw_text = raw_text.replace("\r", "")
            total_chars += len(raw_text)

            lines = raw_text.split("\n")
            for raw_line in lines:
                normalized_line = self._normalize_line(raw_line)
                clean_line = normalized_line.strip()
                if not clean_line:
                    running_offset += 1
                    continue

                skip_reason = self._match_skip_pattern(clean_line)
                if skip_reason:
                    discarded_blocks.append(
                        {
                            "reason": skip_reason,
                            "text": clean_line,
                            "page": page_idx + 1,
                        }
                    )
                    discarded_chars += len(clean_line)
                    running_offset += len(clean_line) + 1
                    continue

                signal = self._detect_section_signal(clean_line)
                if signal is not None:
                    flush_current_section()
                    section_counter += 1
                    current_section = self._start_section(
                        section_index=section_counter,
                        title=signal["title"],
                        level=signal["level"],
                        page_number=page_idx + 1,
                        char_start=running_offset,
                        role_override=signal.get("role"),
                        initial_tags=signal.get("tags"),
                    )
                    continue

                if current_section is None:
                    section_counter += 1
                    current_section = self._start_section(
                        section_index=section_counter,
                        title=f"Section {section_counter}",
                        level=1,
                        page_number=page_idx + 1,
                        char_start=running_offset,
                    )

                self._append_line(current_section, clean_line, page_idx + 1)
                running_offset += len(clean_line) + 1

        flush_current_section()

        if not sections:
            # Fallback: treat entire document as a single section
            entire_text = []
            for page_idx, page in enumerate(reader.pages):
                if page_idx < skip_start or page_idx >= processed_end:
                    continue
                page_text = (page.extract_text() or "").strip()
                if page_text:
                    entire_text.append(page_text)
            combined = "\n\n".join(entire_text).strip()
            if combined:
                section = self._start_section(
                    section_index=1,
                    title=document_path.stem,
                    level=1,
                    page_number=skip_start + 1 if skip_start < page_count else 1,
                    char_start=0,
                )
                section["_text_lines"] = [combined]
                section["page_end"] = processed_end
                current_section = section
                flush_current_section()

        quality = self._build_quality(sections, total_chars, discarded_chars, covered_chars)
        promoted_entities, promoted_relations = self._promote_structured_entities(sections)
        metadata = self._infer_metadata(sections, document_path)

        result = {
            "document_code": self._make_document_code(document_path),
            "title": document_path.stem,
            "datasource_code": self.datasource_code,
            "language": self.default_language,
            "ingestion_id": str(uuid4()),
            "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "source": self._build_source_metadata(document_path, page_count),
            "metadata": metadata,
            "sections": sections,
            "extraction_constraints": self._extraction_constraints(),
            "quality": quality,
            "discarded_blocks": discarded_blocks,
            "promoted_entities": promoted_entities,
            "promoted_relations": promoted_relations,
        }
        return result

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _compile_skip_patterns(
        self, pattern_map: Dict[str, str]
    ) -> List[Tuple[str, re.Pattern]]:
        compiled: List[Tuple[str, re.Pattern]] = []
        for label, pattern in (pattern_map or {}).items():
            if not pattern:
                continue
            compiled.append((label, re.compile(pattern, re.IGNORECASE)))
        return compiled

    def _compile_section_patterns(
        self, patterns: Sequence[str]
    ) -> List[re.Pattern]:
        if not patterns:
            patterns = [
                r"^\d+(?:\.\d+){0,3}\s+.+",
                r"^(Test|Case|Scenario)\s+\d+",
                r"^([A-Z][A-Za-z0-9\s]+):$",
            ]
        return [re.compile(p, re.IGNORECASE) for p in patterns]

    def _match_skip_pattern(self, line: str) -> str:
        for label, pattern in self.skip_patterns:
            if pattern.search(line):
                if label in {"toc", "header", "footer", "non_en", "too_short", "pattern_match"}:
                    return label
                return "pattern_match"
        return ""

    def _normalize_line(self, line: str) -> str:
        normalized = unicodedata.normalize("NFKC", line)
        normalized = normalized.replace("\u00a0", " ").replace("\u2011", "-").replace("\xad", "")
        return normalized

    def _detect_section_signal(self, line: str) -> Optional[dict]:
        if not line:
            return None

        candidate = line.rstrip(":").strip()
        lowered = candidate.lower()

        marker_info = self.block_marker_map.get(lowered)
        if marker_info:
            return {
                "title": candidate,
                "level": marker_info["level"],
                "role": marker_info["role"],
                "tags": marker_info["tags"].copy(),
            }

        if self.numeric_heading_pattern.match(candidate):
            number_token = candidate.split()[0]
            depth = number_token.count(".") + 1
            return {
                "title": candidate,
                "level": max(1, min(depth, 6)),
                "role": "main",
                "tags": set(),
            }

        for pattern in self.section_patterns:
            if pattern.match(candidate):
                return {
                    "title": candidate,
                    "level": 2,
                    "role": "main",
                    "tags": set(),
                }

        return None

    def _start_section(
        self,
        section_index: int,
        title: str,
        level: int,
        page_number: int,
        char_start: int,
        role_override: Optional[str] = None,
        initial_tags: Optional[set] = None,
    ) -> dict:
        tags: set[str] = set(initial_tags or set())
        if self._contains_test_keyword(title):
            tags.add("test_case")
        normalized_role = (role_override or self.default_role or "main").lower()
        if normalized_role not in {"main", "warning", "toc", "suspicious", "measurement"}:
            normalized_role = "main"
        section = {
            "section_id": f"sec_{section_index:03d}",
            "parent_id": None,
            "title": title.rstrip(":"),
            "title_normalized": self._normalize_title(title),
            "role": normalized_role,
            "level": level,
            "language": self.default_language,
            "language_confidence": 1.0,
            "page_start": page_number,
            "page_end": page_number,
            "char_start": char_start,
            "char_end": char_start,
            "is_suspect": False,
            "tables": [],
            "figures": [],
            "tags": tags,
            "measurements": [],
            "acceptance_criteria": [],
            "steps": [],
            "applicability": [],
            "cross_references": [],
            "_text_lines": [],
            "_table_buffers": [],
            "_active_table": None,
            "_measurement_signatures": set(),
            "_acceptance_seen": set(),
            "_applicability_seen": set(),
            "_cross_ref_seen": set(),
            "_figure_seen": set(),
            "_steps_seen": set(),
        }
        return section

    def _append_line(self, section: dict, text_line: str, page_number: int) -> None:
        section["_text_lines"].append(text_line)
        section["page_end"] = max(section["page_end"], page_number)
        if self._contains_test_keyword(text_line):
            section["tags"].add("test_case")
        self._process_line_features(section, text_line, page_number)

    def _process_line_features(
        self,
        section: dict,
        text_line: str,
        page_number: int,
    ) -> None:
        line_lower = text_line.lower()

        # Extract measurements first so other logic can reuse them
        measurements = self._extract_measurements(text_line, page_number)
        for measurement in measurements:
            signature = (
                measurement.get("type"),
                measurement.get("value"),
                measurement.get("unit"),
                measurement.get("min"),
                measurement.get("max"),
                measurement.get("operator"),
                measurement.get("tolerance"),
            )
            if signature in section["_measurement_signatures"]:
                continue
            section["_measurement_signatures"].add(signature)
            section["measurements"].append(measurement)

        if measurements:
            section["tags"].add("measurement")
            if section["role"] == "main":
                section["role"] = "table"

        # Safety markers promote section role
        if any(
            line_lower.startswith(marker) or f"{marker} " in line_lower
            for marker in self.safety_markers
        ):
            section["role"] = "warning"
            section["tags"].add("safety")

        # Steps / bullet lists
        for pattern in self.step_patterns:
            if pattern.match(text_line):
                normalized_step = text_line.strip()
                if normalized_step not in section["_steps_seen"]:
                    section["_steps_seen"].add(normalized_step)
                    section["steps"].append(normalized_step)
                break

        # Applicability hints
        for marker in self.applicability_markers:
            if marker in line_lower:
                normalized_marker = text_line.strip()
                if normalized_marker not in section["_applicability_seen"]:
                    section["_applicability_seen"].add(normalized_marker)
                    section["applicability"].append(normalized_marker)
                    section["tags"].add("applicability")
                break

        # Acceptance criteria derived from measurements or keywords
        has_acceptance_phrase = any(
            phrase in line_lower for phrase in self.acceptance_phrases
        )
        if has_acceptance_phrase or any(m.get("operator") for m in measurements):
            acceptance_entries = self._build_acceptance_entries(
                text_line, measurements, page_number
            )
            if not acceptance_entries and has_acceptance_phrase:
                acceptance_entries.append(
                    {
                        "text": text_line.strip(),
                        "page": page_number,
                    }
                )

            for entry in acceptance_entries:
                key = (
                    entry.get("operator"),
                    entry.get("value"),
                    entry.get("min_value"),
                    entry.get("max_value"),
                    entry.get("tolerance"),
                    entry.get("unit"),
                    entry.get("text"),
                )
                if key in section["_acceptance_seen"]:
                    continue
                section["_acceptance_seen"].add(key)
                section["acceptance_criteria"].append(entry)
                section["tags"].add("acceptance")

        # Cross references
        for match in self.cross_ref_pattern.finditer(text_line):
            page_target = int(match.group("page"))
            signature = (page_target, text_line.strip())
            if signature not in section["_cross_ref_seen"]:
                section["_cross_ref_seen"].add(signature)
                section["cross_references"].append(
                    {"page": page_target, "text": text_line.strip()}
                )
                section["tags"].add("cross_reference")

        # Figures
        if self.figure_pattern.match(text_line):
            caption = text_line.strip()
            if caption not in section["_figure_seen"]:
                section["_figure_seen"].add(caption)
                figure_id = f"{section['section_id']}_fig_{len(section['figures']) + 1:02d}"
                section["figures"].append(
                    {"figure_id": figure_id, "page": page_number, "caption": caption}
                )

        # Tables / multi-column content
        self._handle_table_detection(section, text_line)

    def _extract_measurements(self, text_line: str, page_number: int) -> List[dict]:
        measurements: List[dict] = []
        line_lower = text_line.lower()
        word_operator = None

        if "between" in line_lower:
            word_operator = "between"
        elif "at least" in line_lower or "minimum" in line_lower:
            word_operator = ">="
        elif "at most" in line_lower:
            word_operator = "<="
        elif "less than" in line_lower or "lower than" in line_lower:
            word_operator = "<"
        elif "greater than" in line_lower or "higher than" in line_lower:
            word_operator = ">"
        elif "must be" in line_lower or "should be" in line_lower or "shall be" in line_lower:
            word_operator = "="

        for match in self.measurement_range_pattern.finditer(text_line):
            unit = self._normalize_unit(match.group("unit"))
            measurements.append(
                {
                    "type": "range",
                    "min": float(match.group("min")),
                    "max": float(match.group("max")),
                    "unit": unit,
                    "operator": "between",
                    "source_text": text_line.strip(),
                    "page": page_number,
                }
            )

        for match in self.measurement_tolerance_pattern.finditer(text_line):
            unit = self._normalize_unit(match.group("unit"))
            measurements.append(
                {
                    "type": "tolerance",
                    "value": float(match.group("center")),
                    "tolerance": float(match.group("tolerance")),
                    "unit": unit,
                    "operator": "±",
                    "source_text": text_line.strip(),
                    "page": page_number,
                }
            )

        for match in self.measurement_comparison_pattern.finditer(text_line):
            unit = self._normalize_unit(match.group("unit"))
            measurements.append(
                {
                    "type": "threshold",
                    "operator": match.group("operator"),
                    "value": float(match.group("value")),
                    "unit": unit,
                    "source_text": text_line.strip(),
                    "page": page_number,
                }
            )

        for match in self.measurement_value_pattern.finditer(text_line):
            unit = self._normalize_unit(match.group("unit"))
            measurements.append(
                {
                    "type": "value",
                    "operator": word_operator,
                    "value": float(match.group("value")),
                    "unit": unit,
                    "source_text": text_line.strip(),
                    "page": page_number,
                }
            )
        return measurements

    def _build_acceptance_entries(
        self,
        text_line: str,
        measurements: List[dict],
        page_number: int,
    ) -> List[dict]:
        entries: List[dict] = []
        for measurement in measurements:
            entries.append(
                {
                    "text": text_line.strip(),
                    "page": page_number,
                    "operator": measurement.get("operator"),
                    "value": measurement.get("value"),
                    "min_value": measurement.get("min"),
                    "max_value": measurement.get("max"),
                    "tolerance": measurement.get("tolerance"),
                    "unit": measurement.get("unit"),
                    "source_text": measurement.get("source_text", text_line.strip()),
                }
            )
        return entries

    @staticmethod
    def _normalize_unit(unit: str) -> str:
        clean = unit.strip().replace(" ", "").lower()
        mapping = {
            "ml": "mL",
            "bar": "bar",
            "°c": "°C",
            "c": "°C",
            "°f": "°F",
            "f": "°F",
            "ohm": "Ω",
            "ω": "Ω",
            "kohm": "kΩ",
            "kω": "kΩ",
            "ma": "mA",
            "vdc": "V DC",
            "vd": "V DC",
            "vdcac": "V DC",
            "vdcv": "V DC",
        }
        # cleanup duplicates by fallback
        if clean in mapping:
            return mapping[clean]
        if clean.endswith("dc"):
            return "V DC"
        if clean in {"sec", "s"}:
            return "s"
        return unit.strip()

    def _handle_table_detection(self, section: dict, text_line: str) -> None:
        line_lower = text_line.lower()
        active_table = section.get("_active_table")

        if any(marker in line_lower for marker in self.measurement_markers):
            buffer = {"title": text_line.rstrip(":"), "rows": []}
            section["_table_buffers"].append(buffer)
            section["_active_table"] = buffer
            section["tags"].add("measurement")
            return

        if self.multi_column_pattern.search(text_line):
            if not active_table:
                buffer = {"title": None, "rows": [text_line]}
                section["_table_buffers"].append(buffer)
                section["_active_table"] = buffer
            else:
                active_table["rows"].append(text_line)
            section["tags"].add("measurement")
            return

        if active_table and (not text_line.strip() or not self.multi_column_pattern.search(text_line)):
            section["_active_table"] = None

    def _finalize_section(self, section: dict, text: str) -> None:
        section["text"] = text
        section["char_end"] = section["char_start"] + len(text)
        if len(text) < self.min_section_chars:
            section["is_suspect"] = True

        self._finalize_tables(section)
        section["tags"] = sorted(section.get("tags", set()))

        helper_keys = [
            "_text_lines",
            "_table_buffers",
            "_active_table",
            "_measurement_signatures",
            "_acceptance_seen",
            "_applicability_seen",
            "_cross_ref_seen",
            "_figure_seen",
            "_steps_seen",
        ]
        for key in helper_keys:
            section.pop(key, None)

    def _finalize_tables(self, section: dict) -> None:
        buffers = section.get("_table_buffers") or []
        table_index = len(section.get("tables") or [])
        for buffer in buffers:
            rows = buffer.get("rows") or []
            if not rows:
                continue
            table_index += 1
            csv_text = self._table_rows_to_csv(rows)
            columns = (
                max(len(self._split_table_row(row)) for row in rows) if rows else 0
            )
            section["tables"].append(
                {
                    "table_id": f"{section['section_id']}_tbl_{table_index:02d}",
                    "page": section["page_start"],
                    "caption": buffer.get("title"),
                    "n_rows": len(rows),
                    "n_cols": columns,
                    "csv": csv_text,
                }
            )

    def _split_table_row(self, row: str) -> List[str]:
        if "\t" in row:
            parts = [part.strip() for part in row.split("\t")]
        else:
            parts = [part.strip() for part in re.split(r"\s{2,}", row)]
        return [part for part in parts if part]

    def _table_rows_to_csv(self, rows: List[str]) -> str:
        buffer = StringIO()
        writer = csv.writer(buffer)
        for row in rows:
            writer.writerow(self._split_table_row(row))
        return buffer.getvalue().strip()

    def _promote_structured_entities(
        self,
        sections: List[dict]
    ) -> Tuple[List[dict], List[dict]]:
        promoted_entities: List[dict] = []
        promoted_relations: List[dict] = []
        seen_entity_ids: set = set()
        seen_relation_keys: set = set()
        testcase_by_section: Dict[str, str] = {}
        id_counters: defaultdict = defaultdict(int)

        def next_id(prefix: str, section_id: str) -> str:
            key = (prefix, section_id)
            id_counters[key] += 1
            return f"{prefix}_{section_id.upper()}_{id_counters[key]:02d}"

        def make_span(section: dict, source_text: Optional[str]) -> dict:
            span = {
                "section_id": section.get("section_id"),
                "page": section.get("page_start"),
            }
            if source_text:
                span["source_text"] = source_text
            return span

        def add_entity(entity: dict) -> None:
            if entity.get("id") in seen_entity_ids:
                return
            seen_entity_ids.add(entity.get("id"))
            promoted_entities.append(entity)

        def add_relation(relation: dict) -> None:
            key = (
                relation.get("type"),
                relation.get("from_ref"),
                relation.get("to_ref"),
            )
            if key in seen_relation_keys:
                return
            seen_relation_keys.add(key)
            promoted_relations.append(relation)

        for section in sections:
            sec_id = section.get("section_id")
            if not sec_id:
                continue

            tags = set(section.get("tags") or [])
            title = section.get("title") or f"Section {sec_id}"
            title_lower = title.lower()
            is_test_section = (
                "procedure" in tags
                or "test_case" in tags
                or "measurement" in tags
                or "test" in title_lower
            )

            test_case_id: Optional[str] = None
            if is_test_section:
                test_case_id = f"TESTCASE_{sec_id.upper()}"
                if sec_id not in testcase_by_section:
                    testcase_by_section[sec_id] = test_case_id
                    entity = {
                        "id": test_case_id,
                        "type": "TestCase",
                        "name": title,
                        "confidence": 0.9,
                        "spans": [make_span(section, title)],
                        "metadata": {
                            "section_role": section.get("role"),
                            "section_tags": sorted(tags),
                        },
                    }
                    add_entity(entity)
            else:
                test_case_id = testcase_by_section.get(sec_id)

            # Safety notices
            if "safety" in tags or section.get("role") == "warning":
                safety_id = f"SAFETY_{sec_id.upper()}"
                if safety_id not in seen_entity_ids:
                    entity = {
                        "id": safety_id,
                        "type": "SafetyNotice",
                        "name": f"Safety notice - {title}",
                        "confidence": 0.85,
                        "spans": [make_span(section, section.get("text", ""))],
                        "metadata": {
                            "section_id": sec_id,
                            "context": section.get("text", ""),
                        },
                    }
                    add_entity(entity)

            # Measurements
            for measurement in section.get("measurements", []):
                measurement_id = next_id("MEAS", sec_id)
                entity = {
                    "id": measurement_id,
                    "type": "Measurement",
                    "name": f"{title} measurement",
                    "confidence": 0.85,
                    "spans": [make_span(section, measurement.get("source_text"))],
                    "metadata": {
                        "section_id": sec_id,
                        "context": measurement.get("source_text"),
                        "operator": measurement.get("operator"),
                    },
                }
                if measurement.get("value") is not None:
                    entity["nominal_value"] = measurement.get("value")
                if measurement.get("min") is not None:
                    entity["min_value"] = measurement.get("min")
                if measurement.get("max") is not None:
                    entity["max_value"] = measurement.get("max")
                if measurement.get("tolerance") is not None:
                    entity["tolerance"] = measurement.get("tolerance")
                if measurement.get("unit"):
                    entity["unit_raw"] = measurement.get("unit")
                add_entity(entity)

                if test_case_id:
                    relation = {
                        "type": "validatedBy",
                        "from_ref": test_case_id,
                        "to_ref": measurement_id,
                        "confidence": 0.85,
                        "spans": [make_span(section, measurement.get("source_text"))],
                    }
                    add_relation(relation)

            # Acceptance criteria
            for criterion in section.get("acceptance_criteria", []):
                acceptance_id = next_id("AC", sec_id)
                entity = {
                    "id": acceptance_id,
                    "type": "AcceptanceCriterion",
                    "name": f"{title} criterion",
                    "confidence": 0.85,
                    "spans": [make_span(section, criterion.get("source_text", criterion.get("text")))],
                    "metadata": {
                        "operator": criterion.get("operator"),
                        "unit": criterion.get("unit"),
                        "value": criterion.get("value"),
                        "min_value": criterion.get("min_value"),
                        "max_value": criterion.get("max_value"),
                        "tolerance": criterion.get("tolerance"),
                        "section_id": sec_id,
                    },
                }
                add_entity(entity)

                if test_case_id:
                    relation = {
                        "type": "constrainedBy",
                        "from_ref": test_case_id,
                        "to_ref": acceptance_id,
                        "confidence": 0.85,
                        "spans": [make_span(section, criterion.get("source_text", criterion.get("text")))],
                    }
                    add_relation(relation)

            # Applicability clauses
            for clause in section.get("applicability", []):
                clause_id = next_id("APP", sec_id)
                entity = {
                    "id": clause_id,
                    "type": "ApplicabilityClause",
                    "name": clause,
                    "confidence": 0.8,
                    "spans": [make_span(section, clause)],
                    "metadata": {
                        "section_id": sec_id,
                    },
                }
                add_entity(entity)

                if test_case_id:
                    relation = {
                        "type": "appliesTo",
                        "from_ref": test_case_id,
                        "to_ref": clause_id,
                        "confidence": 0.8,
                        "spans": [make_span(section, clause)],
                    }
                    add_relation(relation)

            # Cross references
            for cross_ref in section.get("cross_references", []):
                ref_page = cross_ref.get("page")
                ref_text = cross_ref.get("text")
                ref_id = next_id("XREF", sec_id)
                entity = {
                    "id": ref_id,
                    "type": "CrossReference",
                    "name": f"Reference page {ref_page}",
                    "confidence": 0.75,
                    "spans": [make_span(section, ref_text)],
                    "metadata": {
                        "target_page": ref_page,
                        "section_id": sec_id,
                    },
                }
                add_entity(entity)

                if test_case_id:
                    relation = {
                        "type": "refersTo",
                        "from_ref": test_case_id,
                        "to_ref": ref_id,
                        "confidence": 0.75,
                        "spans": [make_span(section, ref_text)],
                    }
                    add_relation(relation)

        return promoted_entities, promoted_relations

    def _infer_metadata(self, sections: List[dict], document_path: Path) -> dict:
        metadata = self._build_metadata_stub()
        combined_text = "\n".join(section.get("text", "") for section in sections).lower()
        title_lower = document_path.stem.lower()

        if "nespresso" in combined_text or "nespresso" in title_lower:
            metadata["brand_hint"] = "Nespresso"
        elif "delonghi" in combined_text or "delonghi" in title_lower:
            metadata["brand_hint"] = "DeLonghi"

        if "citiz" in combined_text or "citiz" in title_lower:
            metadata["model_hint"] = "Citiz"

        if "service manual" in combined_text or "service-manual" in title_lower:
            metadata["doc_type"] = "service manual"
        elif "test" in combined_text and not metadata["doc_type"]:
            metadata["doc_type"] = "test manual"

        version_match = re.search(r"revision\s+([a-z0-9.]+)", combined_text)
        if version_match:
            metadata["version"] = version_match.group(1)

        year_match = re.search(r"(20\d{2})", title_lower)
        if year_match:
            metadata["year"] = int(year_match.group(1))

        return metadata

    def _contains_test_keyword(self, text: str) -> bool:
        lowered = text.lower()
        return any(keyword in lowered for keyword in self.testcase_keywords)

    def _normalize_title(self, title: str) -> str:
        text = title.lower()
        text = re.sub(r"[^\w\s-]", "", text)
        text = text.replace("-", " ")
        text = re.sub(r"\s+", "_", text)
        return re.sub(r"_+", "_", text).strip("_") or "section"

    def _make_document_code(self, document_path: Path) -> str:
        slug = re.sub(r"[^A-Za-z0-9]+", "_", document_path.stem).strip("_")
        return slug.upper() or "DOCUMENT"

    def _build_source_metadata(self, document_path: Path, page_count: int) -> dict:
        checksum = hashlib.sha256(document_path.read_bytes()).hexdigest()
        return {
            "uri": str(document_path),
            "filename": document_path.name,
            "mime_type": "application/pdf",
            "checksum_sha256": checksum,
            "page_count": page_count,
        }

    def _build_metadata_stub(self) -> dict:
        return {
            "product_hint": None,
            "model_hint": None,
            "brand_hint": None,
            "doc_type": None,
            "version": None,
            "year": None,
        }

    def _extraction_constraints(self) -> dict:
        structural = (
            self.global_config.get("filtering", {})
            .get("structural_filtering", {})
            or {}
        )
        return {
            "skip_patterns": [
                pattern.pattern for _, pattern in self.skip_patterns
            ],
            "max_chars_per_section": structural.get("max_chars_per_section"),
            "allowed_entities": self.symbolic_config.get("allowed_entities", []),
            "allowed_relations": self.symbolic_config.get("allowed_relations", []),
        }

    def _build_quality(
        self,
        sections: List[dict],
        total_chars: int,
        discarded_chars: int,
        covered_chars: int,
    ) -> dict:
        tables_count = sum(len(section.get("tables") or []) for section in sections)
        figures_count = sum(len(section.get("figures") or []) for section in sections)
        safety_blocks = sum(
            1 for section in sections if "safety" in (section.get("tags") or [])
        )
        steps_count = sum(len(section.get("steps") or []) for section in sections)
        acceptance_count = sum(
            len(section.get("acceptance_criteria") or []) for section in sections
        )
        measurement_count = sum(
            len(section.get("measurements") or []) for section in sections
        )
        applicability_count = sum(
            len(section.get("applicability") or []) for section in sections
        )

        denominator = max(total_chars - discarded_chars, 1)
        coverage_ratio = covered_chars / denominator if denominator else 0.0

        return {
            "text_coverage_ratio": round(coverage_ratio, 4),
            "tables_extracted": tables_count,
            "figures_detected": figures_count,
            "safety_blocks_count": safety_blocks,
            "procedural_steps_count": steps_count,
            "acceptance_criteria_count": acceptance_count,
            "measurements_extracted_count": measurement_count,
            "applicability_clauses_count": applicability_count,
            "notes": f"{len(sections)} sections extracted",
        }
