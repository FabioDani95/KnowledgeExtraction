from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from uuid import uuid4

from PyPDF2 import PdfReader

from ..base_pipeline import BaseSymbolicParser


class OperationModesSymbolicParser(BaseSymbolicParser):
    """Symbolic parser for operation and machine mode manuals."""

    def __init__(
        self,
        root_dir: Path,
        pipeline_config: dict,
        global_config: dict,
    ) -> None:
        super().__init__(root_dir, "operation_modes", pipeline_config, global_config)

        self.symbolic_config = pipeline_config.get("symbolic", {}) or {}
        self.skip_patterns = self._compile_skip_patterns(
            self.symbolic_config.get("skip_header_footer_patterns", {})
        )

        pdf_processing = global_config.get("pdf_processing", {}) or {}
        self.default_language = pdf_processing.get("default_language", "en")
        self.datasource_code = pdf_processing.get("datasource_code")

        page_classification = self.symbolic_config.get("page_classification", {}) or {}
        status_cfg = page_classification.get("status_keywords", {}) or {}
        self.status_must_include = [
            kw.lower() for kw in status_cfg.get("must_include", [])
        ]
        mode_cfg = page_classification.get("mode_keywords", {}) or {}
        self.mode_must_include = [kw.lower() for kw in mode_cfg.get("must_include", [])]
        proc_pattern = page_classification.get("procedure_pattern", r"^\s*\d+\)")
        self.procedure_pattern = re.compile(proc_pattern)
        self.min_procedure_steps = int(page_classification.get("min_procedure_steps", 2))

        status_table_cfg = self.symbolic_config.get("status_table", {}) or {}
        self.status_headers = status_table_cfg.get(
            "csv_headers", ["Status", "LED_Signal"]
        )
        self.status_icon_tokens = set(status_table_cfg.get("icon_tokens", []))
        self.status_tag = status_table_cfg.get("tag", "status_indicators")
        self.status_format = status_table_cfg.get(
            "format", "icon_table_simplified"
        )

        mode_table_cfg = self.symbolic_config.get("mode_table", {}) or {}
        self.mode_header_keywords = [
            kw.lower() for kw in mode_table_cfg.get("header_keywords", [])
        ]
        self.mode_bullet_tokens = mode_table_cfg.get("bullet_tokens", ["•", "-", "–"])
        self.mode_csv_format = mode_table_cfg.get(
            "csv_format", "flattened_bullets"
        )
        self.mode_tag = mode_table_cfg.get("tag", "machine_modes")

        procedure_cfg = self.symbolic_config.get("procedure", {}) or {}
        proc_step_pattern = procedure_cfg.get("step_pattern", r"^\s*(\d+)\)\s+(.+)")
        self.proc_step_pattern = re.compile(proc_step_pattern)
        self.procedure_tag = procedure_cfg.get("tag", "procedure")
        self.procedure_type = procedure_cfg.get("procedure_type", "step_by_step")
        self.procedure_figure_type = procedure_cfg.get("figure_type", "diagram")
        self.procedure_sync_figures = bool(
            procedure_cfg.get("synchronized_figures", True)
        )

        generic_cfg = self.symbolic_config.get("generic", {}) or {}
        self.generic_tag = generic_cfg.get("tag")

        table_detection = self.symbolic_config.get("table_detection", {}) or {}
        self.dotted_min_rows = int(table_detection.get("min_consistent_rows", 2))
        self.dotted_max_separator_variance = int(
            table_detection.get("max_separator_variance", 2)
        )

        self.logger = logging.getLogger("pipeline.operation_modes.symbolic")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def parse_document(self, document_path: Path) -> dict:
        reader = PdfReader(str(document_path))
        page_count = len(reader.pages)
        pdf_config = self.global_config.get("pdf_processing", {}) or {}
        skip_start = max(int(pdf_config.get("skip_start_pages", 0)), 0)
        skip_end = max(int(pdf_config.get("skip_end_pages", 0)), 0)
        processed_end = max(page_count - skip_end, skip_start)

        sections: List[dict] = []
        discarded_blocks: List[dict] = []
        total_chars = 0
        discarded_chars = 0
        running_offset = 0
        total_tables = 0
        total_figures = 0
        covered_chars = 0

        for idx, page in enumerate(reader.pages):
            if idx < skip_start or idx >= processed_end:
                continue

            raw_text = page.extract_text() or ""
            raw_text = raw_text.replace("\r", "")
            total_chars += len(raw_text)

            page_lines, page_discarded, discarded_len = self._filter_page_lines(
                raw_text.split("\n"),
                idx + 1,
            )
            if page_discarded:
                discarded_blocks.extend(page_discarded)
            discarded_chars += discarded_len

            page_text = "\n".join(page_lines).strip()
            if not page_text:
                continue

            classification = self._classify_page(page_lines, page_text)
            section = self._build_section(
                page_number=idx + 1,
                classification=classification,
                lines=page_lines,
                text=page_text,
                char_start=running_offset,
            )
            running_offset = section["char_end"] + 1

            sections.append(section)
            covered_chars += len(section["text"])
            total_tables += len(section.get("tables") or [])
            total_figures += len(section.get("figures") or [])

        denominator = max(total_chars - discarded_chars, 1)
        text_coverage_ratio = covered_chars / denominator if denominator else 0.0

        quality = {
            "text_coverage_ratio": round(text_coverage_ratio, 4),
            "tables_extracted": total_tables,
            "figures_detected": total_figures,
            "notes": "",
        }

        result = {
            "document_code": self._make_document_code(document_path),
            "title": document_path.stem,
            "datasource_code": self.datasource_code,
            "language": self.default_language,
            "ingestion_id": str(uuid4()),
            "created_utc": datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "source": self._build_source_metadata(document_path, page_count),
            "metadata": {
                "product_hint": None,
                "model_hint": None,
                "brand_hint": None,
                "doc_type": None,
                "version": None,
                "year": None,
            },
            "sections": sections,
            "extraction_constraints": self._extraction_constraints(),
            "quality": quality,
            "discarded_blocks": discarded_blocks,
        }

        return result

    # ------------------------------------------------------------------ #
    # Page filtering & classification
    # ------------------------------------------------------------------ #
    def _filter_page_lines(
        self,
        raw_lines: Iterable[str],
        page_number: int,
    ) -> Tuple[List[str], List[dict], int]:
        filtered: List[str] = []
        discarded: List[dict] = []
        discarded_chars = 0

        for raw_line in raw_lines:
            clean_line = raw_line.strip()
            reason = self._match_discard(raw_line, clean_line)
            if reason:
                discarded.append(
                    {
                        "reason": reason,
                        "text": clean_line,
                        "page": page_number,
                    }
                )
                discarded_chars += len(raw_line) + 1
                continue
            if clean_line:
                filtered.append(clean_line)

        return filtered, discarded, discarded_chars

    def _match_discard(self, raw_line: str, clean_line: str) -> Optional[str]:
        for reason, pattern in self.skip_patterns.items():
            target = raw_line if reason == "decoration" else clean_line
            if pattern.match(target):
                return reason
        return None

    def _classify_page(self, lines: Sequence[str], text: str) -> str:
        normalized_text = self._normalize_for_match(text)
        if self._contains_all(normalized_text, self.status_must_include):
            return "status_table"
        if self._contains_all(normalized_text, self.mode_must_include):
            return "mode_table"
        step_matches = sum(1 for line in lines if self.procedure_pattern.match(line))
        if step_matches >= self.min_procedure_steps:
            return "procedure"
        return "generic"

    @staticmethod
    def _contains_all(text: str, keywords: Sequence[str]) -> bool:
        if not keywords:
            return False
        return all(keyword in text for keyword in keywords)

    # ------------------------------------------------------------------ #
    # Section builders
    # ------------------------------------------------------------------ #
    def _build_section(
        self,
        page_number: int,
        classification: str,
        lines: Sequence[str],
        text: str,
        char_start: int,
    ) -> dict:
        section_id = self._make_section_id(classification, page_number)
        title = self._derive_title(classification, lines, page_number)
        tags: List[str] = []
        role = "main"
        tables: List[dict] = []
        figures: List[dict] = []
        extra_fields: Dict[str, object] = {}

        if classification == "status_table":
            role = "table"
            tables = self._build_status_tables(section_id, page_number, lines)
            tags.append(self.status_tag)
        elif classification == "mode_table":
            role = "table"
            tables = self._build_mode_tables(section_id, page_number, lines)
            tags.append(self.mode_tag)
        elif classification == "procedure":
            role = "procedure"
            steps = self._extract_procedure_steps(lines)
            extra_fields["steps"] = steps
            extra_fields["procedure_type"] = self.procedure_type
            tags.append(self.procedure_tag)
            if self.procedure_sync_figures:
                figures.append(
                    {
                        "figure_id": f"fig_{section_id}",
                        "page": page_number,
                        "caption": title,
                        "type": self.procedure_figure_type,
                        "synchronized_with_steps": True,
                    }
                )
        else:
            if self.generic_tag:
                tags.append(self.generic_tag)

        section = {
            "section_id": section_id,
            "parent_id": None,
            "title": title,
            "title_normalized": self._normalize_title(title),
            "role": role,
            "level": 1,
            "language": self.default_language,
            "language_confidence": 1.0,
            "page_start": page_number,
            "page_end": page_number,
            "char_start": char_start,
            "char_end": char_start + len(text),
            "text": text,
            "is_suspect": False,
            "tables": tables,
            "figures": figures,
            "tags": sorted(set(tags)),
        }
        section.update(extra_fields)
        return section

    def _build_status_tables(
        self,
        section_id: str,
        page_number: int,
        lines: Sequence[str],
    ) -> List[dict]:
        description_regex = re.compile(
            r"(blinking|steady|backlight|flash|alternately)", re.IGNORECASE
        )
        rows: List[List[str]] = []
        current_status: Optional[str] = None
        current_description: List[str] = []
        pending_status: Optional[str] = None

        for line in lines:
            line_normalized = self._normalize_for_match(line)
            if (
                "service manual" in line_normalized
                or "machine status" in line_normalized
                or "led signal" in line_normalized
            ):
                continue

            if self._icon_only(line):
                continue

            match = description_regex.search(line)
            if match:
                status_candidate = line[: match.start()].strip(" ,:")
                desc_part = line[match.start():].strip()

                if current_status and current_description:
                    rows.append(
                        [
                            current_status.strip(),
                            " ".join(current_description).strip(),
                        ]
                    )
                    current_description = []

                if pending_status:
                    current_status = pending_status
                    pending_status = None
                elif status_candidate:
                    current_status = status_candidate
                elif current_status is None:
                    current_status = line.strip()

                if current_status:
                    entry = desc_part if desc_part else line[match.start():].strip()
                    current_description = [entry] if entry else []
                continue

            if current_status:
                current_description.append(line.strip())
            else:
                pending_status = line.strip()

        if current_status and current_description:
            rows.append(
                [
                    current_status.strip(),
                    " ".join(current_description).strip(),
                ]
            )

        if not rows:
            return []

        csv_lines = [",".join(self.status_headers)]
        for status, desc in rows:
            csv_lines.append(self._csv_escape(status) + "," + self._csv_escape(desc))

        return [
            {
                "table_id": f"tbl_{section_id}",
                "page": page_number,
                "caption": self.status_headers[0],
                "n_rows": len(rows) + 1,
                "n_cols": 2,
                "csv": "\n".join(csv_lines),
                "format": self.status_format,
            }
        ]

    def _build_mode_tables(
        self,
        section_id: str,
        page_number: int,
        lines: Sequence[str],
    ) -> List[dict]:
        header: Optional[List[str]] = None
        rows: List[List[str]] = []
        current_row: Optional[List[str]] = None
        header_keywords = self.mode_header_keywords

        for line in lines:
            columns = self._split_columns(line)
            if not columns:
                continue

            lower_columns = [self._normalize_for_match(col) for col in columns]
            if not header:
                if header_keywords:
                    if any(
                        keyword in " ".join(lower_columns) for keyword in header_keywords
                    ):
                        header = columns
                        continue
                else:
                    header = columns
                    continue

            if header and len(columns) >= len(header):
                current_row = columns[: len(header)]
                while len(current_row) < len(header):
                    current_row.append("")
                rows.append(current_row)
            elif current_row is not None:
                continuation = line.strip()
                if continuation:
                    current_row[-1] = f"{current_row[-1]} {continuation}".strip()

        if not header or not rows:
            return []

        normalized_rows = [
            [self._flatten_bullets(cell) for cell in row] for row in rows
        ]

        csv_lines = [",".join(self._csv_escape(cell) for cell in header)]
        for row in normalized_rows:
            csv_lines.append(",".join(self._csv_escape(cell) for cell in row))

        return [
            {
                "table_id": f"tbl_{section_id}",
                "page": page_number,
                "caption": header[0],
                "n_rows": len(rows) + 1,
                "n_cols": len(header),
                "csv": "\n".join(csv_lines),
                "format": self.mode_csv_format,
            }
        ]

    def _extract_procedure_steps(self, lines: Sequence[str]) -> List[dict]:
        steps: List[dict] = []
        current: Optional[dict] = None

        for line in lines:
            match = self.proc_step_pattern.match(line)
            if match:
                if current:
                    steps.append(current)
                current = {
                    "number": int(match.group(1)),
                    "text": match.group(2).strip(),
                }
            elif current and line.strip():
                current["text"] = f"{current['text']} {line.strip()}".strip()

        if current:
            steps.append(current)

        return steps

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _compile_skip_patterns(self, patterns: Dict[str, str]) -> Dict[str, re.Pattern]:
        return {
            reason: re.compile(pattern, re.IGNORECASE)
            for reason, pattern in (patterns or {}).items()
            if pattern
        }

    def _split_columns(self, line: str) -> List[str]:
        normalized = line.replace("\u00a0", " ")
        parts = [part for part in re.split(r"\s{2,}", normalized.strip()) if part]
        if len(parts) <= 1 and ":" in line:
            head, tail = normalized.split(":", 1)
            parts = [head.strip(), tail.strip()]
        return parts

    def _icon_only(self, text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return False
        return all(char in self.status_icon_tokens for char in stripped)

    def _flatten_bullets(self, text: str) -> str:
        if not text:
            return text
        normalized = text.replace("\n", " ").replace("\u00a0", " ")
        for token in self.mode_bullet_tokens:
            normalized = re.sub(
                rf"\s*{re.escape(token)}\s*", ", ", normalized
            )
        normalized = re.sub(r"\s{2,}", " ", normalized)
        normalized = re.sub(r",\s*,", ", ", normalized)
        normalized = normalized.strip(" ,")
        return normalized

    def _derive_title(
        self,
        classification: str,
        lines: Sequence[str],
        page_number: int,
    ) -> str:
        if classification == "status_table":
            for line in lines:
                line_norm = self._normalize_for_match(line)
                if "machine status" in line_norm:
                    return line.strip()
            return f"Machine status indicators (page {page_number})"
        if classification == "mode_table":
            for line in lines:
                line_norm = self._normalize_for_match(line)
                if any(keyword in line_norm for keyword in self.mode_header_keywords):
                    return line.strip()
            return f"Machine modes overview (page {page_number})"
        if classification == "procedure":
            for line in lines:
                if not self.procedure_pattern.match(line) and line.strip():
                    return line.strip()
            return f"Procedure (page {page_number})"
        return f"Operational content (page {page_number})"

    def _make_section_id(self, classification: str, page_number: int) -> str:
        slug = classification.replace(" ", "_")
        return f"sec_{slug}_p{page_number}"

    def _csv_escape(self, value: str) -> str:
        if value is None:
            return ""
        if any(ch in value for ch in [",", '"', "\n"]):
            return '"' + value.replace('"', '""') + '"'
        return value

    def _build_source_metadata(self, document_path: Path, page_count: int) -> dict:
        checksum = hashlib.sha256(document_path.read_bytes()).hexdigest()
        return {
            "uri": str(document_path),
            "filename": document_path.name,
            "mime_type": "application/pdf",
            "checksum_sha256": checksum,
            "page_count": page_count,
        }

    def _extraction_constraints(self) -> dict:
        structural = (
            self.global_config.get("filtering", {})
            .get("structural_filtering", {})
            or {}
        )
        return {
            "skip_patterns": list(
                self.symbolic_config.get("skip_header_footer_patterns", {}).values()
            ),
            "max_chars_per_section": structural.get("max_chars_per_section"),
            "allowed_entities": self.symbolic_config.get("allowed_entities", []),
            "allowed_relations": self.symbolic_config.get("allowed_relations", []),
        }

    def _make_document_code(self, document_path: Path) -> str:
        slug = re.sub(r"[^A-Za-z0-9]+", "_", document_path.stem).strip("_")
        return slug.upper() or "DOCUMENT"

    def _normalize_title(self, title: str) -> str:
        text = title.lower()
        text = re.sub(r"[^\w\s-]", "", text)
        text = text.replace("-", " ")
        text = re.sub(r"\s+", "_", text)
        return re.sub(r"_+", "_", text).strip("_")

    @staticmethod
    def _normalize_for_match(value: str) -> str:
        return value.replace("\u00a0", " ").replace("\u2011", "-").lower()
