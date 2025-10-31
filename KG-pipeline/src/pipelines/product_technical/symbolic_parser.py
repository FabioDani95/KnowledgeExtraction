from __future__ import annotations

import csv
import hashlib
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Sequence
from uuid import uuid4

from PyPDF2 import PdfReader

from ..base_pipeline import BaseSymbolicParser


@dataclass
class LineInfo:
    text: str
    clean: str
    page: int
    start: int
    end: int


@dataclass
class SectionDraft:
    section_id: str
    title: str
    title_normalized: str
    level: int
    char_start: int
    char_end: int
    page_start: int
    page_end: int
    parent_id: Optional[str] = None
    lines: List[str] = field(default_factory=list)
    is_suspect: bool = False


class ProductTechnicalSymbolicParser(BaseSymbolicParser):
    """Symbolic parser for product & technical manuals."""

    def __init__(
        self,
        root_dir: Path,
        pipeline_config: dict,
        global_config: dict,
    ) -> None:
        super().__init__(root_dir, "product_technical", pipeline_config, global_config)
        self.symbolic_config = pipeline_config.get("symbolic", {}) or {}
        self.skip_patterns = self._compile_skip_patterns(
            self.symbolic_config.get("skip_header_footer_patterns", {})
        )
        self.section_patterns = self._compile_section_patterns(
            self.symbolic_config.get("section_patterns", {})
        )
        self.role_heuristics = self.symbolic_config.get("role_heuristics", {})
        self.tagging_keywords = self.symbolic_config.get("tagging_keywords", {})
        self.table_detection = self.symbolic_config.get("table_detection", {})
        self.quality_thresholds = self.symbolic_config.get("quality_thresholds", {})
        self.logger = logging.getLogger("pipeline.product_technical.symbolic")

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def parse_document(self, document_path: Path) -> dict:
        reader = PdfReader(str(document_path))
        page_count = len(reader.pages)
        pdf_config = self.global_config.get("pdf_processing", {}) or {}
        skip_start = max(int(pdf_config.get("skip_start_pages", 0)), 0)
        skip_end = max(int(pdf_config.get("skip_end_pages", 0)), 0)
        processed_end = max(page_count - skip_end, skip_start)

        lines: List[LineInfo] = []
        total_chars = 0
        offset = 0

        for idx, page in enumerate(reader.pages):
            if idx < skip_start or idx >= processed_end:
                continue

            raw_text = page.extract_text() or ""
            raw_text = raw_text.replace("\r", "")
            total_chars += len(raw_text)

            for raw_line in raw_text.split("\n"):
                clean_line = raw_line.strip()
                line_length = len(raw_line)
                line = LineInfo(
                    text=raw_line,
                    clean=clean_line,
                    page=idx + 1,
                    start=offset,
                    end=offset + line_length,
                )
                lines.append(line)
                offset += line_length + 1  # account for newline separation

        discarded_blocks: List[dict] = []
        discarded_chars = 0
        sections: List[SectionDraft] = []
        section_stack: List[SectionDraft] = []
        section_ids: Counter = Counter()

        for line in lines:
            discard_reason = self._match_discard(line)
            if discard_reason:
                discarded_blocks.append(
                    {"reason": discard_reason, "text": line.clean, "page": line.page}
                )
                discarded_chars += len(line.text) + 1
                continue

            heading_level = self._detect_heading_level(line.clean)
            if heading_level:
                self._close_stack(section_stack, heading_level, line.start, line.page)
                section = self._start_section(
                    line=line,
                    level=heading_level,
                    section_stack=section_stack,
                    section_ids=section_ids,
                )
                section_stack.append(section)
                sections.append(section)
                continue

            if not section_stack:
                fallback = self._create_fallback_section(
                    line=line, section_ids=section_ids
                )
                section_stack.append(fallback)
                sections.append(fallback)

            active_section = section_stack[-1]
            active_section.lines.append(line.text)
            active_section.char_end = line.end
            active_section.page_end = max(active_section.page_end, line.page)

        for section in section_stack:
            if section.char_end < section.char_start:
                section.char_end = section.char_start + len(section.title)

        output_sections: List[dict] = []
        covered_chars = 0
        total_tables = 0
        total_figures = 0

        for section in sections:
            text = "\n".join(section.lines).strip()
            figures = self._extract_figures(section, text)
            tables = self._extract_tables(section, text)
            tags = self._collect_tags(section, text, tables, figures)
            role = self._classify_role(section, text, tables, figures)

            covered_chars += len(text)

            section_dict = {
                "section_id": section.section_id,
                "parent_id": section.parent_id,
                "title": section.title,
                "title_normalized": section.title_normalized,
                "role": role,
                "level": section.level,
                "language": pdf_config.get("default_language", "en"),
                "language_confidence": 1.0,
                "page_start": section.page_start,
                "page_end": section.page_end,
                "char_start": section.char_start,
                "char_end": max(section.char_end, section.char_start),
                "text": text,
                "is_suspect": section.is_suspect,
                "tables": tables,
                "figures": figures,
                "tags": sorted(tags),
            }
            output_sections.append(section_dict)
            total_tables += len(tables)
            total_figures += len(figures)

        denominator = max(total_chars - discarded_chars, 1)
        text_coverage_ratio = covered_chars / denominator if denominator else 0.0

        quality = {
            "text_coverage_ratio": round(text_coverage_ratio, 4),
            "tables_extracted": total_tables,
            "figures_detected": total_figures,
            "notes": "",
        }

        min_coverage = self.quality_thresholds.get("min_text_coverage", 0.0)
        if text_coverage_ratio < min_coverage:
            quality["notes"] = (
                f"Text coverage below threshold ({text_coverage_ratio:.2f}"
                f" < {min_coverage:.2f})"
            )

        result = {
            "document_code": self._make_document_code(document_path),
            "title": document_path.stem,
            "datasource_code": pdf_config.get("datasource_code"),
            "language": pdf_config.get("default_language", "en"),
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
            "sections": output_sections,
            "extraction_constraints": self._extraction_constraints(),
            "quality": quality,
            "discarded_blocks": discarded_blocks,
        }

        return result

    # --------------------------------------------------------------------- #
    # Helpers - parsing stages
    # --------------------------------------------------------------------- #
    def _compile_skip_patterns(self, patterns: Dict[str, str]) -> Dict[str, re.Pattern]:
        compiled: Dict[str, re.Pattern] = {}
        for reason, pattern in (patterns or {}).items():
            if not pattern:
                continue
            compiled[reason] = re.compile(pattern, re.IGNORECASE)
        return compiled

    def _compile_section_patterns(self, patterns: Dict[str, str]) -> Dict[int, re.Pattern]:
        compiled: Dict[int, re.Pattern] = {}
        mapping = {1: "level1", 2: "level2", 3: "level3"}
        for level, key in mapping.items():
            value = patterns.get(key)
            if value:
                compiled[level] = re.compile(value)
        return compiled

    def _match_discard(self, line: LineInfo) -> Optional[str]:
        for reason, pattern in self.skip_patterns.items():
            target = line.clean
            if reason == "decoration":
                target = line.text
            if pattern.match(target):
                return reason
        return None

    def _detect_heading_level(self, text: str) -> Optional[int]:
        for level in (3, 2, 1):
            pattern = self.section_patterns.get(level)
            if pattern and pattern.match(text):
                return level
        return None

    def _close_stack(
        self,
        section_stack: List[SectionDraft],
        new_level: int,
        new_start: int,
        new_page: int,
    ) -> None:
        while len(section_stack) >= new_level:
            closing = section_stack.pop()
            if new_start > closing.char_start:
                closing.char_end = max(closing.char_end, new_start - 1)
            closing.page_end = max(closing.page_end, new_page)

    def _start_section(
        self,
        line: LineInfo,
        level: int,
        section_stack: List[SectionDraft],
        section_ids: Counter,
    ) -> SectionDraft:
        section_id = self._build_section_id(line.clean, section_ids)
        title_normalized = self._normalize_title(line.clean)
        parent_id = section_stack[-1].section_id if section_stack else None
        section = SectionDraft(
            section_id=section_id,
            title=line.clean,
            title_normalized=title_normalized,
            level=level,
            char_start=line.start,
            char_end=line.end,
            page_start=line.page,
            page_end=line.page,
            parent_id=parent_id,
        )
        return section

    def _create_fallback_section(
        self,
        line: LineInfo,
        section_ids: Counter,
    ) -> SectionDraft:
        section_id = self._build_section_id("0 PREFACE", section_ids)
        title_normalized = self._normalize_title("Preface")
        section = SectionDraft(
            section_id=section_id,
            title="Preface",
            title_normalized=title_normalized,
            level=1,
            char_start=line.start,
            char_end=line.end,
            page_start=line.page,
            page_end=line.page,
            parent_id=None,
        )
        return section

    # --------------------------------------------------------------------- #
    # Helpers - content analysis
    # --------------------------------------------------------------------- #
    def _extract_figures(self, section: SectionDraft, text: str) -> List[dict]:
        figure_keywords = [kw.lower() for kw in self.role_heuristics.get("figure_keywords", [])]
        legend_pattern = re.compile(r"^\s*\d+\)\s+.+", re.MULTILINE)
        text_lower = text.lower()

        has_keyword = any(kw in section.title.lower() or kw in text_lower for kw in figure_keywords)
        legend_items = legend_pattern.findall(text)

        if not has_keyword and not legend_items:
            return []

        figure_type = "diagram"
        for kw in ("photo", "image", "picture"):
            if kw in text_lower:
                figure_type = "photo"
        if "schematic" in text_lower:
            figure_type = "schematic"

        figure_id = f"fig_{section.section_id.replace('sec_', '')}"
        figure = {
            "figure_id": figure_id,
            "page": section.page_start,
            "caption": section.title,
            "type": figure_type,
            "bbox": None,
            "has_legend": bool(legend_items),
            "legend_items_count": len(legend_items),
        }
        return [figure]

    def _extract_tables(self, section: SectionDraft, text: str) -> List[dict]:
        if not text:
            return []

        lines = [line for line in text.split("\n") if line.strip()]
        tables: List[dict] = []

        idx = 0
        table_index = 1
        dotted_threshold = self.table_detection.get("dotted_leader_threshold", 3)
        min_pipe_cols = self.table_detection.get("min_pipe_columns", 3)

        while idx < len(lines):
            line = lines[idx]
            if "|" in line:
                table_lines: List[str] = []
                while idx < len(lines) and "|" in lines[idx]:
                    table_lines.append(lines[idx])
                    idx += 1
                table = self._parse_pipe_table(
                    section, table_lines, table_index, min_pipe_cols
                )
                if table:
                    tables.append(table)
                    table_index += 1
                continue

            if re.search(r"\.{%d,}" % dotted_threshold, line):
                table_lines = [line]
                idx += 1
                while idx < len(lines) and (
                    re.search(r"\.{%d,}" % dotted_threshold, lines[idx])
                    or lines[idx].strip().startswith("-")
                    or lines[idx].strip().startswith("*")
                ):
                    table_lines.append(lines[idx])
                    idx += 1
                table = self._parse_dotted_table(section, table_lines, table_index)
                if table:
                    tables.append(table)
                    table_index += 1
                continue

            idx += 1

        return tables

    def _parse_pipe_table(
        self,
        section: SectionDraft,
        table_lines: Sequence[str],
        table_index: int,
        min_pipe_cols: int,
    ) -> Optional[dict]:
        rows: List[List[str]] = []
        for raw in table_lines:
            parts = [part.strip() for part in raw.split("|")]
            if parts and parts[0] == "":
                parts = parts[1:]
            if parts and parts[-1] == "":
                parts = parts[:-1]
            if len([p for p in parts if p]) < min_pipe_cols:
                continue
            rows.append(parts)

        if not rows:
            return None

        csv_buffer = StringIO()
        writer = csv.writer(csv_buffer)
        for row in rows:
            writer.writerow(row)

        table_id = f"tbl_{section.section_id.replace('sec_', '')}_{table_index}"
        return {
            "table_id": table_id,
            "page": section.page_start,
            "caption": section.title,
            "n_rows": len(rows),
            "n_cols": max(len(row) for row in rows),
            "csv": csv_buffer.getvalue().strip(),
            "format": "pipe_delimited",
        }

    def _parse_dotted_table(
        self,
        section: SectionDraft,
        table_lines: Sequence[str],
        table_index: int,
    ) -> Optional[dict]:
        rows: List[List[str]] = []
        caption = section.title
        dotted_pattern = re.compile(r"\.{3,}")

        for raw in table_lines:
            if not dotted_pattern.search(raw):
                if raw.strip().startswith("-"):
                    cleaned = raw.strip("- ").strip()
                    if cleaned:
                        rows.append([cleaned])
                else:
                    caption = raw.strip()
                continue

            left, right = dotted_pattern.split(raw, maxsplit=1)
            col1 = left.strip(" -\t")
            col2 = right.strip()
            if col1 or col2:
                rows.append([col1, col2])

        if not rows:
            return None

        if len(rows[0]) == 1:
            # pad single column rows
            rows = [[row[0], ""] for row in rows]

        csv_buffer = StringIO()
        writer = csv.writer(csv_buffer)
        writer.writerow(["Item", "Value"])
        for row in rows:
            writer.writerow(row[:2])

        table_id = f"tbl_{section.section_id.replace('sec_', '')}_{table_index}"
        return {
            "table_id": table_id,
            "page": section.page_start,
            "caption": caption or section.title,
            "n_rows": len(rows) + 1,  # include header
            "n_cols": 2,
            "csv": csv_buffer.getvalue().strip(),
            "format": "dotted_leader",
        }

    def _collect_tags(
        self,
        section: SectionDraft,
        text: str,
        tables: Sequence[dict],
        figures: Sequence[dict],
    ) -> List[str]:
        tags: set[str] = set()
        lower_text = text.lower()
        title_lower = section.title.lower()

        comp_pattern = self.tagging_keywords.get("component_list_pattern")
        if comp_pattern:
            matches = re.findall(comp_pattern, text, re.MULTILINE)
            if len(matches) >= 2:
                tags.add("component_list")

        for keyword in self.tagging_keywords.get("dimensions", []):
            if keyword.lower() in lower_text:
                tags.update({"dimensions", "measurements"})
                break

        electrical_keywords = self.tagging_keywords.get("electrical", [])
        if any(kw in lower_text for kw in electrical_keywords):
            tags.add("electrical_specs")
            if "voltage" in lower_text or "hz" in lower_text:
                tags.add("voltage")

        model_keywords = self.tagging_keywords.get("model_codes", [])
        if any(kw in lower_text or kw in title_lower for kw in model_keywords):
            tags.add("model_codes")

        brand_keywords = self.tagging_keywords.get("brand", [])
        if any(kw in lower_text or kw in title_lower for kw in brand_keywords):
            tags.add("brand_specific")

        if tables:
            tags.add("table_present")
        if figures:
            tags.add("figure_present")

        return sorted(tags)

    def _classify_role(
        self,
        section: SectionDraft,
        text: str,
        tables: Sequence[dict],
        figures: Sequence[dict],
    ) -> str:
        if figures:
            return "figure"
        if tables:
            return "table"

        warning_keywords = [
            kw.lower() for kw in self.role_heuristics.get("warning_keywords", [])
        ]
        lower_content = f"{section.title}\n{text}".lower()
        if any(kw in lower_content for kw in warning_keywords):
            return "warning"

        return "main"

    # --------------------------------------------------------------------- #
    # Helpers - metadata and utilities
    # --------------------------------------------------------------------- #
    def _build_source_metadata(self, document_path: Path, page_count: int) -> dict:
        file_bytes = document_path.read_bytes()
        checksum = hashlib.sha256(file_bytes).hexdigest()
        return {
            "uri": str(document_path),
            "filename": document_path.name,
            "mime_type": "application/pdf",
            "checksum_sha256": checksum,
            "page_count": page_count,
        }

    def _make_document_code(self, document_path: Path) -> str:
        slug = re.sub(r"[^A-Za-z0-9]+", "_", document_path.stem).strip("_")
        return slug.upper() or "DOCUMENT"

    def _normalize_title(self, title: str) -> str:
        text = re.sub(r"^\d+(?:\.\d+)*\s*", "", title)
        text = text.lower()
        text = text.replace("&", " and ")
        text = re.sub(r"[^\w\s-]", "", text)
        text = text.replace("-", " ")
        text = re.sub(r"\s+", "_", text)
        text = re.sub(r"_+", "_", text).strip("_")
        return text or "section"

    def _build_section_id(self, heading: str, section_ids: Counter) -> str:
        numbering_match = re.match(r"^(\d+(?:\.\d+)*)", heading.replace(" ", ""))
        if numbering_match:
            base = numbering_match.group(1).replace(".", "_")
        else:
            base = self._normalize_title(heading)
        section_ids[base] += 1
        if section_ids[base] > 1:
            base = f"{base}_{section_ids[base]}"
        return f"sec_{base}"

    def _extraction_constraints(self) -> dict:
        filtering = self.global_config.get("filtering", {}) or {}
        structural = filtering.get("structural_filtering", {}) or {}
        return {
            "skip_patterns": list(self.symbolic_config.get("skip_header_footer_patterns", {}).values()),
            "max_chars_per_section": structural.get("max_chars_per_section"),
            "allowed_entities": self.symbolic_config.get("allowed_entities", []),
            "allowed_relations": self.symbolic_config.get("allowed_relations", []),
        }
