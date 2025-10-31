from __future__ import annotations

import csv
import hashlib
import logging
import re
import copy
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple
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
        self.rating_plate_config = self.symbolic_config.get("rating_plate", {}) or {}
        self.rating_plate_enabled = self.rating_plate_config.get("enabled", True)
        self.rating_plate_keywords = [
            kw.lower() for kw in self.rating_plate_config.get("title_keywords", ["rating plate"])
        ]
        self.rating_plate_brands = [
            brand.strip() for brand in self.rating_plate_config.get("brands", [])
        ]
        if not self.rating_plate_brands:
            self.rating_plate_brands = [brand.strip() for brand in self.tagging_keywords.get("brand", [])]
        self.rating_plate_region_tokens = [
            token.lower().strip() for token in self.rating_plate_config.get("region_tokens", [])
        ]
        self.rating_plate_default_tags = set(self.rating_plate_config.get("default_tags", []))

        self.implicit_config = self.symbolic_config.get("implicit_subsections", {}) or {}
        self.implicit_enabled = self.implicit_config.get("enabled", True)
        self.implicit_heading_patterns = [
            re.compile(pattern)
            for pattern in self.implicit_config.get(
                "heading_patterns",
                [r"^([A-Z][A-Za-z0-9\s,&-]+):?$", r"^([A-Z][A-Z\s/&-]+)$"],
            )
        ]
        self.implicit_max_heading_words = int(self.implicit_config.get("max_heading_words", 8))
        self.implicit_min_block_chars = int(self.implicit_config.get("min_block_chars", 40))

        self.warning_config = self.symbolic_config.get("warning_detection", {}) or {}
        self.warning_enabled = self.warning_config.get("enabled", True)
        self.warning_patterns = [
            re.compile(pattern, re.IGNORECASE | re.MULTILINE)
            for pattern in self.warning_config.get(
                "patterns",
                [
                    "(WARNING|CAUTION|DANGER|IMPORTANT|ATTENTION)",
                    "(not compatible|incompatible)",
                    "(do not|never|avoid).{0,50}(mix|combine|use with)",
                    "(must not|cannot be used|will not work)",
                    "^Note:?|^Nota:?",
                ],
            )
        ]
        self.warning_snippet_window = int(self.warning_config.get("snippet_window", 120))
        self.warning_promote_limit = int(self.warning_config.get("promote_role_max_chars", 600))

        self.dotted_min_rows = int(self.table_detection.get("min_consistent_rows", 3))
        self.dotted_max_separator_variance = int(self.table_detection.get("max_separator_variance", 2))
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

        for section in sections:
            text = "\n".join(section.lines).strip()
            figures = self._extract_figures(section, text)
            tables = self._extract_tables(section, text, figures)
            tags = self._collect_tags(section, text, tables, figures)
            role = self._classify_role(section, text, tables, figures)

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
        output_sections = self._merge_consecutive_tables(output_sections)
        if self.rating_plate_enabled:
            output_sections = self._split_rating_plate_sections(output_sections)
        if self.implicit_enabled:
            output_sections = self._detect_implicit_subsections(output_sections)
        if self.warning_enabled:
            output_sections = self._annotate_warning_sections(output_sections)

        covered_chars = sum(len(sec.get("text", "")) for sec in output_sections)
        total_tables = sum(len(sec.get("tables") or []) for sec in output_sections)
        total_figures = sum(len(sec.get("figures") or []) for sec in output_sections)

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
        text_lower = text.lower()

        has_keyword = any(kw in section.title.lower() or kw in text_lower for kw in figure_keywords)
        legend_items = self._extract_legend_items(text)

        if not has_keyword and not legend_items:
            return []

        figure_type = "diagram"
        for kw in ("photo", "image", "picture"):
            if kw in text_lower:
                figure_type = "photo"
        if "schematic" in text_lower:
            figure_type = "schematic"
        if self._is_rating_plate_section(section, text):
            figure_type = "photo"

        figure_id = f"fig_{section.section_id.replace('sec_', '')}"
        figure = {
            "figure_id": figure_id,
            "page": section.page_start,
            "caption": section.title,
            "type": figure_type,
            "bbox": None,
            "has_legend": bool(legend_items),
            "legend_items_count": len(legend_items),
            "legend_items": legend_items,
        }
        return [figure]

    def _extract_tables(
        self,
        section: SectionDraft,
        text: str,
        figures: Sequence[dict],
    ) -> List[dict]:
        if not text:
            return []

        if figures:
            return []

        if self.rating_plate_enabled and self._is_rating_plate_section(section, text):
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
                if not self._is_valid_pipe_table(table_lines, min_pipe_cols):
                    continue
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
                if not self._is_valid_dotted_table(table_lines):
                    continue
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

    def _merge_consecutive_tables(self, sections: List[dict]) -> List[dict]:
        merged: List[dict] = []
        current: Optional[dict] = None

        def is_mergeable(sec: dict) -> bool:
            if sec.get("role") != "table":
                return False
            tables = sec.get("tables") or []
            if not tables:
                return False
            return tables[0].get("format") == "dotted_leader"

        for section in sections:
            if is_mergeable(section):
                if (
                    current
                    and is_mergeable(current)
                    and section.get("parent_id") == current.get("parent_id")
                ):
                    dst_table = current["tables"][0]
                    src_table = section["tables"][0]
                    dst_lines = dst_table["csv"].splitlines()
                    src_lines = src_table["csv"].splitlines()
                    if src_lines:
                        rows_to_add = src_lines[1:] if len(src_lines) > 1 else []
                        if rows_to_add:
                            dst_table["csv"] = "\n".join(dst_lines + rows_to_add)
                            dst_table["n_rows"] = len(dst_lines + rows_to_add)
                    current["text"] = "\n".join(
                        filter(None, [current.get("text"), section.get("text")])
                    ).strip()
                    current["page_end"] = max(
                        current.get("page_end", 0), section.get("page_end", 0)
                    )
                    current["char_end"] = max(
                        current.get("char_end", 0), section.get("char_end", 0)
                    )
                    merged_tags = set(current.get("tags", [])) | set(
                        section.get("tags", [])
                    )
                    current["tags"] = sorted(merged_tags)
                    continue

                if current:
                    merged.append(current)
                current = section
                continue

            if current:
                merged.append(current)
                current = None
            merged.append(section)

        if current:
            merged.append(current)
        return merged

    def _split_rating_plate_sections(self, sections: List[dict]) -> List[dict]:
        if not self.rating_plate_enabled:
            return sections

        if not self.rating_plate_brands:
            return sections

        result: List[dict] = []
        for section in sections:
            if not self._is_rating_plate_section(section, section.get("text", "")):
                result.append(section)
                continue

            rating_subsections = self._create_rating_plate_subsections(section)
            if not rating_subsections:
                result.append(section)
                continue

            section["text"] = ""
            section["char_end"] = section.get("char_start", 0)
            section["tables"] = []
            section["figures"] = []
            section["tags"] = sorted(
                set(section.get("tags", [])) | self.rating_plate_default_tags
            )
            result.append(section)
            result.extend(rating_subsections)
        return result

    def _create_rating_plate_subsections(self, section: dict) -> List[dict]:
        original_text = section.get("text") or ""
        if not original_text:
            return []

        brand_regex = "|".join(re.escape(name) for name in self.rating_plate_brands)
        if not brand_regex:
            return []

        region_regex = ""
        if self.rating_plate_region_tokens:
            region_regex = "|".join(
                re.escape(token) for token in self.rating_plate_region_tokens
            )

        normalized_text = (
            original_text.replace("\u00ad", "-").replace("\u2010", "-").replace("\u2011", "-")
        )
        lines = normalized_text.split("\n")
        line_positions: List[int] = []
        cursor = 0
        for line in lines:
            line_positions.append(cursor)
            cursor += len(line) + 1
        line_positions.append(len(normalized_text))

        if region_regex:
            region_group = rf"(?:[, ]+({region_regex})[\s\u2010\u00ad-]*version)?"
        else:
            region_group = r"(?:[, ]+([A-Za-z/]{2,5})[\s\u2010\u00ad-]*version)?"

        brand_pattern = re.compile(rf"({brand_regex}){region_group}", re.IGNORECASE)

        headers: List[Tuple[int, str, Optional[str]]] = []
        for idx, line in enumerate(lines):
            match = brand_pattern.search(line)
            if match:
                brand = match.group(1).strip()
                region = match.group(2)
                headers.append((idx, brand, region))

        if not headers:
            return []

        headers.append((len(lines), "", None))
        subsections: List[dict] = []
        existing_tags = set(section.get("tags", []))

        for i in range(len(headers) - 1):
            start_idx, brand, region = headers[i]
            next_idx = headers[i + 1][0]
            if not brand:
                continue
            start_pos = line_positions[start_idx]
            end_pos = line_positions[next_idx]
            block_text = original_text[start_pos:end_pos].strip()
            if not block_text:
                continue

            brand_slug = brand.lower().replace(" ", "_")
            region_slug = f"_{region.lower()}" if region else ""
            base_id = f"{section['section_id']}_{brand_slug}{region_slug}"
            new_section = copy.deepcopy(section)
            region_tag = region.lower() if region else None
            new_section.update(
                {
                    "section_id": base_id,
                    "parent_id": section["section_id"],
                    "title": f"{section['title']} - {brand}"
                    + (f", {region}-version" if region else ""),
                    "title_normalized": f"{section['title_normalized']}_{brand_slug}"
                    + (f"_{region.lower()}" if region else ""),
                    "role": "figure",
                    "level": section["level"] + 1,
                    "char_start": section.get("char_start", 0) + start_pos,
                    "char_end": section.get("char_start", 0) + start_pos + len(block_text),
                    "text": block_text,
                    "tables": [],
                    "figures": section.get("figures") or [],
                    "tags": sorted(
                        existing_tags
                        | self.rating_plate_default_tags
                        | {brand_slug}
                        | ({region_tag} if region_tag else set())
                    ),
                }
            )
            new_section.pop("warnings", None)
            subsections.append(new_section)

        return subsections

    def _detect_implicit_subsections(self, sections: List[dict]) -> List[dict]:
        existing_ids = {sec["section_id"] for sec in sections}
        result: List[dict] = []

        for section in sections:
            extracted = self._extract_implicit_subsections(section, existing_ids)
            if not extracted:
                result.append(section)
                continue

            prefix_text, new_sections = extracted
            section["text"] = prefix_text
            if prefix_text:
                section["char_end"] = section.get("char_start", 0) + len(prefix_text)
            else:
                section["char_end"] = section.get("char_start", 0)
            section["tags"] = sorted(set(section.get("tags", [])))
            result.append(section)
            for new_sec in new_sections:
                existing_ids.add(new_sec["section_id"])
                result.append(new_sec)
        return result

    def _extract_implicit_subsections(
        self,
        section: dict,
        existing_ids: Set[str],
    ) -> Optional[Tuple[str, List[dict]]]:
        text = section.get("text") or ""
        if not text:
            return None

        patterns = self.implicit_heading_patterns or [
            re.compile(r"^([A-Z][A-Za-z0-9\s,&-]+):?$"),
            re.compile(r"^([A-Z][A-Z\s/&-]+)$"),
        ]
        lines = text.split("\n")
        headings: List[Tuple[int, str]] = []
        for idx, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            for pattern in patterns:
                match = pattern.match(stripped)
                if match and len(stripped.split()) <= self.implicit_max_heading_words:
                    headings.append((idx, match.group(1).strip(" :")))
                    break

        if len(headings) < 2:
            return None

        line_positions: List[int] = []
        cursor = 0
        for line in lines:
            line_positions.append(cursor)
            cursor += len(line) + 1
        line_positions.append(len(text))

        subsections: List[dict] = []
        for i, (line_idx, title) in enumerate(headings):
            next_idx = headings[i + 1][0] if i + 1 < len(headings) else len(lines)
            start_pos = line_positions[line_idx]
            end_pos = line_positions[next_idx]
            block_text = text[start_pos:end_pos].strip()
            if len(block_text) < self.implicit_min_block_chars:
                continue

            suffix = self._normalize_title(title) or "section"
            candidate_id = f"{section['section_id']}_{suffix}"
            counter = 1
            new_id = candidate_id
            while new_id in existing_ids:
                counter += 1
                new_id = f"{candidate_id}_{counter}"
            new_section = {
                "section_id": new_id,
                "parent_id": section["section_id"],
                "title": title,
                "title_normalized": f"{section['title_normalized']}_{suffix}",
                "role": "main",
                "level": section["level"] + 1,
                "language": section["language"],
                "language_confidence": section["language_confidence"],
                "page_start": section["page_start"],
                "page_end": section["page_end"],
                "char_start": section.get("char_start", 0) + start_pos,
                "char_end": section.get("char_start", 0) + start_pos + len(block_text),
                "text": block_text,
                "is_suspect": section.get("is_suspect", False),
                "tables": [],
                "figures": [],
                "tags": sorted(set(section.get("tags", []))),
            }
            subsections.append(new_section)
            existing_ids.add(new_id)

        if not subsections:
            return None

        first_heading_idx = headings[0][0]
        prefix_end = line_positions[first_heading_idx]
        prefix_text = text[:prefix_end].strip()

        return prefix_text, subsections

    def _annotate_warning_sections(self, sections: List[dict]) -> List[dict]:
        if not self.warning_enabled:
            return sections

        for section in sections:
            warnings = self._detect_warning_mentions(section.get("text", ""))
            if not warnings:
                continue
            section.setdefault("tags", [])
            if "warning" not in section["tags"]:
                section["tags"].append("warning")
            section["tags"] = sorted(set(section["tags"]))
            section["warnings"] = warnings
            if (
                section.get("role") in {"main", "table"}
                and len(section.get("text", "")) < self.warning_promote_limit
            ):
                section["role"] = "warning"
        return sections

    def _detect_warning_mentions(self, text: str) -> List[dict]:
        if not text or not self.warning_enabled or not self.warning_patterns:
            return []

        warnings: List[dict] = []
        seen_snippets: set[str] = set()

        for pattern in self.warning_patterns:
            for match in pattern.finditer(text):
                start = max(0, match.start() - self.warning_snippet_window)
                end = min(len(text), match.end() + self.warning_snippet_window)
                window = text[start:end]
                match_relative_start = match.start() - start
                match_relative_end = match.end() - start
                sentence_start = window.rfind(". ", 0, match_relative_start)
                sentence_end = window.find(". ", match_relative_end)
                if sentence_start != -1:
                    window = window[sentence_start + 2 :]
                if sentence_end != -1:
                    window = window[: sentence_end + 1]
                snippet = window.strip()
                if not snippet or snippet in seen_snippets:
                    continue
                seen_snippets.add(snippet)
                keyword = match.group(1) if match.lastindex else match.group(0)
                severity = (
                    "warning"
                    if re.search(r"warning|caution|danger|not compatible", keyword, re.IGNORECASE)
                    else "note"
                )
                warnings.append(
                    {
                        "type": severity,
                        "text": snippet,
                        "keyword": keyword,
                    }
                )

        return warnings

    def _is_rating_plate_section(self, section: object, text: str) -> bool:
        if not self.rating_plate_enabled:
            return False

        title = getattr(section, "title", "") if hasattr(section, "title") else ""
        if isinstance(section, dict):
            title = section.get("title", title)
        title_lower = (title or "").lower().replace("\u00a0", " ")
        text_lower = (text or "").lower().replace("\u00a0", " ")
        keywords = self.rating_plate_keywords or ["rating plate"]
        return any(keyword in title_lower or keyword in text_lower for keyword in keywords)

    def _is_valid_pipe_table(
        self,
        lines: Sequence[str],
        min_pipe_cols: int,
    ) -> bool:
        if len(lines) < 2:
            return False
        column_counts = [
            len([part for part in line.split("|") if part.strip()]) for line in lines
        ]
        if max(column_counts, default=0) < min_pipe_cols:
            return False
        return True

    def _is_valid_dotted_table(self, lines: Sequence[str]) -> bool:
        dotted_lines = [line for line in lines if re.search(r"\.{3,}", line)]
        if len(dotted_lines) < self.dotted_min_rows:
            return False
        separator_counts = [
            len(re.findall(r"\.{3,}", line)) for line in dotted_lines
        ]
        if separator_counts and len(set(separator_counts)) > self.dotted_max_separator_variance:
            return False
        leading = lines[: min(3, len(lines))]
        if leading and all(item.strip().startswith("-") for item in leading):
            return False
        return True

    def _extract_legend_items(self, text: str) -> List[dict]:
        pattern = re.compile(
            r"^\s*(\d+)\s*[.)]\s+(.+?)(?=\n\s*\d+[.)]|\n\n|$)",
            re.MULTILINE | re.DOTALL,
        )
        items: List[dict] = []
        for match in pattern.finditer(text):
            number = int(match.group(1))
            description = " ".join(match.group(2).split())
            items.append({"number": number, "description": description})
        return items

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
