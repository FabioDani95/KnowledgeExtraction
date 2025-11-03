from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence
from uuid import uuid4

from PyPDF2 import PdfReader

from ..base_pipeline import BaseSymbolicParser


@dataclass
class PageContent:
    number: int
    text: str
    lines: List[str] = field(default_factory=list)


class TroubleshootingSymbolicParser(BaseSymbolicParser):
    """Symbolic parser for troubleshooting manuals."""

    def __init__(
        self,
        root_dir: Path,
        pipeline_config: dict,
        global_config: dict,
    ) -> None:
        super().__init__(root_dir, "troubleshooting", pipeline_config, global_config)
        self.symbolic_config = pipeline_config.get("symbolic", {}) or {}

        pdf_cfg = global_config.get("pdf_processing", {}) or {}
        self.default_language = pdf_cfg.get("default_language", "en")
        self.datasource_code = pdf_cfg.get("datasource_code")

        self.skip_patterns = self._compile_skip_patterns(
            self.symbolic_config.get("skip_header_footer_patterns", {})
        )

        classification_cfg = self.symbolic_config.get("page_classification", {}) or {}
        self.decision_keywords = [
            kw.lower() for kw in classification_cfg.get("decision_keywords", [])
        ]
        self.torque_keywords = [
            kw.lower() for kw in classification_cfg.get("torque_keywords", [])
        ]
        self.safety_keywords = [
            kw.lower() for kw in classification_cfg.get("safety_keywords", [])
        ]
        self.max_warning_role_chars = int(
            classification_cfg.get("max_warning_role_chars", 500)
        )

        decision_cfg = self.symbolic_config.get("decision_tree", {}) or {}
        self.decision_check_pattern = re.compile(
            decision_cfg.get("check_pattern", r"(\d+)\s+(.+)")
        )
        self.decision_subcheck_pattern = re.compile(
            decision_cfg.get("subcheck_pattern", r"(\d+\.\d+)\s+(.+)")
        )
        default_branch_pattern = (
            r"(YES|NO)\s*(?:[-–—:\u00ad])?\s*(.*?)(?=(YES|NO)\s*(?:[-–—:\u00ad]|$)|$)"
        )
        self.decision_branch_pattern = re.compile(
            decision_cfg.get("branch_pattern", default_branch_pattern),
            re.IGNORECASE,
        )
        self.decision_table_id = decision_cfg.get("table_id", "tbl_troubleshooting")
        self.decision_format = decision_cfg.get(
            "format", "hierarchical_decision_tree"
        )
        self.decision_tags = decision_cfg.get(
            "tags", ["troubleshooting", "checklist"]
        )

        torque_cfg = self.symbolic_config.get("torque_table", {}) or {}
        self.torque_pattern = re.compile(
            torque_cfg.get(
                "value_pattern",
                r"(\d+)\s*\(\+(\d+)/-(\d+)\)\s*(Ncm|Nm)",
            )
        )
        self.torque_headers = torque_cfg.get(
            "csv_headers",
            ["Component", "Nominal", "Tolerance_plus", "Tolerance_minus", "Unit"],
        )
        self.torque_table_id = torque_cfg.get("table_id", "tbl_torque_specs")
        self.torque_format = torque_cfg.get("format", "technical_specs")
        self.torque_tags = torque_cfg.get("tags", ["torque", "repair"])

        safety_cfg = self.symbolic_config.get("safety_detection", {}) or {}
        self.safety_patterns = [
            re.compile(pattern, re.IGNORECASE | re.MULTILINE)
            for pattern in safety_cfg.get(
                "patterns",
                [
                    r"(⚠️|⚠|Risk of|Danger of)",
                    r"(fatal electrical shock|burns|hot parts|injury)",
                ],
            )
        ]
        self.safety_snippet_chars = int(safety_cfg.get("snippet_chars", 220))
        self.safety_tags = safety_cfg.get("tags", ["safety"])

        generic_cfg = self.symbolic_config.get("generic", {}) or {}
        self.generic_tag = generic_cfg.get("tag", "troubleshooting_text")

        self.logger = logging.getLogger("pipeline.troubleshooting.symbolic")

    def parse_document(self, document_path: Path) -> dict:
        reader = PdfReader(str(document_path))
        page_count = len(reader.pages)
        pdf_cfg = self.global_config.get("pdf_processing", {}) or {}
        skip_start = max(int(pdf_cfg.get("skip_start_pages", 0)), 0)
        skip_end = max(int(pdf_cfg.get("skip_end_pages", 0)), 0)
        processed_end = max(page_count - skip_end, skip_start)

        pages: List[PageContent] = []
        total_chars = 0
        discarded_blocks: List[dict] = []
        discarded_chars = 0

        for idx, page in enumerate(reader.pages):
            if idx < skip_start or idx >= processed_end:
                continue
            raw_text = page.extract_text() or ""
            raw_text = raw_text.replace("\r", "")
            filtered_lines: List[str] = []

            for line in raw_text.split("\n"):
                clean_line = line.strip()
                reason = self._match_discard(line, clean_line)
                if reason:
                    discarded_blocks.append(
                        {"reason": reason, "text": clean_line, "page": idx + 1}
                    )
                    discarded_chars += len(line) + 1
                elif clean_line:
                    filtered_lines.append(clean_line)

            text = "\n".join(filtered_lines)
            total_chars += len(text)
            pages.append(PageContent(number=idx + 1, text=text, lines=filtered_lines))

        sections: List[dict] = []
        char_offset = 0
        total_tables = 0
        total_figures = 0
        covered_chars = 0

        for page in pages:
            section = self._build_section(page, char_offset)
            sections.append(section)
            char_offset = section["char_end"] + 1
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

        return {
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

    def _build_section(self, page: PageContent, char_start: int) -> dict:
        classification = self._classify_page(page)
        tags: List[str] = []
        role = "main"
        tables: List[dict] = []
        figures: List[dict] = []
        extras: Dict[str, object] = {}
        text = page.text

        if classification == "decision_tree":
            table, remainder = self._parse_decision_tree(page)
            if table:
                tables.append(table)
                tags.extend(self.decision_tags)
            text = remainder
            role = "table" if table else "main"
        elif classification == "torque":
            table, remainder = self._parse_torque_table(page)
            if table:
                tables.append(table)
                tags.extend(self.torque_tags)
            text = remainder
            role = "table" if table else "main"
        else:
            if self.generic_tag:
                tags.append(self.generic_tag)

        warnings = self._detect_safety_warnings(text, page.number)
        if warnings:
            extras["warnings"] = warnings
            tags.extend(self.safety_tags)
            if len(text) <= self.max_warning_role_chars:
                role = "warning"

        section = {
            "section_id": f"sec_{page.number}",
            "parent_id": None,
            "title": self._derive_title(page, classification),
            "title_normalized": self._normalize_title(
                self._derive_title(page, classification)
            ),
            "role": role,
            "level": 1,
            "language": self.default_language,
            "language_confidence": 1.0,
            "page_start": page.number,
            "page_end": page.number,
            "char_start": char_start,
            "char_end": char_start + len(text),
            "text": text,
            "is_suspect": False,
            "tables": tables,
            "figures": figures,
            "tags": sorted(set(tags)),
        }
        section.update(extras)
        return section

    def _classify_page(self, page: PageContent) -> str:
        normalized = page.text.replace("\u00a0", " ").lower()
        if self._contains_all(normalized, self.decision_keywords):
            return "decision_tree"
        if any(keyword in normalized for keyword in self.torque_keywords):
            if self.torque_pattern.search(page.text):
                return "torque"
        if any(keyword in normalized for keyword in self.safety_keywords):
            return "safety"
        return "generic"

    def _parse_decision_tree(self, page: PageContent) -> (Optional[dict], str):
        checks: List[dict] = []
        current_check: Optional[dict] = None
        current_subcheck: Optional[dict] = None
        remaining_lines: List[str] = []

        for line in page.lines:
            normalized_line = (
                line.replace("\u00a0", " ")
                .replace("\u2013", "-")
                .replace("\u2014", "-")
                .replace("\u00ad", "")
            )

            match = self.decision_check_pattern.search(normalized_line)
            if match:
                prefix = normalized_line[: match.start()].strip()
                if prefix:
                    remaining_lines.append(prefix)
                candidate_name = match.group(2).strip()
                if candidate_name.isupper():
                    remaining_lines.append(normalized_line)
                    continue
                if current_subcheck and current_check:
                    current_check.setdefault("subchecks", []).append(current_subcheck)
                if current_check:
                    checks.append(current_check)
                current_check = {
                    "id": match.group(1).strip(),
                    "name": candidate_name,
                    "subchecks": [],
                }
                current_subcheck = None
                branches, cleaned_tail = self._extract_branches(normalized_line[match.end():])
                if branches:
                    current_check.setdefault("branches", []).extend(branches)
                if cleaned_tail:
                    remaining_lines.append(cleaned_tail)
                continue

            match = self.decision_subcheck_pattern.search(normalized_line)
            if match:
                prefix = normalized_line[: match.start()].strip()
                if prefix:
                    remaining_lines.append(prefix)
                if current_subcheck and current_check:
                    current_check.setdefault("subchecks", []).append(current_subcheck)
                current_subcheck = {
                    "id": match.group(1).strip(),
                    "symptom": match.group(2).strip(),
                    "branches": [],
                }
                tail_branches, cleaned_tail = self._extract_branches(
                    normalized_line[match.end():]
                )
                if tail_branches:
                    current_subcheck.setdefault("branches", []).extend(tail_branches)
                if cleaned_tail:
                    current_subcheck["symptom"] = (
                        f"{current_subcheck['symptom']} {cleaned_tail}".strip()
                    )
                continue

            branch_hits, cleaned_line = self._extract_branches(normalized_line)
            if branch_hits and current_subcheck is not None:
                current_subcheck.setdefault("branches", []).extend(branch_hits)
                if cleaned_line:
                    current_subcheck["symptom"] = (
                        f"{current_subcheck['symptom']} {cleaned_line}".strip()
                    )
                continue

            if current_subcheck is not None:
                current_subcheck["symptom"] = f"{current_subcheck['symptom']} {normalized_line}".strip()
            elif current_check is not None:
                remaining_lines.append(normalized_line)
            else:
                remaining_lines.append(line)

        if current_subcheck and current_check:
            current_check.setdefault("subchecks", []).append(current_subcheck)
        if current_check:
            checks.append(current_check)

        if not checks:
            return None, page.text

        table = {
            "table_id": self.decision_table_id,
            "page": page.number,
            "caption": f"Troubleshooting checks page {page.number}",
            "format": self.decision_format,
            "structure": {"checks": checks},
        }
        return table, "\n".join(remaining_lines).strip()

    def _parse_torque_table(self, page: PageContent) -> (Optional[dict], str):
        rows: List[List[str]] = []
        remaining_lines: List[str] = []

        for line in page.lines:
            match = self.torque_pattern.search(line)
            if match:
                component = line[: match.start()].strip(" :•-•")
                nominal = match.group(1)
                tol_plus = match.group(2)
                tol_minus = match.group(3)
                unit = match.group(4)
                rows.append([component, nominal, tol_plus, tol_minus, unit])
            else:
                remaining_lines.append(line)

        if not rows:
            return None, page.text

        csv_lines = [",".join(self._csv_escape(val) for val in self.torque_headers)]
        for row in rows:
            csv_lines.append(",".join(self._csv_escape(val) for val in row))

        table = {
            "table_id": self.torque_table_id,
            "page": page.number,
            "caption": f"Torque specifications page {page.number}",
            "n_rows": len(rows) + 1,
            "n_cols": len(self.torque_headers),
            "csv": "\n".join(csv_lines),
            "format": self.torque_format,
        }
        return table, "\n".join(remaining_lines).strip()

    def _detect_safety_warnings(self, text: str, page: int) -> List[dict]:
        warnings: List[dict] = []
        normalized = text.replace("\n", " ")

        for pattern in self.safety_patterns:
            for match in pattern.finditer(normalized):
                start = max(match.start() - 20, 0)
                end = min(match.end() + self.safety_snippet_chars, len(normalized))
                snippet = normalized[start:end].strip()
                severity = "warning"
                snippet_lower = snippet.lower()
                if "fatal" in snippet_lower:
                    severity = "fatal"
                elif "danger" in snippet_lower:
                    severity = "danger"

                warnings.append(
                    {
                        "type": "safety_warning",
                        "severity": severity,
                        "text": snippet,
                        "page": page,
                    }
                )
        return warnings

    def _build_source_metadata(self, document_path: Path, page_count: int) -> dict:
        checksum = hashlib.sha256(document_path.read_bytes()).hexdigest()
        return {
            "uri": str(document_path),
            "filename": document_path.name,
            "mime_type": "application/pdf",
            "checksum_sha256": checksum,
            "page_count": page_count,
        }

    def _match_discard(self, raw_line: str, clean_line: str) -> Optional[str]:
        for reason, pattern in self.skip_patterns.items():
            target = raw_line if reason == "decoration" else clean_line
            if pattern.match(target):
                return reason
        return None

    @staticmethod
    def _compile_skip_patterns(patterns: Dict[str, str]) -> Dict[str, re.Pattern]:
        return {
            reason: re.compile(pattern, re.IGNORECASE)
            for reason, pattern in (patterns or {}).items()
            if pattern
        }

    @staticmethod
    def _contains_all(text: str, keywords: Sequence[str]) -> bool:
        return all(keyword in text for keyword in keywords) if keywords else False

    def _extract_branches(self, text: str) -> (List[dict], str):
        branches: List[dict] = []
        if not text:
            return branches, ""

        cleaned_parts: List[str] = []
        cursor = 0
        for match in self.decision_branch_pattern.finditer(text):
            start, end = match.span()
            prefix = text[cursor:start].strip()
            if prefix:
                cleaned_parts.append(prefix)
            condition = match.group(1).upper()
            action = match.group(2).strip()
            if action:
                branches.append({"condition": condition, "action": action})
            cursor = end
        suffix = text[cursor:].strip()
        if suffix:
            cleaned_parts.append(suffix)

        cleaned_text = " ".join(cleaned_parts).strip()
        return branches, cleaned_text

    def _derive_title(self, page: PageContent, classification: str) -> str:
        if page.lines:
            first_line = page.lines[0].replace("\u00a0", " ")
            if first_line:
                return first_line
        if classification == "decision_tree":
            return f"Troubleshooting checks (page {page.number})"
        if classification == "torque":
            return f"Torque specifications (page {page.number})"
        if classification == "safety":
            return f"Safety instructions (page {page.number})"
        return f"Troubleshooting content (page {page.number})"

    def _csv_escape(self, value: str) -> str:
        if value is None:
            return ""
        if any(ch in value for ch in [",", '"', "\n"]):
            return '"' + value.replace('"', '""') + '"'
        return value

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

    def _normalize_title(self, title: str) -> str:
        text = title.lower()
        text = re.sub(r"[^\w\s-]", "", text)
        text = re.sub(r"\s+", "_", text)
        return re.sub(r"_+", "_", text).strip("_")

    def _make_document_code(self, document_path: Path) -> str:
        slug = re.sub(r"[^A-Za-z0-9]+", "_", document_path.stem).strip("_")
        return slug.upper() or "DOCUMENT"
