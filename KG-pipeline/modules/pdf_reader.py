"""
PDF Reader Module
Parses PDF documents from the source directory and structures them according to the parsed_document schema.
"""

import os
import json
import hashlib
import re
import yaml
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
from PyPDF2 import PdfReader
import logging
try:
    from jsonschema import validate, ValidationError
    _JSONSCHEMA_AVAILABLE = True
except ModuleNotFoundError:
    validate = None  # type: ignore[assignment]
    _JSONSCHEMA_AVAILABLE = False

    class ValidationError(Exception):
        """Fallback ValidationError when jsonschema is unavailable."""
        pass

try:
    from langdetect import detect, LangDetectException
    _LANGDETECT_AVAILABLE = True
except ModuleNotFoundError:
    detect = None  # type: ignore[assignment]
    _LANGDETECT_AVAILABLE = False

    class LangDetectException(Exception):
        """Fallback LangDetectException when langdetect is unavailable."""
        pass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFReader:
    """
    Handles PDF document parsing and conversion to structured format.
    """

    def __init__(self, source_dir: str, schema_path: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the PDF reader with source directory and schema.

        Args:
            source_dir: Path to directory containing PDF files
            schema_path: Path to the parsed_document.json schema file
            config: Optional configuration dictionary for filtering and processing
        """
        self.source_dir = Path(source_dir)
        self.schema_path = Path(schema_path)

        if not self.source_dir.exists():
            raise FileNotFoundError(f"Source directory not found: {source_dir}")

        if not self.schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {schema_path}")

        # Load schema
        with open(self.schema_path, "r", encoding="utf-8") as f:
            self.schema = json.load(f)

        # Load configuration
        self.config = config or self._load_default_config()

        logger.info(f"PDFReader initialized with source: {self.source_dir}")
        if self.config.get("filtering", {}).get("enabled"):
            logger.info("Content filtering is ENABLED")

    def _load_default_config(self) -> Dict[str, Any]:
        """
        Load default configuration if no config is provided.
        """
        return {
            "filtering": {
                "enabled": False,
                "pattern_filtering": {"enabled": False},
                "content_quality": {"enabled": False},
                "language_detection": {"enabled": False},
                "structural_filtering": {"enabled": False}
            }
        }

    def _is_english(self, text: str) -> bool:
        """
        Check if text is in English using language detection.
        Returns True if English, False otherwise.
        """
        if not _LANGDETECT_AVAILABLE:
            logger.warning("langdetect not available, skipping language detection")
            return True

        lang_config = self.config.get("filtering", {}).get("language_detection", {})
        min_length = lang_config.get("min_text_for_detection", 50)
        fallback = lang_config.get("fallback_on_error", True)

        if len(text.strip()) < min_length:
            logger.debug(f"Text too short for language detection ({len(text)} chars)")
            return fallback

        try:
            detected_lang = detect(text)  # type: ignore[misc]
            target_lang = lang_config.get("target_language", "en")
            return detected_lang == target_lang
        except (LangDetectException, Exception) as e:
            logger.debug(f"Language detection failed: {e}")
            return fallback

    def _matches_skip_pattern(self, title: str, text: str) -> bool:
        """
        Check if section matches any skip pattern.
        """
        pattern_config = self.config.get("filtering", {}).get("pattern_filtering", {})
        if not pattern_config.get("enabled", False):
            return False

        skip_patterns = pattern_config.get("skip_patterns", [])
        case_sensitive = pattern_config.get("case_sensitive", False)

        content = f"{title} {text}"
        if not case_sensitive:
            content = content.lower()

        for pattern in skip_patterns:
            try:
                flags = 0 if case_sensitive else re.IGNORECASE
                if re.search(pattern, content, flags):
                    logger.debug(f"Section matched skip pattern: {pattern}")
                    return True
            except re.error as e:
                logger.warning(f"Invalid regex pattern '{pattern}': {e}")

        return False

    def _has_sufficient_quality(self, text: str) -> bool:
        """
        Check if section has sufficient content quality.
        """
        quality_config = self.config.get("filtering", {}).get("content_quality", {})
        if not quality_config.get("enabled", False):
            return True

        min_text_length = quality_config.get("min_text_length", 100)
        max_whitespace_ratio = quality_config.get("max_whitespace_ratio", 0.8)
        min_non_whitespace = quality_config.get("min_non_whitespace_chars", 50)

        # Check minimum length
        if len(text) < min_text_length:
            logger.debug(f"Section too short: {len(text)} < {min_text_length}")
            return False

        # Check whitespace ratio
        total_chars = len(text)
        non_whitespace_chars = len(text.replace(" ", "").replace("\n", "").replace("\t", ""))

        if non_whitespace_chars < min_non_whitespace:
            logger.debug(f"Insufficient non-whitespace chars: {non_whitespace_chars}")
            return False

        whitespace_ratio = 1 - (non_whitespace_chars / total_chars) if total_chars > 0 else 1
        if whitespace_ratio > max_whitespace_ratio:
            logger.debug(f"Too much whitespace: {whitespace_ratio:.2f} > {max_whitespace_ratio}")
            return False

        return True

    def _should_skip_section_title(self, title: str) -> bool:
        """
        Check if section title is in the skip list.
        """
        struct_config = self.config.get("filtering", {}).get("structural_filtering", {})
        if not struct_config.get("enabled", False):
            return False

        skip_titles = struct_config.get("skip_section_titles", [])
        title_lower = title.lower().strip()

        for skip_title in skip_titles:
            if skip_title.lower() in title_lower:
                logger.debug(f"Skipping section with title: {title}")
                return True

        return False

    def _apply_filters(self, section: Dict[str, Any]) -> bool:
        """
        Apply all enabled filters to a section.
        Returns True if section should be kept, False if it should be filtered out.
        """
        if not self.config.get("filtering", {}).get("enabled", False):
            return True

        title = section.get("title", "")
        text = section.get("text", "")

        # Priority 1 Filters
        if self._should_skip_section_title(title):
            logger.info(f"Filtered out section by title: {title}")
            return False

        if self._matches_skip_pattern(title, text):
            logger.info(f"Filtered out section by pattern: {title}")
            return False

        if not self._has_sufficient_quality(text):
            logger.info(f"Filtered out section by quality: {title}")
            return False

        # Priority 2 Filter: Language Detection
        lang_config = self.config.get("filtering", {}).get("language_detection", {})
        if lang_config.get("enabled", False):
            if not self._is_english(text):
                logger.info(f"Filtered out non-English section: {title}")
                return False

        return True

    def calculate_checksum(self, filepath: Path) -> str:
        """
        Calculate SHA256 checksum of a file.
        """
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def extract_text_from_pdf(self, pdf_path: Path) -> tuple[str, int]:
        """
        Extract text content from PDF file.
        """
        try:
            with open(pdf_path, "rb") as file:
                pdf_reader = PdfReader(file)
                page_count = len(pdf_reader.pages)

                text_content = []
                for page_num, page in enumerate(pdf_reader.pages, start=1):
                    try:
                        text = page.extract_text() or ""
                        if text.strip():
                            text_content.append(text)
                    except Exception as e:
                        logger.warning(
                            f"Error extracting text from page {page_num} of {pdf_path.name}: {e}"
                        )

                full_text = "\n\n".join(text_content)
                return full_text, page_count

        except Exception as e:
            logger.error(f"Failed to extract text from {pdf_path.name}: {e}")
            raise

    def detect_sections(self, text: str, page_count: int) -> List[Dict[str, Any]]:
        """
        Detect and structure sections from extracted text.
        Basic version (single section). Can be expanded with NLP later.
        Applies filtering based on configuration.
        """
        sections = []
        current_section = {
            "section_id": "S1",
            "title": "Main Content",
            "level": 1,
            "page_start": 1,
            "page_end": page_count,
            "char_start": 0,
            "char_end": len(text),
            "text": text,
            "tables": [],
            "figures": [],
            "tags": [],
        }

        # Apply filters to section
        if self._apply_filters(current_section):
            sections.append(current_section)
            logger.debug(f"Section '{current_section['title']}' passed all filters")
        else:
            logger.info(f"Section '{current_section['title']}' was filtered out")

        return sections

    def extract_metadata(
        self, pdf_path: Path, pdf_reader: Optional[PdfReader] = None
    ) -> Dict[str, Any]:
        """
        Extract metadata from PDF file.
        """
        metadata = {
            "product_hint": "",
            "model_hint": "",
            "brand_hint": "",
            "doc_type": "technical_document",
            "version": "",
            "year": datetime.now().year,
        }

        try:
            if pdf_reader is None:
                with open(pdf_path, "rb") as file:
                    pdf_reader = PdfReader(file)

            info = pdf_reader.metadata or {}

            title = (info.get("/Title") or "").strip()
            subject = (info.get("/Subject") or "").strip()
            author = (info.get("/Author") or "").strip()
            creation = (info.get("/CreationDate") or "").strip()

            if title:
                metadata["product_hint"] = title
            if subject:
                metadata["model_hint"] = subject
            if author:
                metadata["brand_hint"] = author
            if creation.startswith("D:"):
                year_str = creation[2:6]
                if year_str.isdigit():
                    metadata["year"] = int(year_str)

        except Exception as e:
            logger.warning(f"Could not extract metadata from {pdf_path.name}: {e}")

        return metadata

    def calculate_quality_metrics(
        self, text: str, sections: List[Dict[str, Any]], page_count: int
    ) -> Dict[str, Any]:
        """
        Calculate quality metrics for the extracted content.
        """
        total_chars = len(text)
        non_whitespace_chars = len(text.replace(" ", "").replace("\n", ""))
        coverage_ratio = (
            non_whitespace_chars / total_chars if total_chars > 0 else 0
        )

        tables_count = sum(len(section.get("tables", [])) for section in sections)
        figures_count = sum(len(section.get("figures", [])) for section in sections)

        quality = {
            "text_coverage_ratio": round(coverage_ratio, 3),
            "tables_extracted": tables_count,
            "figures_detected": figures_count,
            "notes": f"Extracted {total_chars} characters from {page_count} pages",
        }

        return quality

    def parse_pdf(self, pdf_path: Path, datasource_code: str = "SRC_DOCS_PIPELINE5") -> Dict[str, Any]:
        """
        Parse a single PDF file into structured format following ParsedDocument schema.
        """
        logger.info(f"Parsing PDF: {pdf_path.name}")

        text, page_count = self.extract_text_from_pdf(pdf_path)
        checksum = self.calculate_checksum(pdf_path)
        metadata = self.extract_metadata(pdf_path)
        sections = self.detect_sections(text, page_count)
        quality = self.calculate_quality_metrics(text, sections, page_count)

        document_code = f"doc_{pdf_path.stem}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        parsed_document = {
            "document_code": document_code,
            "title": pdf_path.stem.replace("_", " ").title(),
            "datasource_code": datasource_code,
            "language": "en",
            "ingestion_id": f"ing_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "created_utc": datetime.utcnow().isoformat() + "Z",
            "source": {
                "uri": str(pdf_path.absolute()),
                "filename": pdf_path.name,
                "mime_type": "application/pdf",
                "checksum_sha256": checksum,
                "page_count": page_count,
            },
            "metadata": metadata,
            "sections": sections,
            "extraction_constraints": {
                "skip_patterns": ["^table of contents$", "warranty", "legal notice", "page \\d+"],
                "max_chars_per_section": 10000,
                "allowed_entities": [
                    "Product","Subsystem","ComponentType","Component",
                    "ParameterSpec","Unit","SensorType","Sensor",
                    "FailureMode","RepairAction","MaintenanceTask","Tool","Consumable",
                    "Core","ProcessStep","RoutingDecision","State",
                    "Document","DataSource","AnomalyThreshold","DiagnosticRule",
                    "MachineMode","TestSpec","RatingPlate"
                ],
                "allowed_relations": [
                    "hasPart","instanceOf","hasSpec","hasUnit","measuredBy",
                    "affects","requiresAction","canCause",
                    "implements","uses","requiresTool","requiresConsumable",
                    "justifies","definedFrom","belongsTo",
                    "hasThreshold","appliesTo","diagnoses","targets",
                    "hasMode","hasSetpoint","appliesDuring","verifies","hasRatingPlate",
                    "dependsOn","controls","connectedTo"
                ]
            },
            "quality": quality,
        }

        # Validate against schema
        if _JSONSCHEMA_AVAILABLE:
            try:
                validate(instance=parsed_document, schema=self.schema)  # type: ignore[arg-type]
            except ValidationError as e:
                logger.warning(f"Schema validation failed for {pdf_path.name}: {getattr(e, 'message', e)}")
        else:
            logger.warning(
                "Skipping schema validation for %s because 'jsonschema' is not installed. "
                "Install it with 'pip install -r requirements.txt' to enable validation.",
                pdf_path.name,
            )

        logger.info(f"Successfully parsed {pdf_path.name}: {len(sections)} sections, {page_count} pages")
        return parsed_document

    def process_directory(self, output_dir: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Process all PDF files in the source directory.
        """
        pdf_files = list(self.source_dir.glob("*.pdf"))

        if not pdf_files:
            logger.warning(f"No PDF files found in {self.source_dir}")
            return []

        logger.info(f"Found {len(pdf_files)} PDF files to process")

        parsed_documents = []

        for pdf_file in pdf_files:
            try:
                parsed_doc = self.parse_pdf(pdf_file)
                parsed_documents.append(parsed_doc)

                if output_dir:
                    output_path = Path(output_dir)
                    output_path.mkdir(parents=True, exist_ok=True)

                    output_file = output_path / f"{parsed_doc['document_code']}.json"
                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(parsed_doc, f, indent=2, ensure_ascii=False)

                    logger.info(f"Saved parsed document to {output_file}")

            except Exception as e:
                logger.error(f"Failed to process {pdf_file.name}: {e}")
                continue

        logger.info(
            f"Successfully processed {len(parsed_documents)} out of {len(pdf_files)} PDF files"
        )

        return parsed_documents

    def save_batch(self, parsed_documents: List[Dict[str, Any]], output_file: str):
        """
        Save all parsed documents to a single JSON file.
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        batch = {
            "batch_id": f"batch_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "processed_utc": datetime.utcnow().isoformat() + "Z",
            "document_count": len(parsed_documents),
            "documents": parsed_documents,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(batch, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved batch of {len(parsed_documents)} documents to {output_path}")


def main():
    """
    Example usage of PDFReader module.
    """
    BASE_DIR = Path(__file__).resolve().parent.parent
    SOURCE_DIR = BASE_DIR / "Source"
    SCHEMA_PATH = BASE_DIR / "schemas" / "parsed_document.json"
    CONFIG_PATH = BASE_DIR / "config.yaml"
    OUTPUT_DIR = BASE_DIR / "output" / "parsed"

    # Load configuration if available
    config = {}
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    reader = PDFReader(source_dir=str(SOURCE_DIR), schema_path=str(SCHEMA_PATH), config=config)

    parsed_docs = reader.process_directory(output_dir=str(OUTPUT_DIR))

    reader.save_batch(
        parsed_documents=parsed_docs,
        output_file=str(OUTPUT_DIR / "batch_all.json")
    )

    print(f"\n✅ Processing complete! Parsed {len(parsed_docs)} documents.")
    print(f"📁 Individual files saved to: {OUTPUT_DIR}")
    print(f"📦 Batch file: {OUTPUT_DIR / 'batch_all.json'}")


if __name__ == "__main__":
    main()
