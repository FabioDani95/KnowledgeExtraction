"""
PDF Reader Module
Parses PDF documents from the source directory and structures them according to the parsed_document schema.
"""

import os
import json
import hashlib
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFReader:
    """
    Handles PDF document parsing and conversion to structured format.
    """

    def __init__(self, source_dir: str, schema_path: str):
        """
        Initialize the PDF reader with source directory and schema.

        Args:
            source_dir: Path to directory containing PDF files
            schema_path: Path to the parsed_document.json schema file
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

        logger.info(f"PDFReader initialized with source: {self.source_dir}")

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
        sections.append(current_section)
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
    OUTPUT_DIR = BASE_DIR / "output" / "parsed"

    reader = PDFReader(source_dir=str(SOURCE_DIR), schema_path=str(SCHEMA_PATH))

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
