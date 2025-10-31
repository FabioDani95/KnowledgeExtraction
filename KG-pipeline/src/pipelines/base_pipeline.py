from __future__ import annotations

import abc
import logging
from pathlib import Path
from typing import Iterable, List, Optional


class BasePipeline(abc.ABC):
    """Shared utilities for pipeline stages."""

    def __init__(
        self,
        root_dir: Path,
        pipeline_name: str,
        pipeline_config: dict,
        global_config: dict,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.pipeline_name = pipeline_name
        self.pipeline_config = pipeline_config or {}
        self.global_config = global_config or {}
        self.logger = logging.getLogger(f"pipeline.{pipeline_name}")

    @property
    def source_dir(self) -> Path:
        rel_path = self.pipeline_config.get("source_dir")
        if not rel_path:
            raise ValueError(
                f"Pipeline '{self.pipeline_name}' missing 'source_dir' configuration"
            )
        return self.root_dir / rel_path

    @property
    def output_dir(self) -> Path:
        rel_path = self.pipeline_config.get("output_dir")
        if not rel_path:
            raise ValueError(
                f"Pipeline '{self.pipeline_name}' missing 'output_dir' configuration"
            )
        return self.root_dir / rel_path

    @property
    def document_glob(self) -> str:
        return self.pipeline_config.get("document_glob", "*.pdf")

    def discover_documents(self) -> List[Path]:
        """Return the list of input documents for the pipeline."""
        if not self.source_dir.exists():
            raise FileNotFoundError(
                f"Source directory not found for pipeline '{self.pipeline_name}': "
                f"{self.source_dir}"
            )
        documents = sorted(self.source_dir.glob(self.document_glob))
        return documents

    def ensure_output_dir(self) -> Path:
        """Ensure the output directory exists and return it."""
        out_dir = self.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir


class BaseSymbolicParser(BasePipeline, abc.ABC):
    """Base class for symbolic parsers."""

    @abc.abstractmethod
    def parse_document(self, document_path: Path) -> dict:
        """Parse a document and return the symbolic representation."""

    def run(
        self,
        document_paths: Optional[Iterable[Path]] = None,
    ) -> List[Path]:
        """
        Execute the parser and persist results.

        Returns:
            List of output JSON paths generated.
        """
        output_files: List[Path] = []
        self.ensure_output_dir()
        docs = list(document_paths) if document_paths else self.discover_documents()
        if not docs:
            self.logger.warning(
                "No documents found for pipeline '%s' (pattern: %s)",
                self.pipeline_name,
                self.document_glob,
            )
            return output_files

        for doc_path in docs:
            try:
                result = self.parse_document(doc_path)
            except Exception as exc:
                self.logger.exception("Failed parsing %s: %s", doc_path.name, exc)
                continue

            output_path = self.output_dir / f"{doc_path.stem}.parsed.json"
            self._write_output(result, output_path)
            output_files.append(output_path)
            self.logger.info(
                "Generated symbolic JSON for '%s' → %s",
                doc_path.name,
                output_path,
            )
        return output_files

    def _write_output(self, data: dict, output_path: Path) -> None:
        import json

        output_settings = self.global_config.get("output", {})
        indent = output_settings.get("indent", 2)
        ensure_ascii = output_settings.get("ensure_ascii", False)
        with output_path.open("w", encoding="utf-8") as fp:
            json.dump(data, fp, indent=indent, ensure_ascii=ensure_ascii)
