from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import yaml

if __package__ in (None, ""):
    # Allow running the script directly: python src/orchestrator.py
    sys.path.append(str(Path(__file__).resolve().parent))
    from pipelines.product_technical.symbolic_parser import (  # type: ignore
        ProductTechnicalSymbolicParser,
    )
else:
    from .pipelines.product_technical.symbolic_parser import (
        ProductTechnicalSymbolicParser,
    )


PIPELINE_REGISTRY: Dict[str, Dict[str, object]] = {
    "product_technical": {
        "symbolic": ProductTechnicalSymbolicParser,
    },
}


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def resolve_documents(
    parser: ProductTechnicalSymbolicParser, document_args: Optional[List[str]]
) -> Optional[List[Path]]:
    if not document_args:
        return None

    doc_paths: List[Path] = []
    for doc in document_args:
        candidate = Path(doc)
        if not candidate.is_absolute():
            candidate = parser.source_dir / candidate
        if not candidate.exists():
            raise FileNotFoundError(f"Document not found: {candidate}")
        doc_paths.append(candidate)
    return doc_paths


def run_symbolic_pipeline(
    root_dir: Path,
    pipeline_name: str,
    config: dict,
    document_args: Optional[List[str]] = None,
) -> List[Path]:
    pipeline_config = (
        config.get("pipelines", {}).get(pipeline_name)
        if config.get("pipelines")
        else None
    )
    if not pipeline_config:
        raise ValueError(f"Pipeline '{pipeline_name}' not configured in config.yaml")
    if not pipeline_config.get("enabled", True):
        raise ValueError(f"Pipeline '{pipeline_name}' is disabled in configuration")

    symbolic_cls = PIPELINE_REGISTRY[pipeline_name]["symbolic"]
    parser = symbolic_cls(root_dir, pipeline_config, config)
    docs = resolve_documents(parser, document_args)
    return parser.run(docs)


def summarize_symbolic_outputs(output_paths: List[Path]) -> Optional[dict]:
    if not output_paths:
        return None

    stats = {
        "documents": 0,
        "pages": 0,
        "sections": 0,
        "figures": 0,
        "tables": 0,
        "warnings": 0,
        "avg_text_coverage": None,
    }
    coverage_values: List[float] = []

    for path in output_paths:
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:  # pragma: no cover - defensive logging
            logging.warning("Unable to load %s for summary: %s", path, exc)
            continue

        stats["documents"] += 1
        stats["pages"] += int(data.get("source", {}).get("page_count", 0))

        sections = data.get("sections") or []
        stats["sections"] += len(sections)

        for section in sections:
            stats["figures"] += len(section.get("figures") or [])
            stats["tables"] += len(section.get("tables") or [])
            warning_entries = section.get("warnings") or []
            if warning_entries:
                stats["warnings"] += len(warning_entries)
            elif (section.get("role") or "").lower() == "warning":
                stats["warnings"] += 1

        quality = data.get("quality") or {}
        coverage = quality.get("text_coverage_ratio")
        if isinstance(coverage, (int, float)):
            coverage_values.append(float(coverage))

    if coverage_values:
        stats["avg_text_coverage"] = sum(coverage_values) / len(coverage_values)

    return stats


def configure_logging(config: dict, verbose: bool) -> None:
    log_config = config.get("logging", {}) or {}
    level_name = log_config.get("level", "INFO")
    level = logging.getLevelName(level_name.upper())
    if isinstance(level, int):
        logging.basicConfig(
            level=logging.DEBUG if verbose else level,
            format=log_config.get(
                "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            ),
        )
    else:
        logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Knowledge Extraction Pipeline Orchestrator"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to configuration YAML file (default: config.yaml next to this script)",
    )
    parser.add_argument(
        "--pipeline",
        choices=list(PIPELINE_REGISTRY.keys()),
        default="product_technical",
        help="Pipeline name to execute (default: product_technical)",
    )
    parser.add_argument(
        "--stage",
        choices=["symbolic"],
        default="symbolic",
        help="Pipeline stage to run (currently symbolic only)",
    )
    parser.add_argument(
        "--document",
        dest="documents",
        action="append",
        help="Optional specific document(s) to process (relative or absolute path)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging (DEBUG level)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    if args.config is None:
        config_path = project_root / "config.yaml"
    else:
        config_path = args.config
        if not config_path.is_absolute():
            config_path = (Path.cwd() / config_path).resolve()
    if not config_path.exists():
        logging.basicConfig(level=logging.ERROR)
        logging.error("Configuration file not found: %s", config_path)
        return 1

    config = load_config(config_path)
    configure_logging(config, args.verbose)

    root_dir = config_path.parent
    try:
        if args.stage == "symbolic":
            outputs = run_symbolic_pipeline(
                root_dir=root_dir,
                pipeline_name=args.pipeline,
                config=config,
                document_args=args.documents,
            )
        else:
            raise ValueError(f"Unsupported stage: {args.stage}")
    except Exception as exc:
        logging.exception("Pipeline execution failed: %s", exc)
        return 2

    if not outputs:
        logging.warning("No output generated for pipeline '%s'", args.pipeline)
        return 0

    logging.info("Generated %d file(s):", len(outputs))
    for path in outputs:
        logging.info(" - %s", path)

    summary = summarize_symbolic_outputs(outputs)
    if summary:
        print("\nSymbolic extraction summary")
        print(f" - Documents processed: {summary['documents']}")
        print(f" - Pages covered: {summary['pages']}")
        print(f" - Sections parsed: {summary['sections']}")
        print(f" - Tables detected: {summary['tables']}")
        print(f" - Figures detected: {summary['figures']}")
        print(f" - Warning sections: {summary['warnings']}")
        if summary["avg_text_coverage"] is not None:
            pct = summary['avg_text_coverage'] * 100
            print(f" - Avg text coverage: {pct:.2f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
