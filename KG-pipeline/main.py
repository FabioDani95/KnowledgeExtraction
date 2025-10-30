from pathlib import Path
import sys
import json
import yaml

# === Make project root importable ===
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))  # ensure 'modules' is importable

# Optional but recommended: assert modules package exists
MODULES_DIR = BASE_DIR / "modules"
if not MODULES_DIR.exists():
    raise FileNotFoundError(f"Missing 'modules' directory at: {MODULES_DIR}")

# Import AFTER sys.path manipulation
try:
    from modules.pdf_reader import PDFReader
    from modules.neural_extractor import NeuralExtractor
except Exception as e:
    # Print helpful diagnostics, then raise
    print("⛔ Import failed for 'modules.pdf_reader.PDFReader'")
    print(f"   BASE_DIR = {BASE_DIR}")
    print(f"   sys.path[0] = {sys.path[0]}")
    raise

def main():
    # Use paths relative to the repo root to avoid surprises
    SOURCE_DIR = BASE_DIR / "Source"
    SCHEMA_PATH = BASE_DIR / "schemas" / "parsed_document.json"
    NEURAL_SCHEMA_PATH = BASE_DIR / "schemas" / "neural_extraction.json"
    CONFIG_PATH = BASE_DIR / "config.yaml"
    OUTPUT_BASE = BASE_DIR / "output"
    PDF_OUTPUT_DIR = OUTPUT_BASE / "test_parsed"
    NEURAL_OUTPUT_DIR = OUTPUT_BASE / "neural_extraction"

    PDF_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Sanity checks
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"Source directory not found: {SOURCE_DIR}")
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Schema file not found: {SCHEMA_PATH}")

    # Load configuration
    config = {}
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        print(f"• Loaded config from: {CONFIG_PATH}")
        if config.get("filtering", {}).get("enabled"):
            print("  → Content filtering: ENABLED")
        else:
            print("  → Content filtering: DISABLED")
    else:
        print(f"⚠️  Config file not found: {CONFIG_PATH}")
        print("  → Using default settings (no filtering)")

    # Show where we are writing
    print(f"\n• Using SOURCE_DIR  : {SOURCE_DIR}")
    print(f"• Using SCHEMA_PATH : {SCHEMA_PATH}")
    print(f"• Using PDF_OUTPUT  : {PDF_OUTPUT_DIR}")

    reader = PDFReader(source_dir=str(SOURCE_DIR), schema_path=str(SCHEMA_PATH), config=config)

    pdf_files = sorted(SOURCE_DIR.glob("*.pdf"))
    print(f"\n▶ Found {len(pdf_files)} PDF file(s) ready for parsing.")
    if pdf_files:
        for pdf in pdf_files:
            print(f"   - {pdf.name}")

    print("\n▶ Parsing PDFs...")
    parsed_docs = reader.process_directory(output_dir=str(PDF_OUTPUT_DIR))

    if parsed_docs:
        stats_rows = []
        for doc in parsed_docs:
            source_info = doc.get("source", {})
            sections = doc.get("sections", [])
            page_count = source_info.get("page_count", 0)
            char_count = sum(len(section.get("text") or "") for section in sections)
            stats_rows.append(
                (
                    source_info.get("filename", "N/A"),
                    page_count,
                    char_count,
                )
            )

        headers = ("Filename", "Pages", "Characters")
        col_widths = [
            max(len(str(row[idx])) for row in stats_rows + [headers]) for idx in range(len(headers))
        ]

        def format_row(row):
            return " | ".join(str(value).ljust(col_widths[idx]) for idx, value in enumerate(row))

        print("\n📊 Document statistics:")
        print(f"   {format_row(headers)}")
        print(f"   {'-+-'.join('-' * width for width in col_widths)}")
        for row in stats_rows:
            print(f"   {format_row(row)}")

        batch_file = PDF_OUTPUT_DIR / "test_batch.json"
        if config.get("output", {}).get("batch_file", True):
            reader.save_batch(parsed_documents=parsed_docs, output_file=str(batch_file))
            print(f"   Batch file: {batch_file}")

        print(f"\n✅ Parsed {len(parsed_docs)} document(s).")
        print(f"   Per-document JSON written to: {PDF_OUTPUT_DIR}")
        print(f"   Batch file: {batch_file}")

    else:
        print("\n⚠️  No PDFs found or parsing produced no output.")
        print("   Check that there is at least one .pdf (lowercase) in the Source/ folder.")

    # === Neural extraction module ===
    neural_config = config.get("neural_extractor", {}) or {}
    if neural_config.get("enabled", False):
        if not parsed_docs:
            print("\nℹ️  Neural extractor enabled but no parsed documents available; skipping.")
        elif not NEURAL_SCHEMA_PATH.exists():
            print(f"\n⚠️  Neural extraction schema not found: {NEURAL_SCHEMA_PATH}")
        else:
            print("\n▶ Running neural extraction...")
            extractor = NeuralExtractor(schema_path=str(NEURAL_SCHEMA_PATH), config=config)
            NEURAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            neural_results = extractor.run_batch(parsed_documents=parsed_docs, output_dir=str(NEURAL_OUTPUT_DIR))

            if neural_results:
                neural_batch_file = NEURAL_OUTPUT_DIR / "neural_batch.json"
                #extractor.save_batch(neural_results, str(neural_batch_file))  # Disabilitato: salvataggio batch non voluto, solo output individuali.
                print(f"\n✅ Neural extraction complete for {len(neural_results)} document(s).")
                print(f"   Neural outputs: {NEURAL_OUTPUT_DIR}")
                print(f"   Batch file: {neural_batch_file}")
            else:
                print("\n⚠️  Neural extractor did not produce any outputs. Check logs for details.")
    else:
        print("\nℹ️  Neural extractor disabled in configuration. Skipping module 2.")

if __name__ == "__main__":
    main()
