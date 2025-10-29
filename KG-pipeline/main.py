from pathlib import Path
import sys
import json

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
    OUTPUT_DIR = BASE_DIR / "output" / "test_parsed"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Sanity checks
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"Source directory not found: {SOURCE_DIR}")
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Schema file not found: {SCHEMA_PATH}")

    # Show where we are writing
    print(f"\n• Using SOURCE_DIR  : {SOURCE_DIR}")
    print(f"• Using SCHEMA_PATH : {SCHEMA_PATH}")
    print(f"• Using OUTPUT_DIR  : {OUTPUT_DIR}")

    reader = PDFReader(source_dir=str(SOURCE_DIR), schema_path=str(SCHEMA_PATH))

    print("\n▶ Parsing PDFs...")
    parsed_docs = reader.process_directory(output_dir=str(OUTPUT_DIR))

    if parsed_docs:
        batch_file = OUTPUT_DIR / "test_batch.json"
        reader.save_batch(parsed_documents=parsed_docs, output_file=str(batch_file))

        print(f"\n✅ Parsed {len(parsed_docs)} document(s).")
        print(f"   Per-document JSON written to: {OUTPUT_DIR}")
        print(f"   Batch file: {batch_file}")

        # Preview first document (safe truncation)
        print("\n--- Preview of first parsed document ---")
        print(json.dumps(parsed_docs[0], indent=2, ensure_ascii=False)[:1200], "...")
    else:
        print("\n⚠️  No PDFs found or parsing produced no output.")
        print("   Check that there is at least one .pdf (lowercase) in the Source/ folder.")

if __name__ == "__main__":
    main()
