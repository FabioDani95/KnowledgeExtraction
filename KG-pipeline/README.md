# Knowledge Extraction Pipeline

A modular semantic extraction system for building structured knowledge graphs from technical documentation using a hybrid symbolic-neural approach.

## Overview

The Knowledge Extraction Pipeline is an advanced system designed to extract structured knowledge from technical manuals and transform it into semantically rich knowledge graphs. The system employs a **modular vertical pipeline architecture** that processes different categories of technical information independently, then merges them into a unified knowledge graph.

### Key Features

- **Hybrid Extraction**: Combines rule-based symbolic parsing with AI-powered neural extraction
- **Modular Architecture**: Independent vertical pipelines for different document categories
- **Three-Stage Pipeline**: Symbolic extraction → Neural enrichment → Intelligent merging
- **Deterministic IDs**: Namespace-based entity identification for consistency
- **Quality Assurance**: Multi-level validation with confidence scoring
- **Semantic Correction**: Automatic relationship validation and repair
- **Provenance Tracking**: Complete traceability from source document to extracted entities

---

## Architecture

### System Components

The pipeline consists of three main components:

1. **Symbolic Orchestrator** (`Symbolic_orchestrator.py`)
   - Rule-based pattern matching and extraction
   - Structured data parsing (tables, specifications)
   - Section identification and text preprocessing
   - Generates initial parsed data structures

2. **Neural Extractor** (`Neural_extraction.py`)
   - AI-powered entity and relationship extraction
   - OpenAI GPT-based processing with specialized prompts
   - Three profile specializations (product, operations, troubleshooting)
   - JSON repair and retry mechanisms with progressive degradation

3. **Knowledge Graph Merger** (`merge_kgs.py`)
   - Intelligent deduplication across sub-graphs
   - Conflict resolution with configurable strategies
   - Relationship validation and semantic corrections
   - Unified graph generation with provenance tracking

### Vertical Pipeline Approach

Unlike traditional monolithic extraction systems, this architecture implements **specialized vertical pipelines** for different document categories:

#### Why Vertical Pipelines?

Technical manuals exhibit significant structural and linguistic variations:
- Different formats (tabular vs. descriptive)
- Varying levels of technical detail
- Fragmented information across sections

A single unified pipeline would produce:
- ❌ Partial or redundant extractions
- ❌ Entity classification errors
- ❌ Inconsistent relationships
- ❌ Difficult validation

Vertical pipelines provide:
- ✅ **Semantic precision**: Each LLM receives homogeneous context with calibrated prompts
- ✅ **Incrementality**: Each pipeline can be executed and validated independently
- ✅ **Scalability**: Easy addition of new pipelines for other categories
- ✅ **Reusability**: Pipelines are composable based on available datasets

### Extraction Profiles

The system currently supports three specialized extraction profiles:

#### 1. Product & Technical Data (`product_technical`)
**Purpose**: Extract physical components and technical specifications

**Entity Types**:
- `Product`: Coffee machines, appliances
- `Component`: Physical parts (pumps, thermoblocks, sensors)
- `ComponentType`: Generic component categories
- `ParameterSpec`: Technical specifications (voltage, pressure, temperature)
- `Unit`: Measurement units

**Relationships**: `hasPart`, `instanceOf`, `hasSpec`, `hasUnit`, `connectedTo`

**Input**: Service manuals (chapters 1-4), technical specifications, parts lists

#### 2. Operation Modes (`operation_modes`)
**Purpose**: Extract operational sequences and machine states

**Entity Types**:
- `MachineMode`: Operating modes (brewing, descaling, maintenance)
- `State`: Machine states (ready, heating, error)
- `ProcessStep`: Sequential operational steps
- `MaintenanceTask`: Scheduled maintenance procedures
- `Tool`: Required tools for operations

**Relationships**: `precedes`, `appliesDuring`, `requiresTool`, `transitions`

**Input**: User manuals, operational guides, maintenance procedures

#### 3. Troubleshooting & Diagnostics (`troubleshooting`)
**Purpose**: Extract failure modes and corrective actions

**Entity Types**:
- `FailureMode`: Problems and failure conditions
- `RepairAction`: Corrective procedures
- `DiagnosticRule`: Diagnostic logic
- `Tool`: Required repair tools
- `Consumable`: Replacement parts

**Relationships**: `affects`, `mitigatedBy`, `requiresTool`, `requiresConsumable`

**Input**: Troubleshooting tables, error codes, diagnostic procedures

**Special Features**: Semantic correction engine ensures proper relationship patterns (e.g., `FailureMode --mitigatedBy--> RepairAction --requiresTool--> Tool`)

---

## Project Structure

```
KG-pipeline/
├── src/                                # Source code
│   ├── Symbolic_orchestrator.py        # Symbolic extraction coordinator
│   ├── Neural_extraction.py            # Neural extraction with LLM
│   ├── merge_kgs.py                    # Knowledge graph merger
│   └── pipelines/                      # Profile-specific parsers
│       ├── product_technical/
│       ├── operation_modes/
│       └── troubleshooting/
│
├── output/                             # Generated outputs
│   ├── partial/                        # Symbolic extraction results
│   │   ├── product_technical/
│   │   ├── operation_modes/
│   │   └── troubleshooting/
│   ├── neural_extraction/              # Neural extraction results
│   │   ├── product_technical/
│   │   │   ├── kg.json                 # Knowledge graph
│   │   │   ├── quality.json            # Quality report
│   │   │   └── raw/                    # Raw AI outputs (debug)
│   │   ├── operation_modes/
│   │   └── troubleshooting/
│   └── merged_kg/                      # Final merged knowledge graphs
│       └── merged_kg_*.json
│
├── schemas/                            # JSON schemas
│   └── neural_extraction.json          # Universal schema definition
│
├── source/                             # Input documents (organized by profile)
├── config.yaml                         # System configuration
└── requirements.txt                    # Python dependencies
```

---

## Installation

### Prerequisites

- Python 3.8+
- OpenAI API key

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd KnowledgeExtraction/KG-pipeline
   ```

2. **Create virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   # or
   .venv\Scripts\activate      # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure OpenAI API key**
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

---

## Usage

### Complete Pipeline Execution

Run the full three-stage pipeline for all profiles:

```bash
# Stage 1: Symbolic extraction
python3 src/Symbolic_orchestrator.py --pipeline all

# Stage 2: Neural extraction
python3 src/Neural_extraction.py --profile all

# Stage 3: Merge knowledge graphs
python3 src/merge_kgs.py
```

### Individual Pipeline Execution

#### Symbolic Extraction
```bash
# All profiles
python3 src/Symbolic_orchestrator.py --pipeline all

# Single profile
python3 src/Symbolic_orchestrator.py --pipeline product_technical
python3 src/Symbolic_orchestrator.py --pipeline operation_modes
python3 src/Symbolic_orchestrator.py --pipeline troubleshooting
```

#### Neural Extraction
```bash
# All profiles
python3 src/Neural_extraction.py --profile all

# Single profile
python3 src/Neural_extraction.py --profile product_technical
python3 src/Neural_extraction.py --profile operation_modes
python3 src/Neural_extraction.py --profile troubleshooting

# With verbose output
python3 src/Neural_extraction.py --profile all --verbose

# Dry-run mode (no API calls)
python3 src/Neural_extraction.py --profile all --dry-run
```

#### Knowledge Graph Merging
```bash
# Default merge (all validations enabled)
python3 src/merge_kgs.py

# With custom configuration
python3 src/merge_kgs.py --config custom_merge_config.yaml
```

---

## Configuration

The system is configured via `config.yaml`:

### Neural Extractor Settings

```yaml
neural_extractor:
  enabled: true
  provider: "openai"
  model: "gpt-4o-mini"
  api_key_env: "OPENAI_API_KEY"
  temperature: 0.2
  max_output_tokens: 1500
  request_timeout: 60
  max_tokens_per_chunk: 1000
  retry_on_failure: true
  save_raw_outputs: true
```

### Profile Configuration

```yaml
neural_extraction_profiles:
  product_technical:
    input_glob: "output/partial/product_technical/*.json"
    output_dir: "output/neural_extraction/product_technical"
    class_map:
      - "Product"
      - "ComponentType"
      - "Component"
      - "ParameterSpec"
      - "Unit"
    relation_map:
      - "hasPart"
      - "instanceOf"
      - "hasSpec"
      - "hasUnit"
```

---

## Technical Features

### 1. Deterministic Entity IDs

Entities are assigned consistent namespace-based identifiers:

| Entity Type | Namespace | Example Input | Generated ID |
|-------------|-----------|---------------|--------------|
| Product | `ns:Product/` | "Citiz" | `ns:Product/citiz` |
| ComponentType | `ns:ComponentType/` | "Thermoblock" | `ns:ComponentType/thermoblock` |
| Component | `ns:Component/` | "Pump CP4" | `ns:Component/pump_cp4` |
| MachineMode | `ns:Mode/` | "Descaling Mode" | `ns:Mode/descaling_mode` |
| State | `ns:State/` | "Ready to brew" | `ns:State/ready_to_brew` |
| FailureMode | `ns:FM/` | "No water flow" | `ns:FM/no_water_flow` |
| RepairAction | `ns:RA/` | "Replace pump" | `ns:RA/replace_pump` |
| Tool | `ns:Tool/` | "Torque Wrench" | `ns:Tool/torque_wrench` |

### 2. Robust JSON Parsing

The neural extractor includes sophisticated JSON repair mechanisms:

- Automatic recovery from malformed JSON
- Handling of typographic quotes (`"` → `"`)
- Missing comma insertion
- Trailing comma removal
- Selective extraction of entity/relation arrays from partial responses
- Code fence removal (` ```json ... ``` `)

### 3. Intelligent Retry Strategy

Progressive degradation on extraction failures:

```
Attempt 1: Full prompt, temperature=0.2 (from config)
    ↓ (on failure)
Attempt 2: Simplified prompt, temperature=0.1
    ↓ (on failure)
Attempt 3: Simplified prompt, temperature=0.0 (deterministic)
```

### 4. Advanced Deduplication

Multi-level deduplication across the pipeline:

- **Entity deduplication**: By (type, normalized_name)
- **Relation deduplication**: By (type, from_ref, to_ref)
- **Cross-profile deduplication**: During KG merge phase
- Consolidation based on maximum confidence score

### 5. Semantic Correction (Troubleshooting)

Automatic relationship pattern enforcement:

**Before** (incorrect):
```
FailureMode("No flow") --mitigatedBy--> Tool("Wrench")
```

**After** (corrected):
```
FailureMode("No flow") --mitigatedBy--> RepairAction("Use Wrench")
RepairAction("Use Wrench") --requiresTool--> Tool("Wrench")
```

### 6. Text Normalization

Consistent text processing:
- Synonym unification ("Descaling mode" = "Descaling Mode")
- Consistent slugification for IDs
- Multiple space removal
- Special character handling

---

## Output Schema

### Knowledge Graph Format

```json
{
  "document_code": "KG_PRODUCT_TECHNICAL",
  "ingestion_id": "uuid",
  "extraction_version": "neural_v1.1",
  "datasource_code": "NEURAL_EXTRACTION",
  "extractor": {
    "model": "gpt-4o-mini",
    "prompt_id": "neural_extraction_v1_hardened",
    "temperature": 0.2,
    "max_tokens": 1500
  },
  "allowed_types": ["Product", "Component", ...],
  "allowed_relations": ["hasPart", "instanceOf", ...],
  "entities": [
    {
      "id": "ns:ComponentType/thermoblock",
      "type": "ComponentType",
      "name": "Thermoblock",
      "confidence": 0.94,
      "attributes": {...}
    }
  ],
  "relations": [
    {
      "type": "hasPart",
      "from_ref": "ns:Product/citiz",
      "to_ref": "ns:ComponentType/thermoblock",
      "confidence": 0.88
    }
  ],
  "provenance": {
    "overall_confidence": 0.91,
    "sections_used": [...],
    "source_documents": [...]
  }
}
```

### Quality Report Format

```json
{
  "json_valid": true,
  "entities_count": 42,
  "relations_count": 38,
  "warnings": [],
  "overall_confidence": 0.91,
  "avg_entity_confidence": 0.92,
  "avg_relation_confidence": 0.90,
  "entity_type_distribution": {
    "Component": 15,
    "ComponentType": 12,
    "ParameterSpec": 10,
    "Unit": 5
  },
  "relation_type_distribution": {
    "hasPart": 20,
    "instanceOf": 15,
    "hasSpec": 3
  },
  "profile": "product_technical",
  "generated_at": "2025-11-05T10:30:00Z"
}
```

---

## Quality Assurance

### Validation Layers

1. **Schema Validation**: All outputs validated against `neural_extraction.json`
2. **Confidence Scoring**: Each entity and relationship includes confidence scores
3. **Source Tracing**: Complete provenance from source document to extracted elements
4. **Warning System**: Automatic detection of anomalies, duplicates, and inconsistencies
5. **Relationship Validation**: Semantic checks for proper relationship patterns

### Metrics

- **JSON validity**: Structural correctness
- **Entity/Relation counts**: Extraction completeness
- **Confidence averages**: Overall extraction quality
- **Type distributions**: Coverage analysis
- **Warning reports**: Issue identification

---

## Debugging

### Enable Debug Mode

1. **Save raw AI outputs**:
   ```yaml
   neural_extractor:
     save_raw_outputs: true
   ```

2. **Run with verbose logging**:
   ```bash
   python3 src/Neural_extraction.py --profile all --verbose
   ```

3. **Inspect raw outputs**:
   ```bash
   ls output/neural_extraction/*/raw/
   cat output/neural_extraction/product_technical/raw/chunk_*.json
   ```

### Common Issues

- **API timeouts**: Increase `request_timeout` in config
- **Token limits**: Reduce `max_tokens_per_chunk`
- **Low quality extractions**: Lower `temperature` or adjust prompts
- **Missing relationships**: Check relationship validation rules in profile config

---

## Performance Considerations

- **API Costs**: Estimated 10-20k tokens per document
- **Temperature**: Lower values (0.1-0.3) ensure consistency
- **Chunking**: Text divided into ~1000 token blocks to avoid API limits
- **Retry Logic**: Automatic retry with degradation on failures
- **Parallel Execution**: Profiles can be processed independently

---

## Frontend Interface

The pipeline now includes a **lightweight web frontend** for easy interaction with the knowledge extraction system.

### Features

- **📤 Document Upload**: Drag-and-drop interface for PDF files with target folder selection
- **⚙️ Pipeline Execution**: One-click pipeline start with real-time progress tracking
- **🕸️ Interactive Graph Visualization**: Explore the knowledge graph with Cytoscape.js
  - Multiple layouts (force-directed, circle, grid, hierarchical)
  - Advanced filtering by node/edge types and confidence
  - Full-text search across entities
  - Detail inspection panels
- **📈 Statistics Dashboard**: KPIs, quality metrics, and coverage targets
- **💾 Export Options**: PNG, JSON, and CSV exports

### Quick Start

```bash
# Navigate to frontend directory
cd frontend

# Start web server (choose one)
python3 -m http.server 8080
# or
npx http-server -p 8080

# Open browser
open http://localhost:8080
```

**Architecture:**
- `index.html` - Main UI structure
- `js/api.js` - Backend API adapter (mock/real modes)
- `js/kg-viz.js` - Graph visualization (Cytoscape.js wrapper)
- `js/app.js` - Application orchestration
- `styles.css` - Responsive styling
- `config.json` - Configuration (endpoints, colors, types)

For detailed frontend documentation, see [`frontend/README.md`](frontend/README.md).

---

## Roadmap

- [x] Modular vertical pipeline architecture
- [x] Three-profile extraction system (product, operations, troubleshooting)
- [x] Intelligent retry with prompt simplification
- [x] Deterministic namespace-based entity IDs
- [x] Robust JSON parsing and repair
- [x] Advanced deduplication (entities and relations)
- [x] Semantic correction for troubleshooting relationships
- [x] Symbolic knowledge graph merger with validation
- [x] **Interactive web frontend for visualization and exploration**
- [ ] Additional pipeline profiles (repair structure, testing)
- [ ] Support for alternative AI providers (Anthropic, Azure, Google)
- [ ] Fuzzy matching for cross-profile entity alignment
- [ ] UCUM unit normalization
- [ ] Export formats (RDF, Neo4j, GraphML)
- [ ] Backend REST API for frontend integration

---

## Version History

### v1.1 - Hardening Update (2025-11-05)

**Fixed**:
- Deprecation warning `datetime.utcnow()` → `datetime.now(timezone.utc)`
- JSON parsing errors with automatic repair
- Semantically incorrect relationships in troubleshooting profile

**Added**:
- Deterministic namespace-based IDs (`ns:Type/slug`)
- `safe_json_parse()` with 3-level repair mechanism
- `extract_with_retry()` with automatic degradation (3 attempts)
- `deduplicate_entities()` with name normalization
- `deduplicate_relations()` by (type, from, to) key
- `fix_troubleshooting_semantics()` for correct relationship patterns
- `slugify()` for consistent naming
- Hardened prompts emphasizing valid JSON

**Changed**:
- `normalize_extraction()` now includes complete deduplication
- Extraction version: `neural_v1.0` → `neural_v1.1`
- Prompt ID: `neural_extraction_v1` → `neural_extraction_v1_hardened`
- More informative logging with truncated warnings

**Performance**:
- Reduced failed chunks via intelligent retry
- Deduplication reduces final KG size by ~15-25%

### v1.0 - Initial Release (2025-11-03)
- Base neural extraction system
- Support for 3 profiles (product_technical, operation_modes, troubleshooting)
- OpenAI GPT integration
- Validation and quality metrics

---

## License

Research project for semantic knowledge extraction from technical documentation.

---

## Contributing

This project follows an incremental development approach. To contribute:

1. Follow the `BasePipeline` pattern for new pipelines
2. Add comprehensive tests in `tests/`
3. Document prompts and strategies in code
4. Test on real technical documents
5. Submit pull requests with clear descriptions

---

**Version**: 1.1
**Status**: Production - Active Development
**Last Updated**: November 2025
