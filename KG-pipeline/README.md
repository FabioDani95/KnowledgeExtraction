# Knowledge Extraction Pipeline - Modular Architecture

Sistema modulare per l'estrazione semantica di conoscenza da manuali tecnici e la costruzione di knowledge graphs strutturati.

## 🎯 Paradigma Architetturale

Questo sistema implementa un **ecosistema di pipeline verticali** che sostituisce l'approccio monolitico tradizionale. Ogni pipeline è specializzata per una categoria documentale omogenea e progettata per popolare un sottoinsieme specifico dello schema JSON di riferimento.

### Motivazione

I manuali tecnici presentano forti variazioni strutturali e linguistiche:
- Alcuni sono tabellari, altri descrittivi
- Diversi livelli di dettaglio tecnico
- Frammentazione delle informazioni

Una pipeline unica produce:
- ❌ Estrazioni parziali o ridondanti
- ❌ Errori nella classificazione delle entità
- ❌ Relazioni non coerenti
- ❌ Difficoltà di validazione

### Soluzione: Pipeline Verticali

✅ **Precisione semantica**: Ogni LLM riceve contesto omogeneo e prompt calibrato
✅ **Incrementalità**: Ogni pipeline può essere eseguita e validata indipendentemente
✅ **Scalabilità**: Possibilità di aggiungere nuove pipeline per altre categorie
✅ **Riutilizzabilità**: Le pipeline sono combinabili in base al dataset disponibile

---

## 📂 Struttura del Progetto

```
KG-pipeline/
├── source/                            # Input documenti organizzati per pipeline
│   ├── product_technical/             # Pipeline 1: Dati prodotto/tecnici
│   ├── operation_modes/               # Pipeline 2: Modalità operative
│   ├── troubleshooting/               # Pipeline 3: Diagnostica
│   ├── repair_structure/              # Pipeline 4: Riparazione e parti
│   └── testing/                       # Pipeline 5: Test e verifica
│
├── src/                               # Codice sorgente
│   ├── core/                          # Moduli riusabili
│   │   ├── pdf_reader.py              # Lettura e parsing PDF
│   │   ├── text_preprocessor.py       # Normalizzazione testo
│   │   ├── schema_validator.py        # Validazione JSON schema
│   │   └── llm_client.py              # Client OpenAI generico
│   │
│   ├── pipelines/                     # Pipeline verticali
│   │   ├── base_pipeline.py           # Classe astratta base
│   │   │
│   │   ├── product_technical/         # Pipeline 1
│   │   │   ├── symbolic_parser.py     # Parser basato su regole
│   │   │   ├── neural_extractor.py    # Estrattore LLM-based
│   │   │   └── prompts.py             # Template prompt specifici
│   │   │
│   │   ├── operation_modes/           # Pipeline 2
│   │   ├── troubleshooting/           # Pipeline 3
│   │   ├── repair_structure/          # Pipeline 4
│   │   └── testing/                   # Pipeline 5
│   │
│   ├── merger/                        # Fusione JSON parziali
│   │   ├── graph_merger.py            # Merge entities/relations
│   │   ├── conflict_resolver.py       # Risoluzione conflitti
│   │   └── provenance_tracker.py      # Tracciamento provenienza
│   │
│   ├── orchestrator.py                # Coordinatore pipeline
│   ├── utils.py                       # Utilities generiche
│   └── logger.py                      # Logging configurabile
│
├── output/                            # Risultati estrazione
│   ├── partial/                       # JSON parziali per pipeline
│   │   ├── product_technical/
│   │   ├── operation_modes/
│   │   ├── troubleshooting/
│   │   ├── repair_structure/
│   │   └── testing/
│   ├── merged/                        # JSON finali fusi
│   └── logs/                          # Log esecuzione
│
├── schemas/                           # Schema JSON di riferimento
│   └── neural_extraction.json         # Schema universale
│
├── tests/                             # Test unitari e integrazione
│
├── config.yaml                        # Configurazione globale
├── main.py                            # Entry point
└── requirements.txt                   # Dipendenze Python
```

---

## 🔧 Le Cinque Pipeline Verticali

### 1️⃣ Product & Technical Data
**Input**: Capitoli 1-4 del service manual, specifiche tecniche
**Output**: Entità fisiche e parametri tecnici
**Entità**: `Product`, `Component`, `ComponentType`, `ParameterSpec`, `Unit`, `RatingPlate`
**Relazioni**: `hasPart`, `hasSpec`, `hasUnit`, `hasRatingPlate`, `instanceOf`
**Strumenti**: OCR + regex + LLM estrattivo per coppie componenti-specifiche

### 2️⃣ Operation & Machine Modes
**Input**: User manual, guide operative
**Output**: Sequenze operative e modalità macchina
**Entità**: `ProcessStep`, `MachineMode`, `MaintenanceTask`, `Tool`, `State`
**Relazioni**: `precedes`, `appliesDuring`, `requiresTool`
**Focus**: Sequenze di azioni e logiche operative

### 3️⃣ Troubleshooting & Diagnostics
**Input**: Tabelle errori, codici guasto, regole di riparazione
**Output**: Modalità di guasto e azioni correttive
**Entità**: `FailureMode`, `RepairAction`, `DiagnosticRule`, `Component`
**Relazioni**: `affects`, `mitigatedBy`, `requiresAction`, `requiresConsumable`
**Focus**: Diagnosi e risoluzione problemi

### 4️⃣ Repair & Parts Structure
**Input**: Sezioni disassembly, parts list
**Output**: Albero strutturale del prodotto e procedure di smontaggio
**Entità**: `ComponentType`, `Component`, `Tool`, `ProcessStep`
**Relazioni**: `hasPart`, `belongsTo`, `instanceOf`, `requiresTool`
**Focus**: Gerarchia componenti e procedure riparazione

### 5️⃣ Testing & Verification
**Input**: Procedure di test, specifiche di calibrazione
**Output**: Logiche di verifica e soglie accettazione
**Entità**: `TestSpec`, `AnomalyThreshold`, `ParameterSpec`, `Unit`
**Relazioni**: `verifies`, `hasSpec`, `hasUnit`
**Focus**: Collaudo e accettazione finale

---

## 🧩 Architettura dei Componenti

### BasePipeline (Template Method Pattern)

Ogni pipeline eredita da una classe astratta base che definisce il workflow standard:

```python
class BasePipeline(ABC):
    def execute(self):
        """Template method: orchestrazione standard"""
        documents = self.load_documents()
        parsed = self.run_symbolic_parsing(documents)
        extracted = self.run_neural_extraction(parsed)
        validated = self.validate_output(extracted)
        return validated

    @abstractmethod
    def run_symbolic_parsing(self, documents):
        """Parsing simbolico specifico della pipeline"""
        pass

    @abstractmethod
    def run_neural_extraction(self, parsed_data):
        """Estrazione neurale specifica della pipeline"""
        pass

    @abstractmethod
    def get_target_entities(self):
        """Ritorna subset di entità da estrarre"""
        pass
```

### Due Livelli di Estrazione per Pipeline

Ogni pipeline implementa:

1. **Symbolic Parser** (`symbolic_parser.py`)
   - Estrazione basata su pattern, regex, tabelle
   - Identificazione sezioni rilevanti
   - Pre-processing dominio-specifico
   - Output: Struttura intermedia (parsed data)

2. **Neural Extractor** (`neural_extractor.py`)
   - Estrazione LLM-based (OpenAI GPT)
   - Prompt calibrati sul contesto specifico
   - Generazione entità e relazioni
   - Output: JSON parziale conforme allo schema

### Orchestrator

Coordina l'esecuzione delle pipeline:

```python
class PipelineOrchestrator:
    def run_all(self, parallel=False):
        """Esegue tutte le pipeline"""
        results = []
        for pipeline in self.pipelines:
            result = pipeline.execute()
            results.append(result)

        merged = self.merger.merge(results)
        return merged

    def run_single(self, pipeline_name):
        """Esegue solo una pipeline specifica"""
        pipeline = self.get_pipeline(pipeline_name)
        return pipeline.execute()
```

### Graph Merger

Fonde i JSON parziali in un grafo coerente:

- **Deduplicazione**: Unisce entità identiche estratte da pipeline diverse
- **Conflict Resolution**: Risolve attributi contrastanti (es. valori diversi per stessa proprietà)
- **Provenance Tracking**: Mantiene traccia della pipeline di origine per ogni entità/relazione
- **Schema Validation**: Verifica conformità del grafo finale allo schema universale

---

## 🚀 Installazione e Setup

### 1. Clonare il repository

```bash
git clone <repository-url>
cd KnowledgeExtraction/KG-pipeline
```

### 2. Creare ambiente virtuale

```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# oppure
.venv\Scripts\activate  # Windows
```

### 3. Installare dipendenze

```bash
pip install -r requirements.txt
```

### 4. Configurare API key OpenAI

```bash
cp .env.example .env
# Editare .env e inserire la propria API key
```

### 5. Aggiungere documenti

Copiare i PDF nelle rispettive cartelle in `source/` seguendo le indicazioni in `source/README.md`.

---

## 💻 Utilizzo

### Eseguire tutte le pipeline

```bash
python main.py --mode all
```

### Eseguire una singola pipeline

```bash
# Pipeline 1: Product & Technical Data
python main.py --mode single --pipeline product_technical

# Pipeline 3: Troubleshooting
python main.py --mode single --pipeline troubleshooting
```

### Validare output

```bash
python main.py --mode validate --output output/merged/final_graph.json
```

### Configurazione

Modificare `config.yaml` per:
- Parametri LLM (modello, temperatura, max tokens)
- Filtraggio contenuti
- Impostazioni logging
- Configurazione pipeline-specific

---

## 📊 Schema JSON Universale

Il file `schemas/neural_extraction.json` definisce:

- **Entità**: 14 tipi (Product, Component, FailureMode, TestSpec, ecc.)
- **Relazioni**: 14 tipi (hasPart, affects, requiresTool, verifies, ecc.)
- **Metadati**: Provenance, confidence scores, source spans
- **Quality metrics**: Validazione, conteggi, warning

Ogni pipeline popola un **sottoinsieme** di questo schema.

---

## 🧪 Approccio Incrementale

### Fase 1: Setup Struttura ✅
- Struttura cartelle creata
- Schema JSON universale definito
- Configurazione di base

### Fase 2: Pipeline Pilota (Product & Technical Data)
- Implementare `symbolic_parser.py` per estrazione dati tecnici
- Implementare `neural_extractor.py` con prompt calibrato
- Test su documenti reali
- Validazione output

### Fase 3: Graph Merger
- Implementare logica di merge
- Conflict resolution
- Provenance tracking

### Fase 4: Orchestrator
- Coordinamento pipeline
- Logging e monitoraggio
- Report esecuzione

### Fase 5: Altre Pipeline
- Replicare pattern per pipeline 2-5
- Test integrazione end-to-end

---

## 🔍 Output Generati

### JSON Parziali (output/partial/)

Ogni pipeline genera un JSON parziale:

```json
{
  "document_code": "SERVICE_MANUAL_CH1-4",
  "pipeline": "product_technical",
  "entities": [
    {
      "id": "ent_001",
      "type": "Component",
      "name": "Heating Element",
      "confidence": 0.95
    }
  ],
  "relations": [
    {
      "type": "hasPart",
      "from_ref": "prod_001",
      "to_ref": "ent_001"
    }
  ]
}
```

### JSON Finale (output/merged/)

Il merger produce un grafo unificato con:
- Entità deduplicate
- Relazioni consolidate
- Provenance per ogni elemento
- Quality metrics globali

---

## 🛡️ Quality Assurance

- **Schema Validation**: Tutti gli output sono validati contro `neural_extraction.json`
- **Confidence Scores**: Ogni entità/relazione ha un punteggio di confidenza
- **Source Tracing**: Ogni elemento è tracciato alla sezione/pagina sorgente
- **Warning System**: Il sistema segnala anomalie, duplicati, inconsistenze

---

## 📝 Logging

I log sono salvati in `output/logs/` con:
- Timestamp esecuzione
- Pipeline eseguita
- Errori e warning
- Statistiche performance (tempo, entità estratte, token usati)

---

## 🤝 Contribuire

Questo progetto segue un approccio incrementale. Per contribuire:

1. Implementare una pipeline seguendo il pattern di `BasePipeline`
2. Aggiungere test in `tests/`
3. Documentare prompt e strategie nel codice
4. Testare su documenti reali

---

## 📚 Riferimenti

- **Schema JSON**: `schemas/neural_extraction.json`
- **Configurazione**: `config.yaml`
- **Documentazione source**: `source/README.md`

---

## 📄 Licenza

Progetto di ricerca per estrazione semantica da documentazione tecnica.

## ✨ Caratteristiche Tecniche

- **Modularità**: Design SOLID con separation of concerns
- **Estensibilità**: Facile aggiunta di nuove pipeline
- **Tracciabilità**: Provenance completa da documento a entità
- **Robustezza**: Validazione multi-livello e error handling
- **Performance**: Possibilità di esecuzione parallela delle pipeline

---

**Versione**: 2.0 - Architettura Modulare
**Status**: In sviluppo - Fase 2 (Pipeline Pilota)
