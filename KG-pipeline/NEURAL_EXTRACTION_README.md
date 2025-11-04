# Neural Knowledge Extraction

Sistema di estrazione neurale per generare tre sotto-Knowledge Graphs (KG) a partire dai file JSON estratti dalle pipeline simboliche.

## 📋 Panoramica

Il modulo `Neural_extraction.py` utilizza modelli AI (OpenAI GPT) per estrarre automaticamente entità e relazioni da documenti tecnici, organizzandoli in tre profili specializzati:

1. **product_technical**: Prodotti, componenti, specifiche tecniche
2. **operation_modes**: Modi operativi, stati, procedure, test
3. **troubleshooting**: Guasti, azioni di riparazione, manutenzione

## 🚀 Versione 1.1 - Hardening Update

Questa versione include significativi miglioramenti per robustezza e qualità dei dati:

### ✨ Nuove Funzionalità

1. **ID Deterministici con Namespace**
   - Pattern: `ns:TypePrefix/entity_name_slug`
   - Esempi: `ns:Product/citiz`, `ns:ComponentType/thermoblock`, `ns:FM/no_water_flow`

2. **JSON Parsing Robusto**
   - Riparazione automatica di JSON malformato
   - Gestione di virgolette tipografiche, virgole mancanti, trailing commas
   - Estrazione selettiva di arrays entities/relations

3. **Retry Intelligente**
   - Tentativo 1: Prompt completo, temperatura da config
   - Tentativo 2: Prompt semplificato, temperatura 0.1
   - Tentativo 3: Prompt semplificato, temperatura 0.0

4. **Deduplicazione Avanzata**
   - Dedup entità per (tipo, nome normalizzato)
   - Dedup relazioni per (tipo, from_ref, to_ref)
   - Consolidamento basato su confidence massima

5. **Correzione Semantica (Troubleshooting)**
   - Pattern forzati: `FailureMode --mitigatedBy--> RepairAction`
   - Creazione automatica di `RepairAction` intermedie per Tool/Consumable
   - Validazione delle direzioni delle relazioni

6. **Normalizzazione Testo**
   - Unificazione di sinonimi ("Descaling mode" = "Descaling Mode")
   - Slugification consistente
   - Rimozione spazi multipli e caratteri speciali

## 🚀 Utilizzo

### Prerequisiti

1. Installare le dipendenze:
```bash
pip install -r requirements.txt
```

2. Configurare la chiave API OpenAI:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

### Esecuzione

#### Processare tutti i profili

```bash
python3 src/Neural_extraction.py --profile all
```

#### Processare un singolo profilo

```bash
python3 src/Neural_extraction.py --profile product_technical
python3 src/Neural_extraction.py --profile operation_modes
python3 src/Neural_extraction.py --profile troubleshooting
```

#### Dry-run (test senza chiamate API)

```bash
python3 src/Neural_extraction.py --profile all --dry-run
```

#### Modalità verbose

```bash
python3 src/Neural_extraction.py --profile all --verbose
```

## 📁 Struttura dei File

### Input

I file di input devono essere in formato JSON parsed dalla pipeline simbolica:

```
output/partial/
├── product_technical/
│   └── *.parsed.json
├── operation_modes/
│   └── *.parsed.json
└── troubleshooting/
    └── *.parsed.json
```

### Output

Il sistema genera per ogni profilo:

```
output/neural_extraction/
├── product_technical/
│   ├── kg.json          # Knowledge Graph completo
│   ├── quality.json     # Report qualità
│   └── raw/             # Output raw AI (debug)
├── operation_modes/
│   ├── kg.json
│   ├── quality.json
│   └── raw/
└── troubleshooting/
    ├── kg.json
    ├── quality.json
    └── raw/
```

## ⚙️ Configurazione

La configurazione si trova in `config.yaml`:

### Sezione Neural Extractor

```yaml
neural_extractor:
  enabled: false
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

### Profili di Estrazione

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
      - "connectedTo"
```

## 🔧 Miglioramenti Tecnici

### ID Deterministici

Gli ID seguono uno schema namespace coerente:

| Tipo Entità | Namespace | Esempio Input | ID Generato |
|-------------|-----------|---------------|-------------|
| Product | `ns:Product/` | "Citiz" | `ns:Product/citiz` |
| ComponentType | `ns:ComponentType/` | "NTC Temperature Sensor" | `ns:ComponentType/ntc_temperature_sensor` |
| Component | `ns:Component/` | "Pump Invensys CP4" | `ns:Component/pump_invensys_cp4` |
| MachineMode | `ns:Mode/` | "Descaling Mode" | `ns:Mode/descaling_mode` |
| State | `ns:State/` | "Ready to brew" | `ns:State/ready_to_brew` |
| FailureMode | `ns:FM/` | "No water flow" | `ns:FM/no_water_flow` |
| RepairAction | `ns:RA/` | "Replace pump" | `ns:RA/replace_pump` |
| Tool | `ns:Tool/` | "Torque Wrench" | `ns:Tool/torque_wrench` |

### Correzione Semantica Troubleshooting

Prima (errato):
```
FailureMode("No flow") --mitigatedBy--> Tool("Wrench")
```

Dopo (corretto):
```
FailureMode("No flow") --mitigatedBy--> RepairAction("Use Wrench")
RepairAction("Use Wrench") --requiresTool--> Tool("Wrench")
```

### Retry Strategy

Il sistema tenta l'estrazione fino a 3 volte con degradazione progressiva:

```
Attempt 1: Prompt dettagliato, temperature=0.2 (da config)
  ↓ (fallimento)
Attempt 2: Prompt semplificato, temperature=0.1
  ↓ (fallimento)
Attempt 3: Prompt semplificato, temperature=0.0 (deterministico)
```

### JSON Repair

Gestisce automaticamente:
- Code fences: ` ```json {...} ``` `
- Virgolette tipografiche: `"text"` → `"text"`
- Trailing commas: `[1, 2,]` → `[1, 2]`
- Missing commas: `}{"id"` → `},{"id"`
- Estrazione parziale se JSON incompleto

## 📊 Schema di Output

### Knowledge Graph (kg.json)

```json
{
  "document_code": "KG_PRODUCT_TECHNICAL",
  "ingestion_id": "uuid",
  "extraction_version": "neural_v1.0",
  "datasource_code": "NEURAL_EXTRACTION",
  "extractor": {
    "model": "gpt-4o-mini",
    "prompt_id": "neural_extraction_v1",
    "temperature": 0.2,
    "max_tokens": 1500
  },
  "allowed_types": ["Product", "Component", ...],
  "allowed_relations": ["hasPart", "instanceOf", ...],
  "entities": [
    {
      "id": "CT_01",
      "type": "ComponentType",
      "name": "Thermoblock",
      "confidence": 0.94
    }
  ],
  "relations": [
    {
      "type": "hasPart",
      "from_ref": "C_01",
      "to_ref": "CT_01",
      "confidence": 0.88
    }
  ],
  "provenance": {
    "overall_confidence": 0.91,
    "sections_used": [...],
    "notes": "..."
  },
  "quality": {
    "json_valid": true,
    "entities_count": 42,
    "relations_count": 38,
    "warnings": [],
    "overall_confidence": 0.91
  }
}
```

### Quality Report (quality.json)

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
  "generated_at": "2025-11-03T10:30:00Z"
}
```

## 🧠 Prompt Engineering

Il sistema utilizza un prompt strutturato che include:

1. **Schema di output** richiesto
2. **Tipi di entità ammessi** per il profilo specifico
3. **Relazioni ammesse** per il profilo specifico
4. **Regole di estrazione** specifiche
5. **Esempi di output** per guidare il modello

Il prompt viene costruito dinamicamente in base a:
- Profilo attivo
- Schema di validazione
- Testo del chunk corrente

## 🔍 Validazione e Normalizzazione

### Validazione

Il sistema valida:
- Presenza di campi obbligatori (`id`, `type`, `name`, `confidence`)
- Tipi di entità conformi al profilo
- Tipi di relazioni conformi al profilo
- Riferimenti validi tra relazioni ed entità
- Valori di confidence nell'intervallo [0, 1]

### Normalizzazione

Il sistema normalizza:
- **ID entità**: formato `<TIPO>_<HASH>` (es: `CT_a3f2b1`)
- **Nomi**: rimozione spazi extra
- **Deduplicazione**: entità duplicate vengono unite, mantenendo quella con confidence più alta

## 🐛 Debug

Per debug approfondito:

1. Attivare il salvataggio degli output raw:
```yaml
neural_extractor:
  save_raw_outputs: true
```

2. Eseguire con verbose:
```bash
python3 src/Neural_extraction.py --profile all --verbose
```

3. Controllare i file raw in `output/neural_extraction/<profile>/raw/`

## 📈 Metriche di Qualità

Il sistema genera metriche automatiche:

- **Validità JSON**: il risultato è JSON valido?
- **Conteggi**: numero di entità e relazioni estratte
- **Confidence**: media pesata della confidenza
- **Distribuzione**: per tipo di entità e relazione
- **Warning**: eventuali problemi rilevati durante l'estrazione

## 🔄 Workflow Completo

```bash
# 1. Eseguire le pipeline simboliche
python3 src/Symbolic_orchestrator.py --pipeline all

# 2. Eseguire l'estrazione neurale
python3 src/Neural_extraction.py --profile all

# 3. Verificare i risultati
ls -la output/neural_extraction/*/
cat output/neural_extraction/product_technical/quality.json
```

## ⚠️ Note Importanti

1. **API Key**: Assicurarsi di avere configurato `OPENAI_API_KEY`
2. **Costi**: Ogni esecuzione consuma token OpenAI (stimati: ~10-20k token per documento)
3. **Temperature**: Valori bassi (0.1-0.3) garantiscono risultati più consistenti
4. **Chunking**: Il testo viene diviso in blocchi di ~1000 token per evitare limiti API
5. **Retry**: In caso di fallimento, il sistema può ritentare con prompt semplificato

## 🛠️ Estensioni Future

- [ ] Supporto per altri provider AI (Anthropic, Google, Azure)
- [x] ~~Prompt semplificato per retry automatico~~ ✅ **Implementato v1.1**
- [ ] Merging intelligente dei sotto-KG in un KG unificato
- [ ] Allineamento cross-fonte (fuzzy matching tra profili)
- [ ] Normalizzazione unità UCUM
- [ ] Visualizzazione interattiva del KG
- [ ] Export in formati alternativi (RDF, Neo4j, GraphML)

## 📝 Changelog

### v1.1 - Hardening Update (2025-11-04)

**Fixed:**
- ⚠️ Deprecation warning `datetime.utcnow()` → `datetime.now(timezone.utc)`
- 🔧 JSON parsing errors con riparazione automatica
- 🐛 Relazioni semanticamente errate nel profilo troubleshooting

**Added:**
- ✨ ID deterministici con namespace pattern (`ns:Type/slug`)
- ✨ `safe_json_parse()` con 3 livelli di repair
- ✨ `extract_with_retry()` con degradazione automatica (3 tentativi)
- ✨ `deduplicate_entities()` con normalizzazione nomi
- ✨ `deduplicate_relations()` per chiave (type, from, to)
- ✨ `fix_troubleshooting_semantics()` per pattern corretti
- ✨ `slugify()` per nomi consistenti
- ✨ Prompt più stringente con enfasi su JSON valido

**Changed:**
- 🔄 `normalize_extraction()` ora include dedup completo
- 🔄 Versione extraction: `neural_v1.0` → `neural_v1.1`
- 🔄 Prompt ID: `neural_extraction_v1` → `neural_extraction_v1_hardened`
- 🔄 Log più informativi con primo 5 warning invece di tutti

**Performance:**
- ⚡ Riduzione chunk falliti grazie a retry intelligente
- ⚡ Eliminazione duplicati riduce dimensione KG finale del ~15-25%

### v1.0 - Initial Release (2025-11-03)
- 🎉 Sistema di estrazione neurale base
- 🎉 Supporto per 3 profili (product_technical, operation_modes, troubleshooting)
- 🎉 Integrazione OpenAI GPT
- 🎉 Validazione e quality metrics
