# Neural Knowledge Extraction

Sistema di estrazione neurale per generare tre sotto-Knowledge Graphs (KG) a partire dai file JSON estratti dalle pipeline simboliche.

## 📋 Panoramica

Il modulo `Neural_extraction.py` utilizza modelli AI (OpenAI GPT) per estrarre automaticamente entità e relazioni da documenti tecnici, organizzandoli in tre profili specializzati:

1. **product_technical**: Prodotti, componenti, specifiche tecniche
2. **operation_modes**: Modi operativi, stati, procedure, test
3. **troubleshooting**: Guasti, azioni di riparazione, manutenzione

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
- [ ] Prompt semplificato per retry automatico
- [ ] Merging intelligente dei sotto-KG in un KG unificato
- [ ] Visualizzazione interattiva del KG
- [ ] Export in formati alternativi (RDF, Neo4j, GraphML)
