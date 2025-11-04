# Merge Simbolico dei Knowledge Graphs

## Cosa fa `merge_kgs.py`

Lo script unisce N sotto-KG (file `kg.json`) in un unico KG unificato, in modo **simbolico** (senza LLM), seguendo questi passaggi:

### 1. 📥 **Caricamento KG**
Carica tutti i file JSON che matchano i pattern forniti (es. `Partial_KG/*.json`) e raccoglie entità e relazioni da tutti i file.

### 2. 🔄 **Deduplicazione Entità**
- Normalizza ogni entità con chiave `(tipo, nome_normalizzato)` dove il nome è convertito in lowercase e caratteri speciali → underscore
- Trova duplicati: entità con stesso tipo e nome simile (es. "Flow Meter" = "flow_meter" = "Flow-Meter")
- Per ogni gruppo di duplicati, sceglie un **ID canonico** usando: priorità sorgente > confidence > lunghezza ID
- Unisce le proprietà mantenendo quelle della sorgente prioritaria in caso di conflitto
- Conserva tutti gli span e aggiunge metadati di provenienza

### 3. 🔗 **Remap Relazioni**
- Sostituisce tutti i riferimenti (`from_ref`, `to_ref`) con gli ID canonici delle entità deduplicate
- Elimina relazioni "orfane" che puntano a entità inesistenti
- Deduplica relazioni identiche: stesso tipo + stessi estremi

### 4. 🔧 **Fix Sintattici**
Corregge relazioni invertite per errore:
- `hasUnit`: deve andare da `ParameterSpec` → `Unit` (se invertita, la corregge)
- `hasSpec`: deve andare da `Product/Component/ComponentType` → `ParameterSpec`

### 5. ✅ **Validazione**
Esegue controlli configurabili (da `config.yaml`):
- Integrità referenziale: tutti i riferimenti esistono?
- Domain/range: le relazioni collegano i tipi giusti?
- Cicli: ci sono cicli nelle relazioni `precedes`?
- Self-loop: relazioni che puntano a se stesse?
- Confidence: entità/relazioni sotto soglia minima?
- Unità: parametri numerici hanno l'unità di misura?

### 6. 📊 **Quality Report**
Genera statistiche:
- Conteggi entità/relazioni per tipo
- Confidence medie
- Numero duplicati collassati
- Fix applicati
- Cicli trovati

### 7. 💾 **Output**
Scrive `kg_merged.json` contenente:
```json
{
  "document_code": "KG_MERGED",
  "datasource_code": "SYMBOLIC_MERGE",
  "merge_info": { "source_files": [...], "priority_sources": [...] },
  "entities": [...],
  "relations": [...],
  "validation": { "issues": [...], "counters": {...} },
  "quality": { "entities_count": N, ... }
}
```

## Esempio Output Console

```
================================================================================
KG Symbolic Merge - Starting
================================================================================

[1/8] Loading KG files...
  Found 3 KG files
  Total entities loaded: 98
  Total relations loaded: 75

[2/8] Deduplicating entities...
  Entities after deduplication: 98
  Entity groups with duplicates: 0
  Total duplicates collapsed: 0

[3/8] Remapping relations...
  Relations after remap: 75
  Orphaned relations removed: 0

[4/8] Deduplicating relations...
  Relations after deduplication: 75
  Duplicate relations removed: 0

[5/8] Applying syntactic fixes...
  Relations with fixes: 75
  Inverted relations fixed: 0

[6/8] Running validations...
  Validation issues found:
    Errors: 0
    Warnings: 0
    Info: 0

[7/8] Generating quality report...
  Quality metrics computed

[8/8] Writing output...
  Output written to: output/merged_kg/kg_merged.json

================================================================================
KG Symbolic Merge - Completed Successfully
================================================================================
```

## Uso

```bash
# Merge base
python src/merge_kgs.py "KG-pipeline/output/neural_extraction/Partial_KG/*.json" \
  --out output/merged_kg/kg_merged.json

# Con priorità sorgenti (in caso di conflitti, vince la prima)
python src/merge_kgs.py "KG-pipeline/output/neural_extraction/Partial_KG/*.json" \
  --out output/merged_kg/kg_merged.json \
  --priority NEURAL_EXTRACTION KG_PRODUCT_TECHNICAL KG_OPERATION_MODES

# Con config personalizzato
python src/merge_kgs.py "path/*.json" --out merged.json --config my_config.yaml
```

## Note Importanti

- ⚙️ **Simbolico**: non usa LLM, solo logica deterministica
- 🔐 **Non distruttivo**: non modifica i file originali
- 📝 **Tracciabile**: ogni entità merged conserva la lista di sorgenti originali
- ⚡ **Veloce**: processa grandi KG in pochi secondi
- ✅ **Validato**: ogni merge viene validato secondo le regole in `config.yaml`
