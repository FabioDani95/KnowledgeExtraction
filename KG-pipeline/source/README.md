# Source Documents Directory

Questa cartella contiene i documenti PDF organizzati per pipeline di estrazione.

## Struttura

Ogni sottocartella corrisponde a una pipeline verticale specializzata:

### 📊 product_technical/
**Pipeline 1: Product & Technical Data**
- Capitoli 1-4 del Service Manual
- Specifiche tecniche dettagliate
- Dati di targa e rating plate
- Documentazione componenti e parametri

**Entità estratte**: Product, Component, ComponentType, ParameterSpec, Unit, RatingPlate

---

### ⚙️ operation_modes/
**Pipeline 2: Operation & Machine Modes**
- User Manual completo
- Guide operative
- Documentazione modalità macchina
- Procedure di utilizzo

**Entità estratte**: ProcessStep, MachineMode, MaintenanceTask, Tool, State

---

### 🔧 troubleshooting/
**Pipeline 3: Troubleshooting & Diagnostics**
- Tabelle codici errore
- Guide troubleshooting
- Regole diagnostiche
- Procedure di riparazione

**Entità estratte**: FailureMode, RepairAction, DiagnosticRule, Component

---

### 🛠️ repair_structure/
**Pipeline 4: Repair & Parts Structure**
- Sezioni disassembly del Service Manual
- Parts list e cataloghi ricambi
- Diagrammi esplosi
- Procedure di smontaggio

**Entità estratte**: ComponentType, Component, Tool, ProcessStep

---

### ✅ testing/
**Pipeline 5: Testing & Verification**
- Procedure di test e collaudo
- Specifiche di calibrazione
- Soglie anomalie
- Protocolli di verifica

**Entità estratte**: TestSpec, AnomalyThreshold, ParameterSpec, Unit

---

## Istruzioni

1. Inserire i PDF nelle rispettive cartelle in base al contenuto semantico
2. Un documento può essere copiato in più cartelle se contiene sezioni rilevanti per più pipeline
3. I nomi file devono essere descrittivi (es. `service_manual_chapters_1-4.pdf`)
4. Formati supportati: PDF (il sistema include OCR per documenti scansionati)
