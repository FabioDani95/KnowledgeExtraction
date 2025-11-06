# Frontend - Knowledge Graph Extraction Pipeline

## Come Usare (5 Passi)

### 1. Avvia il backend server
```bash
cd KG-pipeline/frontend
./START_SERVER.sh
```
Lascia questa finestra aperta. Il server DEVE rimanere attivo.

### 2. Apri il frontend nel browser
Apri `index.html` con il browser (doppio click o drag nel browser).

### 3. Carica i 4 file PDF
Nella sezione "Upload", trascina o seleziona i 4 PDF nei rispettivi slot, poi clicca "Upload All Files".

### 4. Avvia la pipeline
Vai alla sezione "Pipeline" e clicca "Start Pipeline". Aspetta che finisca (100%).

### 5. Visualizza il Knowledge Graph
Vai alla sezione "Knowledge Graph" per vedere tutti i nodi e le relazioni estratte.

---

## ⚠️ Importante

**Se il grafo mostra solo 44 nodi invece di 160+**: il server backend NON è attivo!
- Torna allo step 1 e avvia `./START_SERVER.sh`
- Ricarica la pagina del browser (F5)

Il server backend è **obbligatorio** per caricare il KG reale. Senza server = solo mock data.
