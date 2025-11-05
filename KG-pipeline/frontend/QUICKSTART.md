# Quick Start Guide

## 🚀 Launch the Frontend (2 easy options)

### Option 1: Automated Startup (Recommended)

Navigate to the frontend directory and run the startup script:

**Linux/Mac:**
```bash
cd KG-pipeline/frontend
./start_frontend.sh
```

**Windows:**
```cmd
cd KG-pipeline\frontend
start_frontend.bat
```

This will automatically:
- ✅ Install required Python packages
- ✅ Start the upload server (port 8000)
- ✅ Start the frontend web server (port 8080)
- ✅ Open http://localhost:8080 in your browser

### Option 2: Manual Startup

If you prefer to start servers manually:

1. **Install dependencies** (first time only):
   ```bash
   cd KG-pipeline/frontend
   pip install -r requirements.txt
   ```

2. **Start upload server** (in one terminal):
   ```bash
   python3 upload_server.py
   ```

3. **Start frontend server** (in another terminal):
   ```bash
   python3 -m http.server 8080
   ```

4. **Open browser**: http://localhost:8080

---

## 📋 What You'll See

The frontend has **4 main sections**:

### 1. 📤 Upload Section
- Drag-and-drop PDF files
- Select destination folder (testing, product_technical, etc.)
- View upload queue
- Click "Upload Files" to send files to the pipeline

### 2. ⚙️ Pipeline Section
- Click "▶️ Start Pipeline" to run extraction
- Watch real-time progress bar
- View live execution logs
- Download complete logs when finished

### 3. 🕸️ Knowledge Graph Section
- **Interactive graph visualization** with Cytoscape.js
- **Controls:**
  - Change layout (force, circle, grid, etc.)
  - Search nodes by name
  - Filter by node/edge types
  - Adjust confidence threshold
- **Interactions:**
  - Click nodes/edges to see details
  - Drag nodes to reposition
  - Pan/zoom the graph
- **Export:**
  - PNG: High-res graph image
  - JSON: Complete graph data
  - CSV: Tabular export

### 4. 📈 Statistics Section
- Total counts (nodes, edges, types)
- Confidence metrics
- Quality indicators
- Coverage progress bars

---

## 🔧 Configuration

### Mock vs. Real Backend

The frontend works in two modes:

**1. Mock Mode (Default)**
- No backend required
- Uses sample data
- Perfect for testing UI
- Located in: `js/api.js` line 16

**2. Real Backend Mode**
To connect to actual backend:
1. Edit `js/api.js`, line 16: Change `USE_MOCK = true` to `USE_MOCK = false`
2. Edit `config.json`: Set `apiBaseUrl` to your backend URL
3. Ensure backend implements required endpoints (see `frontend/README.md`)

### Customization

**Change colors:**
- Edit `config.json` → `nodeTypes` and `edgeTypes`

**Add entity types:**
- Edit `config.json` → Add to `nodeTypes` with color and icon

**Change upload folders:**
- Edit `config.json` → `uploadTargets`

---

## ❓ Common Questions

**Q: Can I use this without the backend?**
A: Yes! Mock mode loads sample data for testing.

**Q: How do I load my own knowledge graph?**
A: Place your `kg_merged.json` in `../output/merged_kg/` and the frontend will auto-load it.

**Q: What browsers are supported?**
A: Chrome, Firefox, Safari, Edge (latest versions). IE is not supported.

**Q: The graph is slow with 10,000 nodes. What can I do?**
A: Use filters to reduce visible nodes. Try confidence threshold or type filters.

**Q: Can I change the graph colors?**
A: Yes! Edit `config.json` → `nodeTypes` section.

---

## 🐛 Troubleshooting

**Problem:** Blank graph canvas
- **Solution:** Open browser console (F12) and check for errors. Verify Cytoscape.js loaded.

**Problem:** Upload not working
- **Solution:** Check file is PDF and under 100MB. Verify destination folder selected.

**Problem:** "CORS error" in console
- **Solution:** Must use a web server (python, node, etc.). Opening `index.html` directly won't work.

**Problem:** Can't see my KG data
- **Solution:** Check `../output/merged_kg/kg_merged.json` exists. In mock mode, sample data is shown.

---

## 📚 Next Steps

- Read full documentation: `frontend/README.md`
- Explore configuration options: `config.json`
- Backend integration guide: `frontend/README.md` → "Backend Integration"
- Customize styling: `styles.css`

---

**Happy Knowledge Graph Exploring! 🎉**
