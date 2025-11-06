# Knowledge Graph Frontend - Usage Guide

## Quick Start

### 1. Start the Backend Server

The frontend needs the backend server running to load the real Knowledge Graph data.

**Option A: Use the start script (recommended)**
```bash
cd KG-pipeline/frontend
./START_SERVER.sh
```

**Option B: Start manually**
```bash
cd KG-pipeline/frontend
python3 upload_server.py
```

You should see:
```
============================================================
Knowledge Graph Pipeline - Upload Server
============================================================
Starting server on http://localhost:8000
============================================================
```

### 2. Open the Frontend

**Option A: Open the HTML file directly**
```bash
# From the frontend directory
open index.html
# or on Linux
xdg-open index.html
```

**Option B: Through the server**
Navigate to: `http://localhost:8000/index.html`

## Important: Loading Real KG Data

### The Problem

If you see only **44 nodes and 28 edges** in the graph viewer, but the pipeline found **164 entities and 70 relations**, it means:

- ❌ The backend server is NOT running
- ❌ The frontend is loading MOCK data instead of the real KG

### The Solution

1. **Make sure the backend server is running** (see step 1 above)
2. **Open the browser console** (F12 or right-click → Inspect → Console)
3. **Look for these messages**:
   - ✅ `Loaded REAL KG from backend: 164 entities, 70 relations` = **CORRECT**
   - ❌ `WARNING: Could not load actual KG from any source. Using mock data.` = **WRONG**

If you see the warning, the server is not accessible. Check that:
- The server is running on port 8000
- No firewall is blocking localhost:8000
- The KG file exists: `../output/merged_kg/kg_merged.json`

### Expected Console Output (Correct)

When working correctly, you should see:
```
🔍 Attempting to load KG from backend server...
✅ Loaded REAL KG from backend: 164 entities, 70 relations
🔍 Attempting to load stats from backend server...
✅ Loaded REAL stats from backend: 164 nodes, 70 edges
```

## Features

### Upload Section
- 4 dedicated file slots for PDF documents
- Each slot for specific document type
- All 4 files must be uploaded before running pipeline

### Configuration Section
- Neural extractor settings (OpenAI model, temperature, etc.)
- Pipeline settings (enable/disable specific pipelines)
- Validation settings (confidence thresholds)
- Export configuration to YAML

### Pipeline Section
- Start pipeline execution
- Monitor progress with real-time updates
- View logs

### Knowledge Graph Section
- Interactive graph visualization
- Search and filter nodes/edges
- View node/edge details
- Export graph (PNG, JSON, CSV)
- Multiple layout algorithms

### Statistics Section
- Entity and relation counts
- Type distribution
- Confidence scores
- Merge information

## Troubleshooting

### Graph shows only mock data (44 nodes, 28 edges)

**Solution**: Start the backend server (see step 1)

### "Backend server not available" error

**Check**:
1. Is the server running? Look for the startup message
2. Is port 8000 free? Try: `lsof -i :8000` or `netstat -an | grep 8000`
3. Try restarting the server

### Empty graph / "No Knowledge Graph Available"

**This is normal** when:
- No pipeline has been run yet
- The KG file doesn't exist

**Solution**:
1. Upload 4 PDF files
2. Run the pipeline
3. Wait for completion
4. Refresh the page

### Statistics don't match graph

**This happens when** the server is not running. The stats come from mock data while trying to load the real graph.

**Solution**: Start the backend server and refresh the page.

## File Structure

```
frontend/
├── index.html              # Main HTML page
├── styles.css              # All styles
├── config.json             # Frontend configuration
├── i18n/                   # Translations (en, it)
│   ├── en.json
│   └── it.json
├── js/
│   ├── app.js              # Main application logic
│   ├── api.js              # Backend API adapter
│   ├── kg-viz.js           # Graph visualization (Cytoscape)
│   └── upload-slots.js     # File upload management
├── upload_server.py        # Backend server (Flask)
├── START_SERVER.sh         # Server startup script
└── FRONTEND_USAGE.md       # This file
```

## API Endpoints (upload_server.py)

- `POST /upload` - Upload files to source folders
- `POST /pipeline/start` - Start pipeline execution
- `GET /pipeline/status/<job_id>` - Get pipeline status
- `GET /kg/current` - Get merged knowledge graph
- `GET /kg/stats` - Get KG statistics
- `GET /config` - Get pipeline configuration
- `POST /config` - Update pipeline configuration
- `GET /health` - Health check

## Development Notes

### Mock Mode

The frontend has `USE_MOCK = true` in `api.js`. This enables:
- File upload without backend
- Pipeline simulation
- **Fallback to mock data** if backend is unavailable

To disable mock mode entirely:
1. Edit `js/api.js`
2. Change `USE_MOCK = true` to `USE_MOCK = false`
3. Now the frontend will fail hard if backend is not available

### Adding New Node/Edge Types

1. Edit `config.json`
2. Add to `nodeTypes` or `edgeTypes`
3. Add icon and color
4. Refresh the page

### Changing Layout

The graph supports multiple layouts:
- `cose` - Force-directed (default)
- `circle` - Circular layout
- `grid` - Grid layout
- `breadthfirst` - Tree layout
- `concentric` - Concentric circles

Change via the dropdown in the Knowledge Graph section.

## Support

For issues or questions:
1. Check the browser console (F12)
2. Check the server logs
3. Verify file paths and permissions
4. Check the main README in the project root
