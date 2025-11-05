# Knowledge Graph Frontend

A lightweight, modular frontend for the Knowledge Graph Extraction Pipeline. Built with vanilla JavaScript and Cytoscape.js for interactive graph visualization.

## Features

### 1. Document Upload
- **Drag-and-drop interface** for PDF files
- **Multi-target selection**: Upload to specific pipeline folders (testing, product_technical, operation_modes, troubleshooting, repair_structure)
- **File validation**: MIME type checking, size limits, queue management
- **Upload queue**: View pending uploads with status tracking
- **Batch operations**: Upload multiple files at once

### 2. Pipeline Execution
- **One-click pipeline start**: Execute the complete extraction pipeline
- **Real-time progress tracking**: Visual progress bar with phase updates
- **Live logs**: View recent pipeline execution logs
- **Job monitoring**: Automatic polling for job status updates
- **Error handling**: Clear error messages with recovery actions

### 3. Interactive Knowledge Graph Visualization
- **Multiple layouts**: Force-directed (COSE), Circle, Grid, Breadth-first, Concentric
- **Node interactions**: Click, hover, drag nodes
- **Advanced filtering**:
  - Filter by node types (Product, Component, TestCase, etc.)
  - Filter by relationship types
  - Confidence threshold slider
  - Full-text search across node labels
- **Detail panel**: Click nodes/edges to view detailed information
- **Focus mode**: Highlight node neighborhoods (degree 1-2)
- **Color-coded nodes**: Different colors for each entity type
- **Styled edges**: Different styles (solid, dashed, dotted) for relationship types

### 4. Statistics & Quality Metrics
- **KPI Dashboard**: Total nodes, edges, entity types, relation types
- **Confidence metrics**: Average entity and relationship confidence scores
- **Coverage targets**: Track key relationship coverage goals
- **Quality indicators**: Duplicate detection, data completeness

### 5. Export Capabilities
- **PNG Export**: High-resolution graph snapshots
- **JSON Export**: Complete knowledge graph data
- **CSV Export**: Tabular representation of nodes and edges

## Architecture

### Modular Design

The frontend follows a clean, modular architecture with clear separation of concerns:

```
frontend/
├── index.html          # Main HTML structure
├── styles.css          # Responsive CSS with theming
├── config.json         # Configuration (API endpoints, colors, mappings)
├── js/
│   ├── api.js         # Backend API adapter (easily replaceable)
│   ├── kg-viz.js      # Graph visualization module (Cytoscape wrapper)
│   └── app.js         # Application orchestration & state management
└── i18n/
    ├── en.json        # English translations
    └── it.json        # Italian translations
```

### Key Modules

#### **api.js** - Backend Integration Adapter
Encapsulates all backend communication with stub implementations for easy integration.

**Public API:**
```javascript
const api = API.create(config);

// Upload files to target folder
await api.uploadFiles(files, targetFolder);

// Start pipeline execution
await api.startPipeline(payload);

// Poll job status
await api.getJobStatus(jobId);

// Fetch knowledge graph
await api.getKG();

// Get statistics
await api.getStats();

// Get configuration
await api.getConfig();

// Download logs
await api.downloadLogs(jobId);
```

**Integration Note:** Currently uses mock data (`USE_MOCK = true`). Set to `false` and implement actual endpoints when backend is ready.

#### **kg-viz.js** - Graph Visualization
Wraps Cytoscape.js for graph rendering with a clean, library-agnostic API.

**Public API:**
```javascript
const viz = KGViz.create(containerElement, config);

// Initialize visualization
viz.init(options);

// Render graph data
viz.render(kgData, layoutOptions);

// Apply filters
viz.applyFilters({
  nodeTypes: ['Product', 'Component'],
  edgeTypes: ['hasPart'],
  minConfidence: 0.75,
  searchTerm: 'pump'
});

// Focus on node neighborhood
viz.focusNode(nodeId, degree);

// Search nodes
viz.search(query);

// Change layout
viz.changeLayout('cose');

// Export functions
viz.exportPNG();
viz.exportJSON();

// Get details
viz.getNodeDetails(nodeId);
viz.getEdgeDetails(edgeId);

// Event handling
viz.on('nodeSelected', (nodeData) => { ... });
viz.on('edgeSelected', (edgeData) => { ... });
```

**Library Encapsulation:** The Cytoscape.js dependency is completely encapsulated in `kg-viz.js`. To switch visualization libraries, only this file needs to be modified.

#### **app.js** - Application Orchestration
Central state management and UI coordination.

**Responsibilities:**
- Application initialization and configuration loading
- File upload queue management
- Pipeline execution and monitoring
- Graph visualization coordination
- Filter application and state synchronization
- Event handling and UI updates
- Toast notifications and user feedback

## Installation & Setup

### Prerequisites
- Web server (Python, Node.js, or any static file server)
- Modern web browser (Chrome, Firefox, Safari, Edge)

### Quick Start

1. **Navigate to the frontend directory**
   ```bash
   cd KG-pipeline/frontend
   ```

2. **Start a local web server**

   **Option A: Python 3**
   ```bash
   python3 -m http.server 8080
   ```

   **Option B: Python 2**
   ```bash
   python -m SimpleHTTPServer 8080
   ```

   **Option C: Node.js (http-server)**
   ```bash
   npx http-server -p 8080
   ```

   **Option D: PHP**
   ```bash
   php -S localhost:8080
   ```

3. **Open in browser**
   ```
   http://localhost:8080
   ```

### Configuration

Edit `config.json` to customize the frontend:

```json
{
  "apiBaseUrl": "http://localhost:8000",
  "pollingInterval": 3000,
  "maxFileSize": 104857600,
  "maxFilesPerBatch": 10,

  "uploadTargets": {
    "product_technical": {
      "label": "Product & Technical Data",
      "path": "source/product_technical"
    }
  },

  "nodeTypes": {
    "Product": { "color": "#3498db", "icon": "📦" }
  },

  "edgeTypes": {
    "hasPart": { "style": "solid", "color": "#3498db", "width": 2 }
  }
}
```

**Key Configuration Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `apiBaseUrl` | Backend API base URL | `http://localhost:8000` |
| `pollingInterval` | Job status polling interval (ms) | `3000` |
| `maxFileSize` | Maximum upload file size (bytes) | `104857600` (100MB) |
| `maxFilesPerBatch` | Maximum files per upload batch | `10` |
| `uploadTargets` | Available destination folders | See config |
| `nodeTypes` | Entity type styling (color, icon) | See config |
| `edgeTypes` | Relationship type styling | See config |
| `confidenceThresholds` | Confidence level breakpoints | `0.85/0.70/0.50` |
| `coverageTargets` | Quality metric targets | See config |

### Backend Integration

The frontend is designed to work with or without a backend:

**Development Mode (Mock Data):**
- Set `USE_MOCK = true` in `api.js` (line 16)
- Frontend loads sample data and simulates API responses
- Perfect for UI development and testing

**Production Mode (Real Backend):**
1. Set `USE_MOCK = false` in `api.js`
2. Implement the following backend endpoints:

   ```
   POST   /api/upload              - Upload files
   POST   /api/pipeline/start      - Start pipeline
   GET    /api/pipeline/status/:id - Get job status
   GET    /api/kg/current          - Get current KG
   GET    /api/kg/stats            - Get statistics
   GET    /api/config              - Get backend config
   GET    /api/logs/:jobId         - Download logs
   ```

3. Update `apiBaseUrl` in `config.json`

**Expected Backend Response Formats:**

**Upload Response:**
```json
{
  "success": true,
  "data": {
    "uploaded": [
      { "name": "file.pdf", "status": "success", "path": "..." }
    ],
    "count": 1
  }
}
```

**Pipeline Start Response:**
```json
{
  "success": true,
  "data": {
    "jobId": "job_12345",
    "status": "started",
    "startedAt": "2025-11-05T10:00:00Z"
  }
}
```

**Job Status Response:**
```json
{
  "success": true,
  "data": {
    "jobId": "job_12345",
    "phase": "Neural entity extraction",
    "phaseIndex": 2,
    "totalPhases": 7,
    "progress": 35,
    "status": "running",
    "logs": ["Log line 1", "Log line 2"],
    "error": null
  }
}
```

**Knowledge Graph Response:**
```json
{
  "success": true,
  "data": {
    "document_code": "KG_MERGED",
    "entities": [
      {
        "id": "ns:Product/citiz",
        "type": "Product",
        "name": "Citiz Coffee Machine",
        "confidence": 0.95,
        "spans": [...]
      }
    ],
    "relations": [
      {
        "type": "hasPart",
        "from_ref": "ns:Product/citiz",
        "to_ref": "ns:Component/pump",
        "confidence": 0.88
      }
    ]
  }
}
```

## Usage Guide

### 1. Uploading Documents

1. Navigate to the **Upload** section
2. Drag PDF files onto the drop zone or click "Select Files"
3. Choose a destination folder from the dropdown
4. Review the file queue
5. Click "Upload Files"
6. Monitor upload status for each file

### 2. Running the Pipeline

1. Navigate to the **Pipeline** section
2. Click "▶️ Start Pipeline"
3. Monitor the progress bar showing current phase
4. View live logs in the logs panel
5. Wait for completion notification
6. Download complete logs if needed

### 3. Exploring the Knowledge Graph

**Basic Navigation:**
- **Pan**: Click and drag on empty space
- **Zoom**: Mouse wheel or pinch gesture
- **Select node**: Click on a node
- **Select edge**: Click on an edge

**Filtering:**
1. Use the search box to find nodes by name/ID
2. Adjust the confidence slider to hide low-confidence entities
3. Check/uncheck node types to show/hide entity types
4. Check/uncheck relationship types to filter edges

**Layout Options:**
- **Force-directed (COSE)**: Natural clustering based on connections
- **Circle**: Nodes arranged in a circle
- **Grid**: Nodes arranged in a grid
- **Breadth-first**: Hierarchical tree layout
- **Concentric**: Nodes arranged by degree centrality

**Detail Inspection:**
- Click any node to view:
  - Entity type and ID
  - Label/name
  - Confidence score
  - Source document references (pages, sections)
- Click any edge to view:
  - Relationship type
  - Source and target nodes
  - Confidence score

**Exporting:**
- **PNG**: Click "💾 Export PNG" for a high-resolution graph image
- **JSON**: Click "💾 Export JSON" for the complete graph data
- **CSV**: Click "💾 Export CSV" for a tabular export

### 4. Viewing Statistics

Navigate to the **Statistics** section to view:
- Total counts (nodes, edges, types)
- Average confidence scores
- Entity/relationship type distributions
- Coverage target progress
- Quality metrics

## Customization

### Adding New Entity Types

1. **Update `config.json`:**
   ```json
   "nodeTypes": {
     "MyNewType": { "color": "#ff6b6b", "icon": "🆕" }
   }
   ```

2. Node will automatically appear in:
   - Graph visualization (colored)
   - Filter panel
   - Legend

### Adding New Relationship Types

1. **Update `config.json`:**
   ```json
   "edgeTypes": {
     "myNewRelation": {
       "style": "dashed",
       "color": "#4ecdc4",
       "width": 2
     }
   }
   ```

2. Relationship will automatically:
   - Render with specified style
   - Appear in filter panel

### Changing Colors & Theme

**Option 1: Edit CSS Variables (global)**

Edit `styles.css`:
```css
:root {
  --primary: #your-color;
  --secondary: #your-color;
  /* ... */
}
```

**Option 2: Edit `config.json` (per entity type)**
```json
"nodeTypes": {
  "Product": { "color": "#your-color", "icon": "📦" }
}
```

### Internationalization

Add a new language:

1. Create `i18n/fr.json` (example: French)
2. Copy keys from `i18n/en.json`
3. Translate values
4. Set `defaultLanguage: "fr"` in `config.json`

## Troubleshooting

### Graph Not Rendering

**Symptoms:** Blank canvas in Knowledge Graph section

**Solutions:**
1. Check browser console for errors (F12)
2. Verify Cytoscape.js loaded: Check Network tab for `cytoscape.min.js`
3. Ensure `config.json` is valid JSON
4. Check that KG data is loading: Network tab → `api/kg/current` or mock data

### Upload Not Working

**Symptoms:** Files don't upload or error messages appear

**Solutions:**
1. Check file type (only PDF allowed by default)
2. Check file size (max 100MB by default)
3. Verify destination folder is selected
4. Check browser console for API errors
5. If using real backend, verify endpoint is running

### Pipeline Not Starting

**Symptoms:** "Start Pipeline" button does nothing or shows error

**Solutions:**
1. Check `USE_MOCK` setting in `api.js`
2. If using real backend, verify `/api/pipeline/start` endpoint
3. Check browser console for errors
4. Ensure no other pipeline job is running

### Filters Not Working

**Symptoms:** Filters don't affect graph

**Solutions:**
1. Ensure graph is rendered before applying filters
2. Check that filter values are valid
3. Try "Clear Filters" button and reapply
4. Check browser console for JavaScript errors

### Performance Issues with Large Graphs

**Symptoms:** Slow rendering, laggy interactions

**Solutions:**
1. Use confidence threshold to reduce visible nodes
2. Apply node/edge type filters
3. Use simpler layouts (Grid instead of COSE)
4. Consider implementing pagination (requires backend support)

## Browser Compatibility

| Browser | Version | Status |
|---------|---------|--------|
| Chrome | 90+ | ✅ Fully supported |
| Firefox | 88+ | ✅ Fully supported |
| Safari | 14+ | ✅ Fully supported |
| Edge | 90+ | ✅ Fully supported |
| Internet Explorer | Any | ❌ Not supported |

## Dependencies

- **Cytoscape.js** (v3.28.1): Graph visualization library
  - Loaded via CDN
  - No npm/build step required
  - Can be swapped out by modifying `kg-viz.js`

## Performance Considerations

- **Graph size**: Tested up to 5,000 nodes/edges
- **Rendering**: Force-directed layouts are most expensive
- **Filtering**: Pre-render filtering recommended for >3,000 elements
- **Memory**: ~50-100MB for typical knowledge graphs

## Security Notes

- **Local context**: Designed for local/internal use
- **File validation**: Client-side only (always validate server-side)
- **No authentication**: Add authentication layer for production
- **XSS protection**: User inputs are escaped
- **CORS**: May need to configure CORS headers on backend

## Roadmap

- [ ] Advanced query builder for complex filters
- [ ] Subgraph pinning and comparison
- [ ] Layout persistence (save/load custom layouts)
- [ ] PDF preview integration
- [ ] Real-time collaboration features
- [ ] Graph diff view (compare versions)
- [ ] Export to Neo4j, RDF, GraphML

## Contributing

To contribute to the frontend:

1. Follow the modular architecture
2. Keep dependencies minimal
3. Comment public APIs
4. Test on multiple browsers
5. Ensure responsive design works

## License

Part of the Knowledge Extraction Pipeline research project.

---

**Version:** 1.0
**Last Updated:** November 2025
**Maintainer:** Knowledge Graph Team
