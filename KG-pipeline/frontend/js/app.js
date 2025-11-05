/**
 * Main Application Module
 *
 * Orchestrates the entire application: state management, UI events,
 * API communication, and graph visualization coordination.
 */

const App = (function() {
  'use strict';

  // Application state
  const state = {
    config: null,
    i18n: null,
    api: null,
    viz: null,
    filesInQueue: [],
    currentJob: null,
    kgData: null,
    stats: null,
    filters: {
      nodeTypes: [],
      edgeTypes: [],
      minConfidence: 0.0,
      searchTerm: ''
    },
    selectedNode: null,
    selectedEdge: null,
    pollingInterval: null
  };

  /**
   * Initialize application
   */
  async function init() {
    try {
      // Load configuration
      state.config = await loadConfig();

      // Initialize i18n
      state.i18n = await loadI18n(state.config.defaultLanguage || 'en');

      // Initialize API
      state.api = API.create(state.config);

      // Initialize UI
      initUI();

      // Initialize graph visualization
      const vizContainer = document.getElementById('graph-canvas');
      state.viz = KGViz.create(vizContainer, state.config);

      // Attach event listeners
      attachEventListeners();

      // Load initial data
      await loadKnowledgeGraph();

      showToast(t('app_initialized'), 'success');
    } catch (error) {
      console.error('Initialization error:', error);
      showToast(t('init_error') + ': ' + error.message, 'error');
    }
  }

  /**
   * Load configuration from config.json
   */
  async function loadConfig() {
    const response = await fetch('config.json');
    return response.json();
  }

  /**
   * Load i18n translations
   */
  async function loadI18n(lang) {
    try {
      const response = await fetch(`i18n/${lang}.json`);
      return response.json();
    } catch (error) {
      console.warn(`Could not load translations for ${lang}, using fallback`);
      return {};
    }
  }

  /**
   * Initialize UI components
   */
  function initUI() {
    // Populate upload target selector
    const targetSelect = document.getElementById('upload-target');
    Object.entries(state.config.uploadTargets).forEach(([key, target]) => {
      const option = document.createElement('option');
      option.value = key;
      option.textContent = target.label;
      option.title = target.description;
      targetSelect.appendChild(option);
    });

    // Populate node type filters
    const nodeFilterContainer = document.getElementById('node-type-filters');
    Object.keys(state.config.nodeTypes).forEach(type => {
      const label = document.createElement('label');
      label.className = 'filter-checkbox';
      label.innerHTML = `
        <input type="checkbox" value="${type}" checked>
        <span>${state.config.nodeTypes[type].icon} ${type}</span>
      `;
      nodeFilterContainer.appendChild(label);
    });

    // Populate edge type filters
    const edgeFilterContainer = document.getElementById('edge-type-filters');
    Object.keys(state.config.edgeTypes).forEach(type => {
      const label = document.createElement('label');
      label.className = 'filter-checkbox';
      label.innerHTML = `
        <input type="checkbox" value="${type}" checked>
        <span>${type}</span>
      `;
      edgeFilterContainer.appendChild(label);
    });

    // Setup drag and drop
    setupDragAndDrop();
  }

  /**
   * Attach event listeners
   */
  function attachEventListeners() {
    // File upload
    document.getElementById('file-input').addEventListener('change', handleFileSelect);
    document.getElementById('select-files-btn').addEventListener('click', () => {
      document.getElementById('file-input').click();
    });
    document.getElementById('upload-btn').addEventListener('click', handleUpload);
    document.getElementById('clear-queue-btn').addEventListener('click', clearFileQueue);

    // Pipeline
    document.getElementById('start-pipeline-btn').addEventListener('click', handleStartPipeline);

    // Graph controls
    document.getElementById('layout-select').addEventListener('change', handleLayoutChange);
    document.getElementById('search-input').addEventListener('input', debounce(handleSearch, 300));
    document.getElementById('confidence-slider').addEventListener('input', handleConfidenceFilter);
    document.getElementById('clear-filters-btn').addEventListener('click', clearFilters);
    document.getElementById('reset-view-btn').addEventListener('click', () => state.viz.clearFocus());

    // Node/edge type filters
    document.querySelectorAll('#node-type-filters input').forEach(checkbox => {
      checkbox.addEventListener('change', applyFilters);
    });
    document.querySelectorAll('#edge-type-filters input').forEach(checkbox => {
      checkbox.addEventListener('change', applyFilters);
    });

    // Export buttons
    document.getElementById('export-png-btn').addEventListener('click', handleExportPNG);
    document.getElementById('export-json-btn').addEventListener('click', handleExportJSON);
    document.getElementById('export-csv-btn').addEventListener('click', handleExportCSV);

    // Graph visualization events
    state.viz.on('nodeSelected', handleNodeSelected);
    state.viz.on('edgeSelected', handleEdgeSelected);
    state.viz.on('rendered', updateGraphStats);

    // Section navigation
    document.querySelectorAll('[data-section]').forEach(link => {
      link.addEventListener('click', (e) => {
        e.preventDefault();
        showSection(link.dataset.section);
      });
    });

    // Handle URL hash navigation
    window.addEventListener('hashchange', handleHashChange);
  }

  // ===== File Upload =====

  function setupDragAndDrop() {
    const dropZone = document.getElementById('drop-zone');

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
      dropZone.addEventListener(eventName, preventDefaults, false);
    });

    ['dragenter', 'dragover'].forEach(eventName => {
      dropZone.addEventListener(eventName, () => dropZone.classList.add('drag-over'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
      dropZone.addEventListener(eventName, () => dropZone.classList.remove('drag-over'), false);
    });

    dropZone.addEventListener('drop', handleDrop, false);
  }

  function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
  }

  function handleDrop(e) {
    const files = e.dataTransfer.files;
    addFilesToQueue(files);
  }

  function handleFileSelect(e) {
    const files = e.target.files;
    addFilesToQueue(files);
  }

  function addFilesToQueue(files) {
    Array.from(files).forEach(file => {
      // Validate file
      if (!validateFile(file)) return;

      state.filesInQueue.push({
        file,
        id: generateId(),
        status: 'pending',
        progress: 0
      });
    });

    updateFileQueueUI();
  }

  function validateFile(file) {
    // Check MIME type
    if (!state.config.allowedMimeTypes.includes(file.type)) {
      showToast(`${file.name}: ${t('invalid_file_type')}`, 'error');
      return false;
    }

    // Check file size
    if (file.size > state.config.maxFileSize) {
      showToast(`${file.name}: ${t('file_too_large')}`, 'error');
      return false;
    }

    // Check queue size
    if (state.filesInQueue.length >= state.config.maxFilesPerBatch) {
      showToast(t('queue_full'), 'error');
      return false;
    }

    return true;
  }

  function updateFileQueueUI() {
    const queueList = document.getElementById('file-queue');
    queueList.innerHTML = '';

    if (state.filesInQueue.length === 0) {
      queueList.innerHTML = `<div class="empty-message">${t('no_files_queued')}</div>`;
      document.getElementById('upload-btn').disabled = true;
      document.getElementById('clear-queue-btn').disabled = true;
      return;
    }

    document.getElementById('upload-btn').disabled = false;
    document.getElementById('clear-queue-btn').disabled = false;

    state.filesInQueue.forEach((item, index) => {
      const fileItem = document.createElement('div');
      fileItem.className = `file-item status-${item.status}`;
      fileItem.innerHTML = `
        <div class="file-info">
          <span class="file-name">${item.file.name}</span>
          <span class="file-size">${formatFileSize(item.file.size)}</span>
        </div>
        <div class="file-status">
          <span class="status-badge">${t('status_' + item.status)}</span>
          <button class="btn-remove" onclick="App.removeFileFromQueue(${index})">×</button>
        </div>
      `;
      queueList.appendChild(fileItem);
    });
  }

  async function handleUpload() {
    const targetFolder = document.getElementById('upload-target').value;
    if (!targetFolder) {
      showToast(t('select_target_folder'), 'error');
      return;
    }

    const pendingFiles = state.filesInQueue.filter(item => item.status === 'pending');
    if (pendingFiles.length === 0) {
      showToast(t('no_files_to_upload'), 'warning');
      return;
    }

    document.getElementById('upload-btn').disabled = true;

    try {
      const files = pendingFiles.map(item => item.file);
      const result = await state.api.uploadFiles(files, targetFolder);

      if (result.success) {
        pendingFiles.forEach(item => {
          item.status = 'completed';
        });
        showToast(t('upload_success') + `: ${result.data.count} ${t('files')}`, 'success');
      } else {
        throw new Error(result.message);
      }
    } catch (error) {
      showToast(t('upload_error') + ': ' + error.message, 'error');
      state.filesInQueue.forEach(item => {
        if (item.status === 'uploading') {
          item.status = 'error';
        }
      });
    }

    updateFileQueueUI();
    document.getElementById('upload-btn').disabled = false;
  }

  function clearFileQueue() {
    state.filesInQueue = [];
    updateFileQueueUI();
  }

  function removeFileFromQueue(index) {
    state.filesInQueue.splice(index, 1);
    updateFileQueueUI();
  }

  // ===== Pipeline =====

  async function handleStartPipeline() {
    try {
      const result = await state.api.startPipeline();

      if (result.success) {
        state.currentJob = result.data;
        document.getElementById('job-id').textContent = result.data.jobId;
        document.getElementById('job-status').style.display = 'block';
        showToast(t('pipeline_started'), 'success');

        // Start polling
        startJobPolling();
      } else {
        throw new Error(result.message);
      }
    } catch (error) {
      showToast(t('pipeline_start_error') + ': ' + error.message, 'error');
    }
  }

  function startJobPolling() {
    if (state.pollingInterval) {
      clearInterval(state.pollingInterval);
    }

    state.pollingInterval = setInterval(async () => {
      await updateJobStatus();
    }, state.config.pollingInterval);
  }

  async function updateJobStatus() {
    if (!state.currentJob) return;

    try {
      const result = await state.api.getJobStatus(state.currentJob.jobId);

      if (result.success) {
        const status = result.data;

        // Update progress bar
        const progressBar = document.getElementById('progress-bar');
        const progressText = document.getElementById('progress-text');
        progressBar.style.width = status.progress + '%';
        progressText.textContent = `${status.phase} (${Math.round(status.progress)}%)`;

        // Update logs
        if (status.logs && status.logs.length > 0) {
          const logsContainer = document.getElementById('job-logs');
          logsContainer.innerHTML = status.logs.slice(-10).map(log =>
            `<div class="log-entry">${escapeHtml(log)}</div>`
          ).join('');
          logsContainer.scrollTop = logsContainer.scrollHeight;
        }

        // Check if completed
        if (status.status === 'completed') {
          clearInterval(state.pollingInterval);
          state.pollingInterval = null;
          showToast(t('pipeline_completed'), 'success');

          // Reload KG
          await loadKnowledgeGraph();
        } else if (status.status === 'failed') {
          clearInterval(state.pollingInterval);
          state.pollingInterval = null;
          showToast(t('pipeline_failed'), 'error');
        }
      }
    } catch (error) {
      console.error('Job status update error:', error);
    }
  }

  // ===== Knowledge Graph =====

  async function loadKnowledgeGraph() {
    try {
      const result = await state.api.getKG();

      if (result.success) {
        state.kgData = result.data;
        state.viz.render(state.kgData);
        showToast(t('kg_loaded'), 'success');

        // Load stats
        await loadStats();
      } else {
        throw new Error(result.message);
      }
    } catch (error) {
      console.error('KG load error:', error);
      showToast(t('kg_load_error') + ': ' + error.message, 'error');
    }
  }

  async function loadStats() {
    try {
      const result = await state.api.getStats();

      if (result.success) {
        state.stats = result.data;
        updateStatsUI();
      }
    } catch (error) {
      console.error('Stats load error:', error);
    }
  }

  function updateStatsUI() {
    if (!state.stats) return;

    document.getElementById('stat-nodes').textContent = state.stats.nodeCount;
    document.getElementById('stat-edges').textContent = state.stats.edgeCount;
    document.getElementById('stat-entity-types').textContent = Object.keys(state.stats.entityTypes).length;
    document.getElementById('stat-relation-types').textContent = Object.keys(state.stats.relationTypes).length;
    document.getElementById('stat-avg-confidence').textContent =
      (state.stats.avgEntityConfidence * 100).toFixed(1) + '%';
  }

  function updateGraphStats() {
    const stats = state.viz.getStats();
    document.getElementById('graph-stats').innerHTML = `
      <span>${t('visible')}: ${stats.visibleNodes} ${t('nodes')}, ${stats.visibleEdges} ${t('edges')}</span>
    `;
  }

  // ===== Filters =====

  function applyFilters() {
    const nodeTypes = Array.from(
      document.querySelectorAll('#node-type-filters input:checked')
    ).map(cb => cb.value);

    const edgeTypes = Array.from(
      document.querySelectorAll('#edge-type-filters input:checked')
    ).map(cb => cb.value);

    state.filters.nodeTypes = nodeTypes.length < Object.keys(state.config.nodeTypes).length ? nodeTypes : [];
    state.filters.edgeTypes = edgeTypes.length < Object.keys(state.config.edgeTypes).length ? edgeTypes : [];

    state.viz.applyFilters(state.filters);
  }

  function handleConfidenceFilter(e) {
    const value = parseFloat(e.target.value);
    state.filters.minConfidence = value;
    document.getElementById('confidence-value').textContent = value.toFixed(2);
    state.viz.applyFilters(state.filters);
  }

  function handleSearch(e) {
    state.filters.searchTerm = e.target.value;
    state.viz.applyFilters(state.filters);
  }

  function clearFilters() {
    // Reset checkboxes
    document.querySelectorAll('#node-type-filters input, #edge-type-filters input').forEach(cb => {
      cb.checked = true;
    });

    // Reset confidence slider
    document.getElementById('confidence-slider').value = 0;
    document.getElementById('confidence-value').textContent = '0.00';

    // Reset search
    document.getElementById('search-input').value = '';

    // Reset state
    state.filters = {
      nodeTypes: [],
      edgeTypes: [],
      minConfidence: 0.0,
      searchTerm: ''
    };

    state.viz.applyFilters(state.filters);
  }

  function handleLayoutChange(e) {
    state.viz.changeLayout(e.target.value);
  }

  // ===== Node/Edge Selection =====

  function handleNodeSelected(nodeData) {
    state.selectedNode = nodeData;
    state.selectedEdge = null;
    showDetailsPanel('node', nodeData);
  }

  function handleEdgeSelected(edgeData) {
    state.selectedEdge = edgeData;
    state.selectedNode = null;
    showDetailsPanel('edge', edgeData);
  }

  function showDetailsPanel(type, data) {
    const panel = document.getElementById('details-panel');
    panel.style.display = 'block';

    let html = `<h3>${type === 'node' ? t('node_details') : t('edge_details')}</h3>`;

    if (type === 'node') {
      const icon = state.config.nodeTypes[data.type]?.icon || '●';
      html += `
        <div class="detail-item"><strong>${t('type')}:</strong> ${icon} ${data.type}</div>
        <div class="detail-item"><strong>${t('id')}:</strong> ${data.id}</div>
        <div class="detail-item"><strong>${t('label')}:</strong> ${data.label || data.name}</div>
        <div class="detail-item"><strong>${t('confidence')}:</strong> ${(data.confidence * 100).toFixed(1)}%</div>
      `;

      if (data.spans && data.spans.length > 0) {
        html += `<div class="detail-item"><strong>${t('sources')}:</strong></div><ul>`;
        data.spans.forEach(span => {
          html += `<li>${span.section_title} (p.${span.page_start})</li>`;
        });
        html += `</ul>`;
      }
    } else {
      html += `
        <div class="detail-item"><strong>${t('type')}:</strong> ${data.type}</div>
        <div class="detail-item"><strong>${t('from')}:</strong> ${data.source}</div>
        <div class="detail-item"><strong>${t('to')}:</strong> ${data.target}</div>
        <div class="detail-item"><strong>${t('confidence')}:</strong> ${(data.confidence * 100).toFixed(1)}%</div>
      `;
    }

    html += `<button class="btn-close" onclick="App.closeDetailsPanel()">×</button>`;
    panel.innerHTML = html;
  }

  function closeDetailsPanel() {
    document.getElementById('details-panel').style.display = 'none';
  }

  // ===== Export =====

  function handleExportPNG() {
    const dataUrl = state.viz.exportPNG();
    downloadFile(dataUrl, 'knowledge_graph.png');
    showToast(t('export_success'), 'success');
  }

  function handleExportJSON() {
    const json = JSON.stringify(state.kgData, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    downloadFile(url, 'knowledge_graph.json');
    showToast(t('export_success'), 'success');
  }

  function handleExportCSV() {
    const csv = convertToCSV(state.kgData);
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    downloadFile(url, 'knowledge_graph.csv');
    showToast(t('export_success'), 'success');
  }

  function convertToCSV(data) {
    let csv = 'Type,ID,Label,Confidence\n';

    data.nodes.forEach(node => {
      csv += `Node,${node.id},${node.label},${node.confidence}\n`;
    });

    data.edges.forEach(edge => {
      csv += `Edge,${edge.id},${edge.type},${edge.confidence}\n`;
    });

    return csv;
  }

  // ===== Navigation =====

  function showSection(sectionId) {
    document.querySelectorAll('.section').forEach(section => {
      section.classList.remove('active');
    });

    const section = document.getElementById(sectionId);
    if (section) {
      section.classList.add('active');
      window.location.hash = sectionId;
    }
  }

  function handleHashChange() {
    const hash = window.location.hash.slice(1);
    if (hash) {
      showSection(hash);
    }
  }

  // ===== Utilities =====

  function t(key) {
    return state.i18n[key] || key;
  }

  function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    document.body.appendChild(toast);

    setTimeout(() => toast.classList.add('show'), 10);
    setTimeout(() => {
      toast.classList.remove('show');
      setTimeout(() => toast.remove(), 300);
    }, 3000);
  }

  function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
  }

  function generateId() {
    return Date.now().toString(36) + Math.random().toString(36).substr(2);
  }

  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
      const later = () => {
        clearTimeout(timeout);
        func(...args);
      };
      clearTimeout(timeout);
      timeout = setTimeout(later, wait);
    };
  }

  function downloadFile(url, filename) {
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }

  // Public API
  return {
    init,
    removeFileFromQueue,
    closeDetailsPanel
  };
})();

// Initialize on DOM ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', App.init);
} else {
  App.init();
}
