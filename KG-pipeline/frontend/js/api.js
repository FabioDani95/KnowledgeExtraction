/**
 * API Adapter Module
 *
 * Encapsulates all backend communication with configurable endpoints.
 * Provides stub implementations for easy integration with actual backend.
 *
 * Usage:
 *   const api = API.create(config);
 *   api.uploadFiles(files, 'product_technical').then(result => ...);
 */

const API = (function() {
  'use strict';

  /**
   * Create API instance with configuration
   * @param {Object} config - Configuration object with apiBaseUrl
   * @returns {Object} API interface
   */
  function create(config) {
    const baseUrl = config.apiBaseUrl || 'http://localhost:8000';
    const USE_MOCK = true; // Set to false when real backend is available

    /**
     * Generic HTTP request handler
     * @param {string} endpoint - API endpoint path
     * @param {Object} options - Fetch options
     * @returns {Promise<Object>} Response data
     */
    async function request(endpoint, options = {}) {
      const url = `${baseUrl}${endpoint}`;
      const defaultOptions = {
        headers: {
          'Content-Type': 'application/json',
          ...options.headers
        },
        ...options
      };

      try {
        const response = await fetch(url, defaultOptions);

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const data = await response.json();
        return { success: true, data };
      } catch (error) {
        console.error(`API Error [${endpoint}]:`, error);
        return {
          success: false,
          error: error.message,
          message: `Failed to ${endpoint}: ${error.message}`
        };
      }
    }

    /**
     * Upload files to target folder
     * @param {FileList|Array} files - Files to upload
     * @param {string} targetFolder - Destination folder key
     * @returns {Promise<Object>} Upload result with status per file
     */
    async function uploadFiles(files, targetFolder) {
      if (USE_MOCK) {
        // Mock implementation
        await delay(1500);
        return {
          success: true,
          data: {
            uploaded: Array.from(files).map(f => ({
              name: f.name,
              size: f.size,
              status: 'success',
              path: `${targetFolder}/${f.name}`
            })),
            targetFolder,
            count: files.length
          }
        };
      }

      const formData = new FormData();
      Array.from(files).forEach(file => formData.append('files', file));
      formData.append('targetFolder', targetFolder);

      return request('/api/upload', {
        method: 'POST',
        body: formData,
        headers: {} // Let browser set Content-Type for FormData
      });
    }

    /**
     * Start pipeline execution
     * @param {Object} payload - Pipeline configuration (optional)
     * @returns {Promise<Object>} Job info with jobId
     */
    async function startPipeline(payload = {}) {
      if (USE_MOCK) {
        await delay(500);
        const jobId = `job_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        return {
          success: true,
          data: {
            jobId,
            status: 'started',
            startedAt: new Date().toISOString()
          }
        };
      }

      return request('/api/pipeline/start', {
        method: 'POST',
        body: JSON.stringify(payload)
      });
    }

    /**
     * Get pipeline job status
     * @param {string} jobId - Job identifier
     * @returns {Promise<Object>} Job status with phase, progress, logs
     */
    async function getJobStatus(jobId) {
      if (USE_MOCK) {
        await delay(300);
        const mockPhases = [
          'Parsing PDF documents',
          'Symbolic extraction',
          'Neural entity extraction',
          'Relationship promotion',
          'Knowledge graph merging',
          'Validation & quality checks',
          'Export final KG'
        ];

        const elapsed = Date.now() - parseInt(jobId.split('_')[1]);
        const phaseIndex = Math.min(
          Math.floor(elapsed / 5000),
          mockPhases.length - 1
        );
        const isComplete = phaseIndex >= mockPhases.length - 1 && elapsed > 35000;

        return {
          success: true,
          data: {
            jobId,
            phase: mockPhases[phaseIndex],
            phaseIndex,
            totalPhases: mockPhases.length,
            progress: Math.min(95, ((phaseIndex + 1) / mockPhases.length) * 100),
            status: isComplete ? 'completed' : 'running',
            logs: [
              `[${new Date().toISOString()}] Processing pipeline phase: ${mockPhases[phaseIndex]}`,
              `[${new Date().toISOString()}] Files processed: ${Math.floor(Math.random() * 10) + 1}`,
              `[${new Date().toISOString()}] Entities extracted: ${Math.floor(Math.random() * 500) + 100}`
            ],
            error: null
          }
        };
      }

      return request(`/api/pipeline/status/${jobId}`);
    }

    /**
     * Fetch knowledge graph data
     * @returns {Promise<Object>} KG data with nodes and edges
     */
    async function getKG() {
      if (USE_MOCK) {
        await delay(800);

        // Try to load actual KG from filesystem (if available via fetch)
        try {
          const response = await fetch('../output/merged_kg/kg_merged.json');
          if (response.ok) {
            const kg = await response.json();
            return {
              success: true,
              data: transformKGToGraph(kg)
            };
          }
        } catch (e) {
          console.log('Could not load actual KG, using mock data');
        }

        // Fallback to mock data
        return {
          success: true,
          data: getMockKG()
        };
      }

      const result = await request('/api/kg/current');
      if (result.success && result.data) {
        result.data = transformKGToGraph(result.data);
      }
      return result;
    }

    /**
     * Transform backend KG format to graph structure
     * @param {Object} kg - Knowledge graph from backend
     * @returns {Object} Graph data with nodes and edges
     */
    function transformKGToGraph(kg) {
      const nodes = (kg.entities || []).map(entity => ({
        id: entity.id,
        type: entity.type,
        label: entity.name,
        confidence: entity.confidence || 1.0,
        attributes: entity.attributes || {},
        spans: entity.spans || [],
        ...entity
      }));

      const edges = (kg.relations || []).map((rel, idx) => ({
        id: `edge_${idx}`,
        source: rel.from_ref,
        target: rel.to_ref,
        type: rel.type,
        confidence: rel.confidence || 1.0,
        attributes: rel.attributes || {},
        ...rel
      }));

      return {
        nodes,
        edges,
        meta: {
          document_code: kg.document_code,
          datasource_code: kg.datasource_code,
          merge_info: kg.merge_info,
          extraction_version: kg.extraction_version
        }
      };
    }

    /**
     * Get statistics about the knowledge graph
     * @returns {Promise<Object>} Statistics data
     */
    async function getStats() {
      if (USE_MOCK) {
        await delay(400);
        return {
          success: true,
          data: {
            nodeCount: 156,
            edgeCount: 234,
            entityTypes: {
              'Product': 3,
              'Component': 25,
              'ComponentType': 18,
              'ParameterSpec': 45,
              'Unit': 12,
              'MachineMode': 8,
              'State': 6,
              'FailureMode': 15,
              'RepairAction': 12,
              'Tool': 8,
              'TestCase': 4
            },
            relationTypes: {
              'hasPart': 35,
              'hasSpec': 42,
              'hasUnit': 40,
              'instanceOf': 20,
              'precedes': 15,
              'mitigatedBy': 18,
              'requiresTool': 10,
              'validatedBy': 8,
              'constrainedBy': 6
            },
            avgEntityConfidence: 0.87,
            avgRelationConfidence: 0.83,
            duplicatesCollapsed: 23,
            lastUpdated: new Date().toISOString()
          }
        };
      }

      return request('/api/kg/stats');
    }

    /**
     * Get configuration from backend
     * @returns {Promise<Object>} Backend configuration
     */
    async function getConfig() {
      if (USE_MOCK) {
        await delay(200);
        // Return local config
        return {
          success: true,
          data: config
        };
      }

      return request('/api/config');
    }

    /**
     * Download pipeline logs
     * @param {string} jobId - Job identifier
     * @returns {Promise<Object>} Log file data
     */
    async function downloadLogs(jobId) {
      if (USE_MOCK) {
        await delay(500);
        return {
          success: true,
          data: {
            logs: `Pipeline Execution Log - Job ${jobId}\n${'='.repeat(50)}\n\nPipeline completed successfully.\n\nPhase 1: PDF Parsing\n- Processed 4 documents\n- Extracted 234 pages\n\nPhase 2: Symbolic Extraction\n- Identified 89 entities\n- Created 112 relationships\n\nPhase 3: Neural Extraction\n- Enhanced 156 entities\n- Added 122 relationships\n\nPhase 4: Merge\n- Deduplicated 23 entities\n- Final graph: 156 nodes, 234 edges\n`,
            filename: `pipeline_log_${jobId}.txt`
          }
        };
      }

      return request(`/api/logs/${jobId}`);
    }

    // Helper functions

    function delay(ms) {
      return new Promise(resolve => setTimeout(resolve, ms));
    }

    function getMockKG() {
      return {
        nodes: [
          { id: 'ns:Product/citiz', type: 'Product', label: 'Citiz Coffee Machine', confidence: 0.95 },
          { id: 'ns:ComponentType/thermoblock', type: 'ComponentType', label: 'Thermoblock', confidence: 0.92 },
          { id: 'ns:Component/pump_cp4', type: 'Component', label: 'Pump CP4', confidence: 0.90 },
          { id: 'ns:ParameterSpec/pressure', type: 'ParameterSpec', label: 'Pressure: 19 bar', confidence: 0.88 },
          { id: 'ns:Unit/bar', type: 'Unit', label: 'bar', confidence: 0.99 },
          { id: 'ns:Mode/brewing_mode', type: 'MachineMode', label: 'Brewing Mode', confidence: 0.93 },
          { id: 'ns:State/ready', type: 'State', label: 'Ready State', confidence: 0.91 },
          { id: 'ns:FM/no_water_flow', type: 'FailureMode', label: 'No Water Flow', confidence: 0.87 },
          { id: 'ns:RA/check_pump', type: 'RepairAction', label: 'Check Pump', confidence: 0.85 },
          { id: 'ns:Tool/multimeter', type: 'Tool', label: 'Multimeter', confidence: 0.96 }
        ],
        edges: [
          { id: 'e1', source: 'ns:Product/citiz', target: 'ns:ComponentType/thermoblock', type: 'hasPart' },
          { id: 'e2', source: 'ns:Product/citiz', target: 'ns:Component/pump_cp4', type: 'hasPart' },
          { id: 'e3', source: 'ns:Component/pump_cp4', target: 'ns:ParameterSpec/pressure', type: 'hasSpec' },
          { id: 'e4', source: 'ns:ParameterSpec/pressure', target: 'ns:Unit/bar', type: 'hasUnit' },
          { id: 'e5', source: 'ns:Mode/brewing_mode', target: 'ns:State/ready', type: 'precedes' },
          { id: 'e6', source: 'ns:FM/no_water_flow', target: 'ns:RA/check_pump', type: 'mitigatedBy' },
          { id: 'e7', source: 'ns:RA/check_pump', target: 'ns:Tool/multimeter', type: 'requiresTool' }
        ],
        meta: {
          document_code: 'KG_MERGED',
          datasource_code: 'MOCK_DATA'
        }
      };
    }

    // Public API
    return {
      uploadFiles,
      startPipeline,
      getJobStatus,
      getKG,
      getStats,
      getConfig,
      downloadLogs
    };
  }

  return { create };
})();
