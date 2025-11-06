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
     * @param {string} targetFolder - Destination folder key (e.g., "product_technical")
     * @returns {Promise<Object>} Upload result with status per file
     */
    async function uploadFiles(files, targetFolder) {
      if (USE_MOCK) {
        // In mock mode, simulate file storage by sending to backend helper
        // This will actually copy files to the correct source folders
        await delay(1500);

        try {
          // Send files to backend upload endpoint
          const formData = new FormData();
          Array.from(files).forEach(file => formData.append('files', file));
          formData.append('targetFolder', targetFolder);

          // Try to upload via backend helper script
          const response = await fetch('http://localhost:8000/upload', {
            method: 'POST',
            body: formData
          });

          if (response.ok) {
            const result = await response.json();
            return {
              success: true,
              data: result
            };
          } else {
            throw new Error('Backend upload endpoint not available');
          }
        } catch (error) {
          // If backend not available, show warning but allow to continue
          console.warn('Backend upload not available. Files saved in browser memory only.');
          return {
            success: true,
            data: {
              uploaded: Array.from(files).map(f => ({
                name: f.name,
                size: f.size,
                status: 'success',
                path: `source/${targetFolder}/${f.name}`,
                warning: 'Saved in browser only - restart backend upload service'
              })),
              targetFolder,
              count: files.length,
              warning: 'Backend upload service not running. Files are in browser memory only.'
            }
          };
        }
      }

      // Real backend mode
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
        const isComplete = phaseIndex >= mockPhases.length - 1 && elapsed > 38000;

        // Calculate progress: complete phases contribute 100%, current phase adds partial
        let progress;
        if (isComplete) {
          progress = 100;
        } else {
          const completedPhases = phaseIndex;
          const currentPhaseProgress = ((elapsed % 5000) / 5000) * (100 / mockPhases.length);
          progress = Math.min(99, (completedPhases / mockPhases.length) * 100 + currentPhaseProgress);
        }

        return {
          success: true,
          data: {
            jobId,
            phase: isComplete ? 'Completed' : mockPhases[phaseIndex],
            phaseIndex,
            totalPhases: mockPhases.length,
            progress: Math.round(progress),
            status: isComplete ? 'completed' : 'running',
            logs: [
              `[${new Date().toISOString()}] ${isComplete ? 'Pipeline completed successfully!' : 'Processing pipeline phase: ' + mockPhases[phaseIndex]}`,
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
        await delay(500);

        // PRIORITY: Try backend server first (this loads the REAL KG)
        try {
          console.log('🔍 Attempting to load KG from backend server...');
          const response = await fetch('http://localhost:8000/kg/current');
          if (response.ok) {
            const data = await response.json();
            // If response has success wrapper, unwrap it
            const kg = data.success ? data.data : data;
            const nodeCount = kg.entities?.length || 0;
            const edgeCount = kg.relations?.length || 0;
            console.log(`✅ Loaded REAL KG from backend: ${nodeCount} entities, ${edgeCount} relations`);
            return {
              success: true,
              data: transformKGToGraph(kg)
            };
          }
        } catch (e) {
          console.warn('❌ Backend server not available:', e.message);
        }

        // Try file paths as fallback
        const filePaths = [
          '../output/merged_kg/kg_merged.json',
          '../../output/merged_kg/kg_merged.json'
        ];

        for (const path of filePaths) {
          try {
            const response = await fetch(path);
            if (response.ok) {
              const kg = await response.json();
              const nodeCount = kg.entities?.length || 0;
              const edgeCount = kg.relations?.length || 0;
              console.log(`✅ Loaded REAL KG from file ${path}: ${nodeCount} entities, ${edgeCount} relations`);
              return {
                success: true,
                data: transformKGToGraph(kg)
              };
            }
          } catch (e) {
            console.log(`❌ Could not load KG from ${path}`);
          }
        }

        // LAST RESORT: Use mock data
        console.error('⚠️ WARNING: Could not load actual KG from any source. Using mock data.');
        console.error('⚠️ Make sure upload_server.py is running: python3 frontend/upload_server.py');
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

        // PRIORITY: Try to get real stats from backend server
        try {
          console.log('🔍 Attempting to load stats from backend server...');
          const response = await fetch('http://localhost:8000/kg/stats');
          if (response.ok) {
            const data = await response.json();
            if (data.success) {
              console.log(`✅ Loaded REAL stats from backend: ${data.data.nodeCount} nodes, ${data.data.edgeCount} edges`);
              return data;
            }
          }
        } catch (e) {
          console.warn('❌ Could not load stats from backend:', e.message);
        }

        // Fallback to mock stats
        console.warn('⚠️ Using mock statistics');
        return {
          success: true,
          data: {
            nodeCount: 44,
            edgeCount: 28,
            entityTypes: {
              'Product': 2,
              'Component': 6,
              'ComponentType': 4,
              'ParameterSpec': 5,
              'Unit': 5,
              'MachineMode': 4,
              'State': 4,
              'FailureMode': 4,
              'RepairAction': 4,
              'Tool': 3,
              'TestCase': 3
            },
            relationTypes: {
              'hasPart': 8,
              'hasSpec': 6,
              'hasUnit': 5,
              'instanceOf': 4,
              'precedes': 3,
              'mitigatedBy': 2
            },
            avgEntityConfidence: 0.87,
            avgRelationConfidence: 0.83,
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
      // Expanded mock data with more variety for better testing
      const nodes = [
        // Products
        { id: 'ns:Product/citiz', type: 'Product', label: 'Citiz Coffee Machine', confidence: 0.95 },
        { id: 'ns:Product/milk_professional', type: 'Product', label: 'Citiz & Milk Professional', confidence: 0.94 },

        // Component Types
        { id: 'ns:ComponentType/thermoblock', type: 'ComponentType', label: 'Thermoblock', confidence: 0.92 },
        { id: 'ns:ComponentType/water_pump', type: 'ComponentType', label: 'Water Pump', confidence: 0.91 },
        { id: 'ns:ComponentType/heating_element', type: 'ComponentType', label: 'Heating Element', confidence: 0.90 },
        { id: 'ns:ComponentType/pcb', type: 'ComponentType', label: 'PCB Control Board', confidence: 0.93 },

        // Components
        { id: 'ns:Component/pump_cp4', type: 'Component', label: 'Pump CP4', confidence: 0.90 },
        { id: 'ns:Component/pump_ulka', type: 'Component', label: 'Pump ULKA EP5', confidence: 0.89 },
        { id: 'ns:Component/thermoblock_nespresso', type: 'Component', label: 'Thermoblock Nespresso', confidence: 0.91 },
        { id: 'ns:Component/pcb_main', type: 'Component', label: 'Main PCB v2.3', confidence: 0.92 },
        { id: 'ns:Component/water_tank', type: 'Component', label: 'Water Tank 1L', confidence: 0.88 },
        { id: 'ns:Component/capsule_container', type: 'Component', label: 'Capsule Container', confidence: 0.87 },

        // Parameter Specs
        { id: 'ns:ParameterSpec/pressure_19bar', type: 'ParameterSpec', label: 'Pressure: 19 bar', confidence: 0.88 },
        { id: 'ns:ParameterSpec/voltage_230v', type: 'ParameterSpec', label: 'Voltage: 230 V', confidence: 0.90 },
        { id: 'ns:ParameterSpec/power_1260w', type: 'ParameterSpec', label: 'Power: 1260 W', confidence: 0.89 },
        { id: 'ns:ParameterSpec/temp_92c', type: 'ParameterSpec', label: 'Temperature: 92 °C', confidence: 0.87 },
        { id: 'ns:ParameterSpec/capacity_1l', type: 'ParameterSpec', label: 'Capacity: 1 L', confidence: 0.86 },

        // Units
        { id: 'ns:Unit/bar', type: 'Unit', label: 'bar', confidence: 0.99 },
        { id: 'ns:Unit/v', type: 'Unit', label: 'V', confidence: 0.99 },
        { id: 'ns:Unit/w', type: 'Unit', label: 'W', confidence: 0.99 },
        { id: 'ns:Unit/celsius', type: 'Unit', label: '°C', confidence: 0.99 },
        { id: 'ns:Unit/liter', type: 'Unit', label: 'L', confidence: 0.99 },

        // Machine Modes
        { id: 'ns:Mode/brewing_mode', type: 'MachineMode', label: 'Brewing Mode', confidence: 0.93 },
        { id: 'ns:Mode/descaling_mode', type: 'MachineMode', label: 'Descaling Mode', confidence: 0.92 },
        { id: 'ns:Mode/heat_up_mode', type: 'MachineMode', label: 'Heat Up Mode', confidence: 0.94 },
        { id: 'ns:Mode/standby_mode', type: 'MachineMode', label: 'Standby Mode', confidence: 0.91 },

        // States
        { id: 'ns:State/ready', type: 'State', label: 'Ready State', confidence: 0.91 },
        { id: 'ns:State/heating', type: 'State', label: 'Heating State', confidence: 0.90 },
        { id: 'ns:State/brewing', type: 'State', label: 'Brewing State', confidence: 0.92 },
        { id: 'ns:State/error', type: 'State', label: 'Error State', confidence: 0.88 },

        // Failure Modes
        { id: 'ns:FM/no_water_flow', type: 'FailureMode', label: 'No Water Flow', confidence: 0.87 },
        { id: 'ns:FM/no_heating', type: 'FailureMode', label: 'No Heating', confidence: 0.86 },
        { id: 'ns:FM/pump_noise', type: 'FailureMode', label: 'Pump Makes Excessive Noise', confidence: 0.85 },
        { id: 'ns:FM/leaking_water', type: 'FailureMode', label: 'Water Leaking', confidence: 0.84 },

        // Repair Actions
        { id: 'ns:RA/check_pump', type: 'RepairAction', label: 'Check Pump', confidence: 0.85 },
        { id: 'ns:RA/replace_thermoblock', type: 'RepairAction', label: 'Replace Thermoblock', confidence: 0.84 },
        { id: 'ns:RA/descale_machine', type: 'RepairAction', label: 'Descale Machine', confidence: 0.86 },
        { id: 'ns:RA/check_connections', type: 'RepairAction', label: 'Check Electrical Connections', confidence: 0.83 },

        // Tools
        { id: 'ns:Tool/multimeter', type: 'Tool', label: 'Multimeter', confidence: 0.96 },
        { id: 'ns:Tool/torque_wrench', type: 'Tool', label: 'Torque Wrench', confidence: 0.95 },
        { id: 'ns:Tool/screwdriver_set', type: 'Tool', label: 'Screwdriver Set', confidence: 0.94 },

        // Test Cases
        { id: 'ns:Test/pressure_test', type: 'TestCase', label: 'Pressure Test', confidence: 0.89 },
        { id: 'ns:Test/heating_test', type: 'TestCase', label: 'Heating Test', confidence: 0.88 },
        { id: 'ns:Test/flow_rate_test', type: 'TestCase', label: 'Flow Rate Test', confidence: 0.87 }
      ];

      const edges = [
        // Product structure
        { id: 'e1', source: 'ns:Product/citiz', target: 'ns:Component/pump_cp4', type: 'hasPart', confidence: 0.92 },
        { id: 'e2', source: 'ns:Product/citiz', target: 'ns:Component/thermoblock_nespresso', type: 'hasPart', confidence: 0.91 },
        { id: 'e3', source: 'ns:Product/citiz', target: 'ns:Component/pcb_main', type: 'hasPart', confidence: 0.90 },
        { id: 'e4', source: 'ns:Product/citiz', target: 'ns:Component/water_tank', type: 'hasPart', confidence: 0.89 },
        { id: 'e5', source: 'ns:Product/milk_professional', target: 'ns:Component/pump_ulka', type: 'hasPart', confidence: 0.88 },

        // Component instances
        { id: 'e6', source: 'ns:Component/pump_cp4', target: 'ns:ComponentType/water_pump', type: 'instanceOf', confidence: 0.93 },
        { id: 'e7', source: 'ns:Component/thermoblock_nespresso', target: 'ns:ComponentType/thermoblock', type: 'instanceOf', confidence: 0.92 },
        { id: 'e8', source: 'ns:Component/pcb_main', target: 'ns:ComponentType/pcb', type: 'instanceOf', confidence: 0.91 },

        // Specifications
        { id: 'e9', source: 'ns:Component/pump_cp4', target: 'ns:ParameterSpec/pressure_19bar', type: 'hasSpec', confidence: 0.89 },
        { id: 'e10', source: 'ns:Product/citiz', target: 'ns:ParameterSpec/voltage_230v', type: 'hasSpec', confidence: 0.90 },
        { id: 'e11', source: 'ns:Product/citiz', target: 'ns:ParameterSpec/power_1260w', type: 'hasSpec', confidence: 0.88 },
        { id: 'e12', source: 'ns:Component/thermoblock_nespresso', target: 'ns:ParameterSpec/temp_92c', type: 'hasSpec', confidence: 0.87 },

        // Units
        { id: 'e13', source: 'ns:ParameterSpec/pressure_19bar', target: 'ns:Unit/bar', type: 'hasUnit', confidence: 0.95 },
        { id: 'e14', source: 'ns:ParameterSpec/voltage_230v', target: 'ns:Unit/v', type: 'hasUnit', confidence: 0.96 },
        { id: 'e15', source: 'ns:ParameterSpec/power_1260w', target: 'ns:Unit/w', type: 'hasUnit', confidence: 0.95 },
        { id: 'e16', source: 'ns:ParameterSpec/temp_92c', target: 'ns:Unit/celsius', type: 'hasUnit', confidence: 0.94 },

        // State transitions
        { id: 'e17', source: 'ns:Mode/heat_up_mode', target: 'ns:State/heating', type: 'appliesDuring', confidence: 0.91 },
        { id: 'e18', source: 'ns:State/heating', target: 'ns:State/ready', type: 'precedes', confidence: 0.90 },
        { id: 'e19', source: 'ns:State/ready', target: 'ns:State/brewing', type: 'precedes', confidence: 0.89 },
        { id: 'e20', source: 'ns:Mode/brewing_mode', target: 'ns:State/brewing', type: 'appliesDuring', confidence: 0.92 },

        // Troubleshooting
        { id: 'e21', source: 'ns:FM/no_water_flow', target: 'ns:RA/check_pump', type: 'mitigatedBy', confidence: 0.86 },
        { id: 'e22', source: 'ns:FM/no_heating', target: 'ns:RA/replace_thermoblock', type: 'mitigatedBy', confidence: 0.85 },
        { id: 'e23', source: 'ns:FM/pump_noise', target: 'ns:RA/descale_machine', type: 'mitigatedBy', confidence: 0.84 },

        // Tools
        { id: 'e24', source: 'ns:RA/check_pump', target: 'ns:Tool/multimeter', type: 'requiresTool', confidence: 0.88 },
        { id: 'e25', source: 'ns:RA/replace_thermoblock', target: 'ns:Tool/torque_wrench', type: 'requiresTool', confidence: 0.87 },
        { id: 'e26', source: 'ns:RA/replace_thermoblock', target: 'ns:Tool/screwdriver_set', type: 'requiresTool', confidence: 0.86 },

        // Tests
        { id: 'e27', source: 'ns:Test/pressure_test', target: 'ns:ParameterSpec/pressure_19bar', type: 'validatedBy', confidence: 0.89 },
        { id: 'e28', source: 'ns:Test/heating_test', target: 'ns:ParameterSpec/temp_92c', type: 'validatedBy', confidence: 0.88 }
      ];

      return {
        nodes,
        edges,
        meta: {
          document_code: 'KG_MOCK_EXPANDED',
          datasource_code: 'MOCK_DATA',
          nodeCount: nodes.length,
          edgeCount: edges.length
        }
      };
    }

    /**
     * Save configuration
     * @param {Object} config - Configuration object
     * @returns {Promise<Object>} Save result
     */
    async function saveConfig(config) {
      if (USE_MOCK) {
        await delay(500);
        // In mock mode, try to send to backend if available
        try {
          const response = await fetch('http://localhost:8000/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
          });

          if (response.ok) {
            return await response.json();
          }
        } catch (e) {
          console.warn('Backend config save not available');
        }

        return {
          success: true,
          message: 'Configuration saved to browser only'
        };
      }

      return request('/api/config', {
        method: 'POST',
        body: JSON.stringify(config)
      });
    }

    // Public API
    return {
      uploadFiles,
      startPipeline,
      getJobStatus,
      getKG,
      getStats,
      getConfig,
      downloadLogs,
      saveConfig
    };
  }

  return { create };
})();
