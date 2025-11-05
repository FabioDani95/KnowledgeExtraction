/**
 * Knowledge Graph Visualization Module
 *
 * Encapsulates graph rendering and interactions using Cytoscape.js.
 * Provides a clean API for graph manipulation independent of the visualization library.
 *
 * Usage:
 *   const viz = KGViz.create(containerElement, config);
 *   viz.render(kgData);
 *   viz.applyFilters({ nodeTypes: ['Product', 'Component'] });
 */

const KGViz = (function() {
  'use strict';

  /**
   * Create visualization instance
   * @param {HTMLElement} container - Container element for the graph
   * @param {Object} config - Configuration with node/edge types and styles
   * @returns {Object} Visualization API
   */
  function create(container, config) {
    let cy = null;
    let currentData = null;
    let currentFilters = {};
    let eventHandlers = {};

    /**
     * Initialize Cytoscape instance
     * @param {Object} options - Initialization options
     */
    function init(options = {}) {
      const defaultOptions = {
        container,
        style: buildCytoscapeStyles(config),
        layout: { name: 'cose', animate: false },
        wheelSensitivity: 0.2,
        minZoom: 0.1,
        maxZoom: 3
      };

      cy = cytoscape({ ...defaultOptions, ...options });

      // Attach event listeners
      cy.on('tap', 'node', handleNodeClick);
      cy.on('tap', 'edge', handleEdgeClick);
      cy.on('mouseover', 'node', handleNodeHover);
      cy.on('mouseout', 'node', handleNodeUnhover);
      cy.on('layoutstop', () => emit('layoutChanged', { layout: cy.layout().options.name }));

      return cy;
    }

    /**
     * Render graph data
     * @param {Object} data - Graph data with nodes and edges
     * @param {Object} layoutOptions - Layout configuration
     */
    function render(data, layoutOptions = {}) {
      if (!cy) {
        init();
      }

      currentData = data;

      // Transform data to Cytoscape format
      const elements = {
        nodes: data.nodes.map(node => ({
          data: {
            id: node.id,
            label: node.label || node.name,
            type: node.type,
            confidence: node.confidence,
            ...node
          },
          classes: [node.type.toLowerCase(), getConfidenceClass(node.confidence)]
        })),
        edges: data.edges.map(edge => ({
          data: {
            id: edge.id,
            source: edge.source,
            target: edge.target,
            type: edge.type,
            confidence: edge.confidence,
            ...edge
          },
          classes: [edge.type.toLowerCase(), getConfidenceClass(edge.confidence)]
        }))
      };

      cy.elements().remove();
      cy.add(elements);

      // Apply layout
      const defaultLayout = { name: 'cose', animate: true, animationDuration: 500 };
      const layout = cy.layout({ ...defaultLayout, ...layoutOptions });
      layout.run();

      emit('rendered', { nodeCount: data.nodes.length, edgeCount: data.edges.length });
    }

    /**
     * Apply filters to the graph
     * @param {Object} filters - Filter configuration
     */
    function applyFilters(filters) {
      if (!cy) return;

      currentFilters = { ...currentFilters, ...filters };

      cy.elements().removeClass('filtered hidden');

      let visibleNodes = cy.nodes();
      let visibleEdges = cy.edges();

      // Node type filter
      if (filters.nodeTypes && filters.nodeTypes.length > 0) {
        visibleNodes = visibleNodes.filter(node =>
          filters.nodeTypes.includes(node.data('type'))
        );
      }

      // Edge type filter
      if (filters.edgeTypes && filters.edgeTypes.length > 0) {
        visibleEdges = visibleEdges.filter(edge =>
          filters.edgeTypes.includes(edge.data('type'))
        );
      }

      // Confidence filter
      if (filters.minConfidence !== undefined) {
        visibleNodes = visibleNodes.filter(node =>
          node.data('confidence') >= filters.minConfidence
        );
        visibleEdges = visibleEdges.filter(edge =>
          edge.data('confidence') >= filters.minConfidence
        );
      }

      // Search filter
      if (filters.searchTerm) {
        const term = filters.searchTerm.toLowerCase();
        visibleNodes = visibleNodes.filter(node => {
          const label = (node.data('label') || '').toLowerCase();
          const name = (node.data('name') || '').toLowerCase();
          return label.includes(term) || name.includes(term);
        });
      }

      // Hide filtered elements
      cy.elements().addClass('hidden');
      visibleNodes.removeClass('hidden');
      visibleEdges.removeClass('hidden');

      // Only show edges between visible nodes
      visibleEdges = visibleEdges.filter(edge => {
        const source = edge.source();
        const target = edge.target();
        return !source.hasClass('hidden') && !target.hasClass('hidden');
      });

      visibleEdges.removeClass('hidden');

      emit('filtersApplied', {
        visibleNodes: visibleNodes.length,
        visibleEdges: visibleEdges.length
      });
    }

    /**
     * Focus on a specific node
     * @param {string} nodeId - Node identifier
     * @param {number} degree - Neighborhood degree (1 or 2)
     */
    function focusNode(nodeId, degree = 1) {
      if (!cy) return;

      const node = cy.getElementById(nodeId);
      if (!node.length) return;

      cy.elements().addClass('dimmed');

      // Highlight node and neighborhood
      const neighborhood = node.neighborhood();
      let highlighted = node.union(neighborhood);

      if (degree === 2) {
        neighborhood.forEach(n => {
          highlighted = highlighted.union(n.neighborhood());
        });
      }

      highlighted.removeClass('dimmed');

      // Fit to highlighted elements
      cy.fit(highlighted, 50);

      emit('nodeFocused', { nodeId, degree });
    }

    /**
     * Clear focus and show all elements
     */
    function clearFocus() {
      if (!cy) return;
      cy.elements().removeClass('dimmed');
      cy.fit();
    }

    /**
     * Search nodes by text
     * @param {string} query - Search query
     * @returns {Array} Matching nodes
     */
    function search(query) {
      if (!cy || !query) return [];

      const term = query.toLowerCase();
      const matches = cy.nodes().filter(node => {
        const label = (node.data('label') || '').toLowerCase();
        const name = (node.data('name') || '').toLowerCase();
        const id = (node.data('id') || '').toLowerCase();
        return label.includes(term) || name.includes(term) || id.includes(term);
      });

      return matches.map(node => ({
        id: node.data('id'),
        label: node.data('label'),
        type: node.data('type')
      }));
    }

    /**
     * Change graph layout
     * @param {string} layoutName - Layout algorithm name
     * @param {Object} options - Layout-specific options
     */
    function changeLayout(layoutName, options = {}) {
      if (!cy) return;

      const layouts = {
        cose: { name: 'cose', animate: true, nodeRepulsion: 400000 },
        circle: { name: 'circle', animate: true },
        grid: { name: 'grid', animate: true },
        breadthfirst: { name: 'breadthfirst', animate: true, directed: true },
        concentric: { name: 'concentric', animate: true, concentric: n => n.degree() }
      };

      const layoutConfig = { ...layouts[layoutName], ...options };
      const layout = cy.layout(layoutConfig);
      layout.run();

      emit('layoutChanged', { layout: layoutName });
    }

    /**
     * Export graph as PNG
     * @returns {string} Data URL of the image
     */
    function exportPNG() {
      if (!cy) return null;
      return cy.png({ full: true, scale: 2 });
    }

    /**
     * Export graph as JSON
     * @returns {Object} Graph data
     */
    function exportJSON() {
      return currentData;
    }

    /**
     * Get node details
     * @param {string} nodeId - Node identifier
     * @returns {Object} Node data
     */
    function getNodeDetails(nodeId) {
      if (!cy) return null;
      const node = cy.getElementById(nodeId);
      return node.length ? node.data() : null;
    }

    /**
     * Get edge details
     * @param {string} edgeId - Edge identifier
     * @returns {Object} Edge data
     */
    function getEdgeDetails(edgeId) {
      if (!cy) return null;
      const edge = cy.getElementById(edgeId);
      return edge.length ? edge.data() : null;
    }

    /**
     * Get graph statistics
     * @returns {Object} Statistics
     */
    function getStats() {
      if (!cy) return { nodes: 0, edges: 0 };

      const visibleNodes = cy.nodes(':visible');
      const visibleEdges = cy.edges(':visible');

      return {
        totalNodes: cy.nodes().length,
        totalEdges: cy.edges().length,
        visibleNodes: visibleNodes.length,
        visibleEdges: visibleEdges.length,
        nodeTypes: getDistribution(cy.nodes(), 'type'),
        edgeTypes: getDistribution(cy.edges(), 'type')
      };
    }

    // Event handling

    function on(event, handler) {
      if (!eventHandlers[event]) {
        eventHandlers[event] = [];
      }
      eventHandlers[event].push(handler);
    }

    function emit(event, data) {
      if (eventHandlers[event]) {
        eventHandlers[event].forEach(handler => handler(data));
      }
    }

    function handleNodeClick(evt) {
      const node = evt.target;
      emit('nodeSelected', node.data());
    }

    function handleEdgeClick(evt) {
      const edge = evt.target;
      emit('edgeSelected', edge.data());
    }

    function handleNodeHover(evt) {
      const node = evt.target;
      node.addClass('hover');
    }

    function handleNodeUnhover(evt) {
      const node = evt.target;
      node.removeClass('hover');
    }

    // Helper functions

    function buildCytoscapeStyles(config) {
      const styles = [
        {
          selector: 'node',
          style: {
            'label': 'data(label)',
            'text-valign': 'center',
            'text-halign': 'center',
            'font-size': '10px',
            'text-wrap': 'wrap',
            'text-max-width': '80px',
            'width': '30px',
            'height': '30px',
            'border-width': 2,
            'border-color': '#fff'
          }
        },
        {
          selector: 'edge',
          style: {
            'width': 2,
            'line-color': '#999',
            'target-arrow-color': '#999',
            'target-arrow-shape': 'triangle',
            'curve-style': 'bezier',
            'arrow-scale': 1
          }
        },
        {
          selector: '.hidden',
          style: { 'display': 'none' }
        },
        {
          selector: '.dimmed',
          style: { 'opacity': 0.2 }
        },
        {
          selector: '.hover',
          style: {
            'border-width': 4,
            'border-color': '#000'
          }
        },
        {
          selector: '.confidence-high',
          style: { 'opacity': 1.0 }
        },
        {
          selector: '.confidence-medium',
          style: { 'opacity': 0.8 }
        },
        {
          selector: '.confidence-low',
          style: { 'opacity': 0.6 }
        }
      ];

      // Add node type-specific styles
      Object.entries(config.nodeTypes || {}).forEach(([type, style]) => {
        styles.push({
          selector: `node.${type.toLowerCase()}`,
          style: {
            'background-color': style.color || '#3498db'
          }
        });
      });

      // Add edge type-specific styles
      Object.entries(config.edgeTypes || {}).forEach(([type, style]) => {
        const edgeStyle = {
          'line-color': style.color || '#999',
          'target-arrow-color': style.color || '#999',
          'width': style.width || 2
        };

        if (style.style === 'dashed') {
          edgeStyle['line-style'] = 'dashed';
        } else if (style.style === 'dotted') {
          edgeStyle['line-style'] = 'dotted';
        }

        styles.push({
          selector: `edge.${type.toLowerCase()}`,
          style: edgeStyle
        });
      });

      return styles;
    }

    function getConfidenceClass(confidence) {
      if (confidence >= 0.85) return 'confidence-high';
      if (confidence >= 0.70) return 'confidence-medium';
      return 'confidence-low';
    }

    function getDistribution(collection, attribute) {
      const dist = {};
      collection.forEach(el => {
        const value = el.data(attribute);
        dist[value] = (dist[value] || 0) + 1;
      });
      return dist;
    }

    // Public API
    return {
      init,
      render,
      applyFilters,
      focusNode,
      clearFocus,
      search,
      changeLayout,
      exportPNG,
      exportJSON,
      getNodeDetails,
      getEdgeDetails,
      getStats,
      on
    };
  }

  return { create };
})();
