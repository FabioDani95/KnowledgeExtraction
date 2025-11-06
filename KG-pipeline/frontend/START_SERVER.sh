#!/bin/bash
#
# Start the Knowledge Graph Pipeline Web Server
#
# This server provides:
# - File upload endpoints
# - KG data access
# - Pipeline execution
# - Real-time statistics
#

echo "=========================================="
echo "Knowledge Graph Pipeline - Web Server"
echo "=========================================="
echo ""
echo "Starting server on http://localhost:8000"
echo ""
echo "Open your browser to:"
echo "  http://localhost:8000"
echo ""
echo "Or open the frontend directly:"
echo "  file://$(pwd)/index.html"
echo ""
echo "Press Ctrl+C to stop the server"
echo "=========================================="
echo ""

python3 upload_server.py
