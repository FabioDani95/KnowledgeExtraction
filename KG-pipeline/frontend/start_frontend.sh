#!/bin/bash

# Knowledge Graph Frontend Startup Script
# This script starts both the upload server and the frontend web server

echo "=========================================="
echo "Knowledge Graph Frontend Startup"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "index.html" ]; then
    echo "❌ Error: Please run this script from the frontend directory"
    echo "   cd KG-pipeline/frontend && ./start_frontend.sh"
    exit 1
fi

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is required but not installed"
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"
echo ""

# Check if upload server dependencies are installed
echo "📦 Checking upload server dependencies..."
if python3 -c "import flask, flask_cors, yaml" 2>/dev/null; then
    echo "✅ All dependencies installed"
else
    echo "⚠️  Some dependencies missing. Installing..."
    pip3 install -r requirements.txt
fi
echo ""

# Start upload server in background
echo "🚀 Starting upload server on port 8000..."
python3 upload_server.py > upload_server.log 2>&1 &
UPLOAD_PID=$!
echo "   Upload server PID: $UPLOAD_PID"
echo "   Logs: frontend/upload_server.log"
echo ""

# Wait a moment for upload server to start
sleep 2

# Check if upload server started successfully
if ps -p $UPLOAD_PID > /dev/null; then
    echo "✅ Upload server started successfully"
else
    echo "❌ Upload server failed to start. Check upload_server.log for details"
    exit 1
fi
echo ""

# Start frontend web server
echo "🌐 Starting frontend web server on port 8080..."
echo "   Access the frontend at: http://localhost:8080"
echo ""
echo "=========================================="
echo "🎉 Frontend is ready!"
echo "=========================================="
echo ""
echo "📋 Quick Guide:"
echo "   1. Open http://localhost:8080 in your browser"
echo "   2. Go to Upload section to add PDF files"
echo "   3. Go to Configuration to set up pipeline parameters"
echo "   4. Go to Pipeline section to start extraction"
echo "   5. Go to Knowledge Graph to visualize results"
echo ""
echo "🛑 To stop: Press Ctrl+C"
echo "   Or run: kill $UPLOAD_PID"
echo ""
echo "=========================================="
echo ""

# Start frontend server (this will block until Ctrl+C)
python3 -m http.server 8080

# Cleanup on exit
echo ""
echo "🛑 Stopping upload server..."
kill $UPLOAD_PID 2>/dev/null
echo "✅ Shutdown complete"
