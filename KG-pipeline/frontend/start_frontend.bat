@echo off
REM Knowledge Graph Frontend Startup Script for Windows
REM This script starts both the upload server and the frontend web server

echo ==========================================
echo Knowledge Graph Frontend Startup
echo ==========================================
echo.

REM Check if we're in the right directory
if not exist "index.html" (
    echo ERROR: Please run this script from the frontend directory
    echo   cd KG-pipeline\frontend
    echo   start_frontend.bat
    exit /b 1
)

REM Check Python
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python is required but not installed
    exit /b 1
)

echo Python found:
python --version
echo.

REM Check dependencies
echo Checking upload server dependencies...
python -c "import flask, flask_cors, yaml" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Installing dependencies...
    pip install -r requirements.txt
)
echo.

REM Start upload server
echo Starting upload server on port 8000...
start /B python upload_server.py > upload_server.log 2>&1
timeout /t 2 /nobreak >nul
echo Upload server started
echo.

REM Start frontend server
echo Starting frontend web server on port 8080...
echo Access the frontend at: http://localhost:8080
echo.
echo ==========================================
echo Frontend is ready!
echo ==========================================
echo.
echo Quick Guide:
echo   1. Open http://localhost:8080 in your browser
echo   2. Go to Upload section to add PDF files
echo   3. Go to Configuration to set up pipeline parameters
echo   4. Go to Pipeline section to start extraction
echo   5. Go to Knowledge Graph to visualize results
echo.
echo To stop: Press Ctrl+C
echo.
echo ==========================================
echo.

REM Start frontend server
python -m http.server 8080

echo.
echo Shutdown complete
