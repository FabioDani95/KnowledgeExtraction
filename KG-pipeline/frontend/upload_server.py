#!/usr/bin/env python3
"""
Simple upload server for frontend file management.
Receives files from the web frontend and saves them to the correct pipeline folders.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path
import os
import shutil

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# Base directory (parent of frontend)
BASE_DIR = Path(__file__).resolve().parent.parent

# Mapping of folder keys to actual paths
FOLDER_MAPPING = {
    'product_technical': BASE_DIR / 'source' / 'product_technical',
    'operation_modes': BASE_DIR / 'source' / 'operation_modes',
    'troubleshooting': BASE_DIR / 'source' / 'troubleshooting',
    'testing': BASE_DIR / 'source' / 'testing',
    'repair_structure': BASE_DIR / 'source' / 'repair_structure'
}


@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle file upload from frontend"""
    try:
        if 'files' not in request.files:
            return jsonify({'success': False, 'error': 'No files provided'}), 400

        target_folder = request.form.get('targetFolder')
        if not target_folder or target_folder not in FOLDER_MAPPING:
            return jsonify({'success': False, 'error': 'Invalid target folder'}), 400

        target_path = FOLDER_MAPPING[target_folder]
        target_path.mkdir(parents=True, exist_ok=True)

        uploaded_files = []
        files = request.files.getlist('files')

        for file in files:
            if file and file.filename:
                # Secure filename
                filename = file.filename
                filepath = target_path / filename

                # Save file
                file.save(str(filepath))

                uploaded_files.append({
                    'name': filename,
                    'size': os.path.getsize(filepath),
                    'status': 'success',
                    'path': str(filepath.relative_to(BASE_DIR))
                })

        return jsonify({
            'success': True,
            'uploaded': uploaded_files,
            'targetFolder': target_folder,
            'count': len(uploaded_files)
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/pipeline/start', methods=['POST'])
def start_pipeline():
    """Start pipeline execution"""
    try:
        config = request.json or {}

        # Execute main_pipeline.py with config
        import subprocess
        cmd = ['python3', str(BASE_DIR / 'main_pipeline.py')]

        if config.get('verbose'):
            cmd.append('--verbose')

        # Run pipeline in background
        process = subprocess.Popen(
            cmd,
            cwd=str(BASE_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        job_id = f"job_{process.pid}"

        return jsonify({
            'success': True,
            'jobId': job_id,
            'status': 'started',
            'pid': process.pid
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/pipeline/status/<job_id>', methods=['GET'])
def get_pipeline_status(job_id):
    """Get pipeline execution status"""
    # For now, return mock status
    # TODO: Implement actual process monitoring
    return jsonify({
        'success': True,
        'jobId': job_id,
        'phase': 'Processing',
        'progress': 50,
        'status': 'running',
        'logs': ['Pipeline is running...']
    })


@app.route('/kg/current', methods=['GET'])
def get_current_kg():
    """Get current merged knowledge graph"""
    try:
        kg_path = BASE_DIR / 'output' / 'merged_kg' / 'kg_merged.json'

        if not kg_path.exists():
            return jsonify({'success': False, 'error': 'KG not found'}), 404

        import json
        with open(kg_path, 'r') as f:
            kg_data = json.load(f)

        return jsonify({'success': True, 'data': kg_data})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/config', methods=['GET', 'POST'])
def handle_config():
    """Get or update pipeline configuration"""
    config_path = BASE_DIR / 'config.yaml'

    if request.method == 'GET':
        try:
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return jsonify({'success': True, 'data': config})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500

    elif request.method == 'POST':
        try:
            import yaml
            new_config = request.json

            # Backup current config
            backup_path = config_path.with_suffix('.yaml.backup')
            shutil.copy2(config_path, backup_path)

            # Update config
            with open(config_path, 'r') as f:
                current_config = yaml.safe_load(f)

            # Merge changes
            if 'neural_extractor' in new_config:
                current_config['neural_extractor'].update(new_config['neural_extractor'])

            # Save updated config
            with open(config_path, 'w') as f:
                yaml.dump(current_config, f, default_flow_style=False)

            return jsonify({'success': True, 'message': 'Configuration updated'})

        except Exception as e:
            # Restore backup on error
            if backup_path.exists():
                shutil.copy2(backup_path, config_path)
            return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'service': 'KG Pipeline Upload Server'})


if __name__ == '__main__':
    print("=" * 60)
    print("Knowledge Graph Pipeline - Upload Server")
    print("=" * 60)
    print(f"Base directory: {BASE_DIR}")
    print(f"Available folders:")
    for key, path in FOLDER_MAPPING.items():
        print(f"  - {key}: {path}")
    print("\nStarting server on http://localhost:8000")
    print("=" * 60)

    app.run(host='0.0.0.0', port=8000, debug=True)
