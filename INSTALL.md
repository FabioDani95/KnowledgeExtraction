# Installation Guide

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installing Dependencies

### Required Dependencies

Install the core dependencies needed to run the application:

```bash
pip install -r requirements.txt
```

This will install:
- **PyYAML**: For reading YAML configuration files
- **PyPDF2**: For PDF document processing
- **jsonschema**: For schema validation

### Optional Dependencies

For advanced features (language detection in content filtering):

```bash
pip install -r requirements-optional.txt
```

**Note**: `langdetect` may require build tools on some systems. If installation fails, the application will still work but language detection filtering will be disabled.

## Quick Start

1. Activate your virtual environment:
   ```bash
   source .venv/bin/activate  # On macOS/Linux
   # or
   .venv\Scripts\activate  # On Windows
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python KG-pipeline/main.py
   ```

## Troubleshooting

### "ModuleNotFoundError: No module named 'yaml'"

Make sure you've installed the requirements:
```bash
pip install -r requirements.txt
```

### Virtual Environment Issues

If you encounter issues with your virtual environment:

1. Deactivate the current environment:
   ```bash
   deactivate
   ```

2. Remove the old environment:
   ```bash
   rm -rf .venv
   ```

3. Create a new virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On macOS/Linux
   pip install -r requirements.txt
   ```
