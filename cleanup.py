#!/usr/bin/env python3
"""
Cleanup script for FarmaBot Hugging Face deployment.
This script removes unnecessary files and directories before deployment.
"""

import os
import shutil
import glob

def cleanup_project():
    """Clean up the project directory by removing unnecessary files."""
    print("Starting FarmaBot cleanup for Hugging Face deployment...")
    
    # Files needed for the main application
    essential_files = [
        'app.py', 
        'init_db.py',
        'farmabot.db',
        'requirements.txt',
        'requirements-huggingface.txt',
        'setup_database.sql'
    ]
    
    # Directories to keep
    essential_dirs = [
        'services',
        'core',
        'database'
    ]
    
    # Services files to keep
    essential_service_files = [
        'database_service.py',
        'medicine_service.py',
        'store_service.py',
        '__init__.py'
    ]
    
    # Files to explicitly remove
    files_to_remove = [
        'setup_cloud_storage.py',
        'cloud_storage_setup_README.md',
        'services/pdf_service.py',
        'services/storage_service.py'
    ]
    
    # Patterns to remove
    patterns_to_remove = [
        '*.pdf',
        'orders_*.json',
        'check_*.py',
        'test_*.py',
        'debug_*.py',
        'verify_*.py',
        'query_*.py',
        'import_*.py',
        'add_*.py',
        'generate_*.py',
        'excel_*.py',
        '*.log',
        'bugfix_summary.md'
    ]
    
    # 1. Remove all files matching patterns
    for pattern in patterns_to_remove:
        for file_path in glob.glob(pattern):
            if os.path.isfile(file_path) and file_path not in essential_files:
                try:
                    os.remove(file_path)
                    print(f"Removed file: {file_path}")
                except Exception as e:
                    print(f"Error removing {file_path}: {e}")
    
    # 2. Remove explicitly listed files
    for file_path in files_to_remove:
        if os.path.isfile(file_path):
            try:
                os.remove(file_path)
                print(f"Removed file: {file_path}")
            except Exception as e:
                print(f"Error removing {file_path}: {e}")
    
    # 3. Clean up the services directory
    if os.path.isdir('services'):
        for file_name in os.listdir('services'):
            file_path = os.path.join('services', file_name)
            if os.path.isfile(file_path) and file_name not in essential_service_files:
                try:
                    os.remove(file_path)
                    print(f"Removed file: {file_path}")
                except Exception as e:
                    print(f"Error removing {file_path}: {e}")
    
    # 4. Remove __pycache__ directories
    for root, dirs, files in os.walk('.', topdown=False):
        for dir_name in dirs:
            if dir_name == '__pycache__':
                dir_path = os.path.join(root, dir_name)
                try:
                    shutil.rmtree(dir_path)
                    print(f"Removed directory: {dir_path}")
                except Exception as e:
                    print(f"Error removing {dir_path}: {e}")
    
    # 5. Create a basic README file if it doesn't exist
    if not os.path.exists('README.md'):
        with open('README.md', 'w') as f:
            f.write("""# FarmaBot

A pharmacy chatbot deployed on Hugging Face Spaces.

## Features

- Multi-language support (English and Spanish)
- Medicine search functionality
- View pharmacy locations and hours
- Purchase workflow with text receipts
- Interactive chat interface

## How to Use

1. Select your language (English or Spanish)
2. Choose from the main menu options:
   - Search medicines
   - View locations
   - View hours
3. Follow the prompts to search for medicines and make purchases

## Technical Details

Built with Python, SQLite, and Gradio for the chat interface.
""")
        print("Created README.md file")
    
    # 6. Ensure the .gitignore file is appropriate
    with open('.gitignore', 'w') as f:
        f.write("""# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg

# Environments
.env
.venv
env/
venv/
ENV/

# Log files
*.log

# PDF files
*.pdf

# JSON order files
orders_*.json

# Other
.DS_Store
Thumbs.db
""")
        print("Created .gitignore file")
    
    print("Cleanup complete! The project is ready for Hugging Face deployment.")

if __name__ == "__main__":
    cleanup_project() 