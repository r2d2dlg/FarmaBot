#!/usr/bin/env python3
"""
Cleanup script for FarmaBot GitHub Repository
This script removes unnecessary files before GitHub upload.
"""

import os
import shutil
import glob

def cleanup_project():
    """Clean up the project directory by removing unnecessary files."""
    print("Starting FarmaBot cleanup for GitHub repository...")
    
    # Files needed for the main application
    essential_files = [
        'app.py', 
        'init_db.py',
        'farmabot.db',
        'requirements.txt',
        'README.md',
        'setup_database.sql',
        '.gitignore',
        'config.py'
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
    
    # Files/directories to explicitly remove
    files_to_remove = [
        '.ipynb_checkpoints',
        '__pycache__',
        '.gradio',
        '.venv',
        'venv',
        '.conda'
    ]
    
    # Patterns to remove
    patterns_to_remove = [
        '*.pdf',
        'orders_*.json',
        'test_*.py',
        '*.log',
        '*.pyc',
        '.DS_Store',
        'Thumbs.db',
        '*.ipynb'
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
    
    # 2. Remove explicitly listed files and directories
    for path in files_to_remove:
        if os.path.isfile(path) and path not in essential_files:
            try:
                os.remove(path)
                print(f"Removed file: {path}")
            except Exception as e:
                print(f"Error removing {path}: {e}")
        elif os.path.isdir(path) and path not in essential_dirs:
            try:
                shutil.rmtree(path)
                print(f"Removed directory: {path}")
            except Exception as e:
                print(f"Error removing {path}: {e}")
    
    # 3. Remove all __pycache__ directories
    for root, dirs, files in os.walk('.', topdown=False):
        for dir_name in dirs:
            if dir_name == '__pycache__':
                dir_path = os.path.join(root, dir_name)
                try:
                    shutil.rmtree(dir_path)
                    print(f"Removed directory: {dir_path}")
                except Exception as e:
                    print(f"Error removing {dir_path}: {e}")
    
    # 4. Create/update .gitignore
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

# Virtual Environment
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

# Jupyter Notebook
.ipynb_checkpoints
*.ipynb

# IDE files
.idea/
.vscode/
*.swp
*.swo

# macOS
.DS_Store

# Windows
Thumbs.db
ehthumbs.db
Desktop.ini
""")
        print("Created/updated .gitignore file")
    
    print("Cleanup complete! The project is ready for GitHub upload.")

if __name__ == "__main__":
    cleanup_project() 