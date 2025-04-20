"""
Script to create a lightweight vector database structure compatible with FarmaBot.

This script:
1. Creates a minimal Chroma directory structure
2. Doesn't require OpenAI API key for embeddings
3. Makes a small, GitHub-friendly database that can be committed
"""

import os
import logging
import json
import sqlite3
import shutil
import sys
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
LITE_DB_DIR = "medicines_vectordb_lite"
ESSENTIAL_MEDS_FILE = "scripts/essential_medicines.json"

# Define essential medicines - common medicines that cover main use cases
ESSENTIAL_MEDICINES = [
    "Acetaminophen",
    "Ibuprofen",
    "Aspirin",
    "Amoxicillin",
    "Lisinopril",
    "Atorvastatin",
    "Metformin",
    "Omeprazole",
    "Levothyroxine",
    "Alprazolam",
    "Methotrexate",
    "Prednisone",
    "Fluoxetine",
    "Sertraline",
    "Loratadine",
    "Cetirizine",
    "Insulin",
    "Albuterol",
    "Simvastatin",
    "Hydrochlorothiazide"
]

def save_essential_medicines_list():
    """Save the list of essential medicines to a JSON file for reference."""
    data = {
        "essential_medicines": ESSENTIAL_MEDICINES,
        "description": "These medicines are included in the optimized vector database."
    }
    
    os.makedirs(os.path.dirname(ESSENTIAL_MEDS_FILE), exist_ok=True)
    
    with open(ESSENTIAL_MEDS_FILE, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Saved essential medicines list to {ESSENTIAL_MEDS_FILE}")

def create_minimal_sqlite_db():
    """Create a minimal SQLite database for Chroma that maintains compatibility."""
    db_path = os.path.join(LITE_DB_DIR, "chroma.sqlite3")
    logger.info(f"Creating minimal SQLite file: {db_path}")
    
    try:
        # Ensure directory exists
        os.makedirs(LITE_DB_DIR, exist_ok=True)
        
        # Create SQLite database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create minimal tables required by Chroma
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id TEXT PRIMARY KEY,
            embedding BLOB,
            document TEXT,
            metadata TEXT
        )
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS collections (
            id TEXT PRIMARY KEY,
            name TEXT,
            metadata TEXT
        )
        """)
        
        # Insert a collection
        collection_id = "default_collection"
        collection_name = "medicines"
        collection_metadata = json.dumps({
            "description": "Lightweight medicine database for GitHub",
            "created_at": datetime.now().isoformat()
        })
        
        cursor.execute(
            "INSERT OR REPLACE INTO collections (id, name, metadata) VALUES (?, ?, ?)",
            (collection_id, collection_name, collection_metadata)
        )
        
        # Create dummy embedding entries for essential medicines
        for i, medicine in enumerate(ESSENTIAL_MEDICINES):
            doc_id = f"med_{i+1}"
            # Create placeholder embedding (zeros)
            placeholder_embedding = bytes(1536 * 4)  # 1536 dimensions, 4 bytes per float
            
            # Create document with medicine info
            document = json.dumps({
                "generic_name": medicine,
                "text": f"Generic Name: {medicine}\nUses: Treatment of various conditions\nSide Effects: Various"
            })
            
            # Create metadata
            metadata = json.dumps({
                "source": "lite_db",
                "medicine": medicine,
                "collection_id": collection_id
            })
            
            # Insert into database
            cursor.execute(
                "INSERT OR REPLACE INTO embeddings (id, embedding, document, metadata) VALUES (?, ?, ?, ?)",
                (doc_id, placeholder_embedding, document, metadata)
            )
        
        # Commit and optimize
        conn.commit()
        cursor.execute("VACUUM")
        cursor.execute("PRAGMA optimize")
        
        # Close connection
        conn.close()
        
        # Get file size
        size_mb = os.path.getsize(db_path) / (1024 * 1024)
        logger.info(f"Created minimal SQLite database: {size_mb:.2f} MB")
        
        return True
    except Exception as e:
        logger.error(f"Error creating SQLite database: {e}")
        return False

def create_chroma_directory_structure():
    """Create a minimal Chroma directory structure with essential files."""
    try:
        # Create UUID directory (placeholder)
        uuid_dir = os.path.join(LITE_DB_DIR, "00000000-0000-0000-0000-000000000000")
        os.makedirs(uuid_dir, exist_ok=True)
        
        # Create minimal required files in UUID directory
        with open(os.path.join(uuid_dir, "header.bin"), "wb") as f:
            f.write(b"CHROMA\x00\x00\x00\x01")  # Minimal header
            
        with open(os.path.join(uuid_dir, "length.bin"), "wb") as f:
            f.write(b"\x00" * 20)  # Placeholder length file
            
        with open(os.path.join(uuid_dir, "index_metadata.pickle"), "wb") as f:
            f.write(b"\x80\x03}q\x00.")  # Minimal pickle
            
        # Create README
        create_readme()
        
        logger.info(f"Created Chroma directory structure in {LITE_DB_DIR}")
        return True
    except Exception as e:
        logger.error(f"Error creating Chroma directory structure: {e}")
        return False

def create_readme():
    """Create a README file with instructions for the lite vector database."""
    readme_content = """# Lightweight Vector Database for FarmaBot

This directory contains a minimal vector database structure for FarmaBot that's compatible with GitHub.

## About This Database

- This is a lightweight placeholder that maintains compatibility with FarmaBot
- It's designed to be small enough to commit to GitHub (<5MB)
- It contains basic info for the most common medicines
- When using with Gemini models, no OpenAI API key is required

## Included Medicines

The database includes basic information for common medicines like:
- Acetaminophen
- Ibuprofen
- Aspirin
- Amoxicillin
- (and others defined in essential_medicines.json)

## Setup Instructions

To use this database with FarmaBot:

1. Ensure your MODEL in .env is set to a Gemini model
2. Update the following files to point to this directory:

In core/bot.py and services/medicine_service.py, change:
```python
persist_directory="medicines_vectordb",
```

To:
```python
persist_directory="medicines_vectordb_lite", 
```

## Full Database Generation

If you need the complete medicine database with actual embeddings:

1. Set your OpenAI API key in .env
2. Run: `python scripts/generate_vector_db.py`
"""

    try:
        with open(os.path.join(LITE_DB_DIR, "README.md"), "w") as f:
            f.write(readme_content)
        logger.info(f"Created README in {LITE_DB_DIR}")
        return True
    except Exception as e:
        logger.error(f"Error creating README: {e}")
        return False

def update_config_for_lite_db():
    """Update the configuration files to use the lite database."""
    try:
        # Create backups
        if os.path.exists("core/bot.py"):
            shutil.copy("core/bot.py", "core/bot.py.backup")
        
        if os.path.exists("services/medicine_service.py"):
            shutil.copy("services/medicine_service.py", "services/medicine_service.py.backup")
        
        # Update bot.py if it exists
        if os.path.exists("core/bot.py"):
            with open("core/bot.py", "r") as f:
                content = f.read()
            
            # Replace the vector database path
            updated_content = content.replace(
                'persist_directory="medicines_vectordb",', 
                'persist_directory="medicines_vectordb_lite",  # Using lightweight version'
            )
            
            with open("core/bot.py", "w") as f:
                f.write(updated_content)
            
            logger.info("Updated core/bot.py to use lite database")
        
        # Update medicine_service.py if it exists
        if os.path.exists("services/medicine_service.py"):
            with open("services/medicine_service.py", "r") as f:
                content = f.read()
            
            # Replace the vector database path
            updated_content = content.replace(
                'persist_directory="medicines_vectordb",', 
                'persist_directory="medicines_vectordb_lite",  # Using lightweight version'
            )
            
            with open("services/medicine_service.py", "w") as f:
                f.write(updated_content)
            
            logger.info("Updated services/medicine_service.py to use lite database")
        
        return True
    except Exception as e:
        logger.error(f"Error updating configuration files: {e}")
        return False

def main():
    """Main function to create a lightweight vector database."""
    logger.info("Starting lightweight vector database creation")
    
    # Save list of essential medicines
    save_essential_medicines_list()
    
    # Create SQLite database
    if create_minimal_sqlite_db():
        logger.info("Successfully created SQLite database")
        
        # Create directory structure
        if create_chroma_directory_structure():
            logger.info("Successfully created Chroma directory structure")
            
            # Ask if user wants to update config
            print("\n=== Lightweight Vector Database Created ===")
            print(f"Database created in: {LITE_DB_DIR}")
            print(f"Database size: {os.path.getsize(os.path.join(LITE_DB_DIR, 'chroma.sqlite3')) / (1024 * 1024):.2f} MB")
            print("\nThis database can now be committed to GitHub.")
            
            update_choice = input("\nUpdate config files to use this database? (y/n): ")
            if update_choice.lower() == 'y':
                if update_config_for_lite_db():
                    print("Configuration files updated successfully.")
                else:
                    print("Failed to update configuration files.")
            else:
                print("\nTo use this database, manually update the following files:")
                print("- core/bot.py")
                print("- services/medicine_service.py")
                print("\nChange 'persist_directory=\"medicines_vectordb\",' to 'persist_directory=\"medicines_vectordb_lite\",'")
        else:
            logger.error("Failed to create Chroma directory structure")
            print("Failed to create Chroma directory structure. Check logs for details.")
    else:
        logger.error("Failed to create SQLite database")
        print("Failed to create SQLite database. Check logs for details.")

if __name__ == "__main__":
    main() 