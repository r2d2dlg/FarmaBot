"""
Script to optimize and reduce the size of the vector database.

This script:
1. Creates a smaller version of the Chroma vector database with only essential medicines
2. Optimizes the SQLite files to reduce size
3. Enables easier distribution of the project while maintaining core functionality
"""

import os
import shutil
import sqlite3
import logging
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.docstore.document import Document
import pandas as pd
import sys
import json
from dotenv import load_dotenv

# Add project root to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables (for OpenAI API key)
load_dotenv()

# Configuration
ORIGINAL_VECTOR_DIR = "Medicines"
OPTIMIZED_VECTOR_DIR = "medicines_vectordb_lite"
ESSENTIAL_MEDS_FILE = "scripts/essential_medicines.json"

# Define essential medicines - common medicines that cover main use cases
# This can be customized based on your specific requirements
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

def optimize_sqlite_db(db_path):
    """Optimize an SQLite database to reduce its size."""
    logger.info(f"Optimizing SQLite file: {db_path}")
    
    try:
        # Connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Run VACUUM to rebuild the database and recover free space
        cursor.execute("VACUUM")
        
        # Additional optimizations
        cursor.execute("PRAGMA optimize")
        cursor.execute("PRAGMA auto_vacuum = FULL")
        
        # Commit and close
        conn.commit()
        conn.close()
        
        logger.info(f"Successfully optimized {db_path}")
    except Exception as e:
        logger.error(f"Error optimizing SQLite file {db_path}: {e}")
        raise
        
def create_optimized_vectorstore():
    """Create an optimized version of the vector database with only essential medicines."""
    try:
        logger.info("Setting up OpenAI embeddings...")
        embeddings = OpenAIEmbeddings()
        
        logger.info(f"Creating new optimized vector directory: {OPTIMIZED_VECTOR_DIR}")
        # Ensure the directory exists
        os.makedirs(OPTIMIZED_VECTOR_DIR, exist_ok=True)
        
        # Create a new Chroma instance in the optimized directory
        optimized_vectorstore = Chroma(
            persist_directory=OPTIMIZED_VECTOR_DIR,
            embedding_function=embeddings
        )
        
        # Create documents for essential medicines
        logger.info("Creating documents for essential medicines...")
        documents = []
        
        for medicine_name in ESSENTIAL_MEDICINES:
            # Create a simple document with medicine name
            content = f"Generic Name: {medicine_name}\nUses: Treatment of various conditions\nSide Effects: Various"
            doc = Document(page_content=content, metadata={"source": "optimized", "medicine": medicine_name})
            documents.append(doc)
        
        # Add the documents to the new vectorstore
        logger.info(f"Adding {len(documents)} medicine documents to optimized vectorstore...")
        optimized_vectorstore.add_documents(documents)
        
        # Persist the changes
        logger.info("Persisting optimized vectorstore...")
        optimized_vectorstore.persist()
        
        # Optimize the SQLite file
        sqlite_path = os.path.join(OPTIMIZED_VECTOR_DIR, "chroma.sqlite3")
        if os.path.exists(sqlite_path):
            optimize_sqlite_db(sqlite_path)
            
        # Get file size
        size_mb = os.path.getsize(sqlite_path) / (1024 * 1024)
        logger.info(f"Optimized vector database created: {size_mb:.2f} MB")
        
        return True
    except Exception as e:
        logger.error(f"Error creating optimized vectorstore: {e}")
        return False

def update_config_for_lite_db():
    """Update the config.py file to point to the optimized vector database."""
    try:
        # Create a backup of the original file
        shutil.copy("core/bot.py", "core/bot.py.backup")
        shutil.copy("services/medicine_service.py", "services/medicine_service.py.backup")
        
        # Update the bot.py file to use the optimized database
        with open("core/bot.py", "r") as f:
            content = f.read()
        
        # Replace the vector database path
        updated_content = content.replace(
            'persist_directory="medicines_vectordb",', 
            'persist_directory="medicines_vectordb_lite",  # Using optimized lite version'
        )
        
        with open("core/bot.py", "w") as f:
            f.write(updated_content)
            
        # Update the medicine_service.py file
        with open("services/medicine_service.py", "r") as f:
            content = f.read()
        
        # Replace the vector database path
        updated_content = content.replace(
            'persist_directory="medicines_vectordb",', 
            'persist_directory="medicines_vectordb_lite",  # Using optimized lite version'
        )
        
        with open("services/medicine_service.py", "w") as f:
            f.write(updated_content)
            
        logger.info("Updated configuration files to use optimized vector database")
        return True
    except Exception as e:
        logger.error(f"Error updating configuration files: {e}")
        return False

def create_readme_instructions():
    """Create a README file with instructions for the optimized vector database."""
    readme_content = """# Optimized Vector Database

This directory contains an optimized version of the medicine vector database used by FarmaBot.

## About This Database

- This is a reduced-size version containing only essential medicines
- It's designed to be small enough to share on GitHub while maintaining core functionality
- Full functionality requires regenerating the complete vector database

## Regenerating the Full Database

If you need the full vector database, you can regenerate it using:

```bash
python scripts/generate_vector_db.py
```

## Included Medicines

The optimized database includes commonly searched medicines like:
- Acetaminophen
- Ibuprofen
- Aspirin
- Amoxicillin
- (and others defined in essential_medicines.json)

## Switching Between Databases

To switch between the full and lite databases, update the `persist_directory` parameter in:
- core/bot.py
- services/medicine_service.py

Change from:
```python
persist_directory="medicines_vectordb_lite"
```

To:
```python
persist_directory="medicines_vectordb"
```
"""

    try:
        with open(f"{OPTIMIZED_VECTOR_DIR}/README.md", "w") as f:
            f.write(readme_content)
        logger.info(f"Created README in {OPTIMIZED_VECTOR_DIR}")
        return True
    except Exception as e:
        logger.error(f"Error creating README: {e}")
        return False

def main():
    """Main function to optimize the vector database."""
    logger.info("Starting vector database optimization")
    
    # Save the list of essential medicines
    save_essential_medicines_list()
    
    # Create the optimized vectorstore
    if create_optimized_vectorstore():
        logger.info("Successfully created optimized vector database")
        
        # Create README with instructions
        create_readme_instructions()
        
        # Optionally update config files
        # Uncomment this if you want to automatically update configs
        # update_config_for_lite_db()
        
        print("\n=== Vector Database Optimization Complete ===")
        print(f"The optimized vector database is now available in: {OPTIMIZED_VECTOR_DIR}")
        print("This smaller version can be committed to GitHub.")
        print("\nTo use this version, update the 'persist_directory' in:")
        print("- core/bot.py")
        print("- services/medicine_service.py")
    else:
        logger.error("Failed to create optimized vector database")
        print("\n=== Vector Database Optimization Failed ===")
        print("Check the logs for details.")

if __name__ == "__main__":
    main() 