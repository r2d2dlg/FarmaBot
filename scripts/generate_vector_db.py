"""
Script to generate the full vector database from medicine data in the SQL database.

This script:
1. Connects to the SQL database to retrieve all medicine information
2. Creates document entries for each medicine
3. Generates and persists embeddings in a Chroma vector database
"""

import os
import logging
import sys
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.docstore.document import Document
import pandas as pd

# Add project root to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project services
from services.database_service import DatabaseService

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration
VECTOR_DB_DIR = "medicines_vectordb"

def get_all_medicines(db_service: DatabaseService) -> List[Dict[str, Any]]:
    """Get all medicines from the database."""
    try:
        logger.info("Retrieving all medicines from database...")
        query = """
        SELECT 
            m.[MedicineID],
            m.[Generic Name],
            m.[Brand Name 1],
            m.[Brand Name 2],
            m.[Brand Name 3],
            m.[Brand Name 4],
            m.[Brand Name 5],
            m.[Brand Name 6],
            m.[Uses],
            m.[Side Effects (Common)],
            m.[Side Effects (Rare)],
            m.[Similar Drugs],
            m.[Prescription]
        FROM dbo.Medicines m
        """
        
        results = db_service.execute_query(query)
        logger.info(f"Retrieved {len(results)} medicines from database")
        return results
    except Exception as e:
        logger.error(f"Error getting medicines from database: {e}")
        raise

def create_medicine_documents(medicines: List[Dict[str, Any]]) -> List[Document]:
    """Convert medicine data into document format for the vector database."""
    documents = []
    
    for medicine in medicines:
        # Extract medicine details
        generic_name = medicine.get('Generic Name', '')
        
        # Collect brand names
        brand_names = []
        for i in range(1, 7):
            brand_name = medicine.get(f'Brand Name {i}')
            if brand_name:
                brand_names.append(brand_name)
        
        # Get other information
        uses = medicine.get('Uses', '')
        side_effects_common = medicine.get('Side Effects (Common)', '')
        side_effects_rare = medicine.get('Side Effects (Rare)', '')
        similar_drugs = medicine.get('Similar Drugs', '')
        prescription = "Yes" if medicine.get('Prescription') else "No"
        
        # Create document content
        content = f"Generic Name: {generic_name}\n"
        
        if brand_names:
            content += f"Brand Names: {', '.join(brand_names)}\n"
        
        content += f"Uses: {uses}\n"
        content += f"Side Effects (Common): {side_effects_common}\n"
        content += f"Side Effects (Rare): {side_effects_rare}\n"
        content += f"Similar Drugs: {similar_drugs}\n"
        content += f"Prescription Required: {prescription}\n"
        
        # Create document with metadata
        doc = Document(
            page_content=content,
            metadata={
                "medicine_id": medicine.get('MedicineID'),
                "generic_name": generic_name,
                "brand_names": brand_names,
                "prescription_required": medicine.get('Prescription', False)
            }
        )
        
        documents.append(doc)
    
    logger.info(f"Created {len(documents)} medicine documents")
    return documents

def generate_vector_database(documents: List[Document]) -> bool:
    """Generate the vector database from medicine documents."""
    try:
        logger.info("Setting up OpenAI embeddings...")
        embeddings = OpenAIEmbeddings()
        
        logger.info(f"Creating vector database directory: {VECTOR_DB_DIR}")
        # Ensure the directory exists
        os.makedirs(VECTOR_DB_DIR, exist_ok=True)
        
        # Create Chroma instance
        logger.info("Initializing Chroma vector database...")
        vectorstore = Chroma(
            persist_directory=VECTOR_DB_DIR,
            embedding_function=embeddings
        )
        
        # Add documents in batches to avoid memory issues
        batch_size = 50
        total_documents = len(documents)
        
        for i in range(0, total_documents, batch_size):
            batch_end = min(i + batch_size, total_documents)
            batch = documents[i:batch_end]
            
            logger.info(f"Adding batch {i//batch_size + 1}/{(total_documents + batch_size - 1)//batch_size} ({len(batch)} documents)...")
            vectorstore.add_documents(batch)
        
        # Persist the changes
        logger.info("Persisting vector database...")
        vectorstore.persist()
        
        # Get file size if the SQLite file exists
        sqlite_path = os.path.join(VECTOR_DB_DIR, "chroma.sqlite3")
        if os.path.exists(sqlite_path):
            size_mb = os.path.getsize(sqlite_path) / (1024 * 1024)
            logger.info(f"Vector database created: {size_mb:.2f} MB")
        
        return True
    except Exception as e:
        logger.error(f"Error generating vector database: {e}")
        return False

def main():
    """Main function to generate the vector database."""
    logger.info("Starting vector database generation")
    
    try:
        # Get database connection string from environment
        db_connection_string = os.getenv("DB_CONNECTION_STRING")
        if not db_connection_string:
            raise ValueError("DB_CONNECTION_STRING environment variable not set")
        
        # Initialize database service
        logger.info("Initializing database service...")
        db_service = DatabaseService(connection_string=db_connection_string)
        
        # Get all medicines from the database
        medicines = get_all_medicines(db_service)
        
        # Create document representations
        documents = create_medicine_documents(medicines)
        
        # Generate the vector database
        if generate_vector_database(documents):
            logger.info("Successfully created vector database")
            print("\n=== Vector Database Generation Complete ===")
            print(f"The vector database is now available in: {VECTOR_DB_DIR}")
            print("This directory is used by FarmaBot for semantic search of medicines.")
        else:
            logger.error("Failed to create vector database")
            print("\n=== Vector Database Generation Failed ===")
            print("Check the logs for details.")
            
    except Exception as e:
        logger.error(f"Error in vector database generation: {e}")
        print("\n=== Vector Database Generation Failed ===")
        print(f"Error: {e}")
        print("Check the logs for details.")

if __name__ == "__main__":
    main() 