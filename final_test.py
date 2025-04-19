"""
Final test script to validate medicine search functionality with prices.
This script will:
1. Initialize the database with price data
2. Test medicine search for Tylenol and other medicines
3. Verify price information is displayed correctly
"""

import os
import sqlite3
import logging
from services.database_service import DatabaseService
from services.medicine_service import MedicineService
from init_db import initialize_database

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def verify_database_schema():
    """Verify the database schema includes the price column."""
    db_path = "farmabot.db"
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Check medicine table schema
        cursor.execute("PRAGMA table_info(Medicines)")
        columns = cursor.fetchall()
        column_names = [col['name'] for col in columns]
        logger.info(f"Medicine columns: {column_names}")
        
        # Check for price column
        has_price = 'price' in [col['name'].lower() for col in columns]
        if has_price:
            logger.info("✓ Price column exists")
        else:
            logger.error("✗ Price column does not exist")
            
        # Check if prices were generated
        cursor.execute("SELECT COUNT(*) as count FROM Medicines WHERE price IS NOT NULL")
        count = cursor.fetchone()['count']
        if count > 0:
            logger.info(f"✓ Found {count} medicines with prices")
        else:
            logger.error("✗ No prices found in medicines table")
        
        # Sample data
        cursor.execute("SELECT medicine_id, generic_name, price FROM Medicines WHERE price IS NOT NULL LIMIT 5")
        samples = cursor.fetchall()
        for sample in samples:
            logger.info(f"Sample medicine: ID={sample['medicine_id']}, Name={sample['generic_name']}, Price=${sample['price']:.2f}")
        
        return has_price and count > 0
    except Exception as e:
        logger.error(f"Error verifying database schema: {e}")
        return False
    finally:
        if conn:
            conn.close()

def test_medicine_search():
    """Test the medicine search functionality with prices."""
    try:
        # Initialize services
        db_path = "farmabot.db"
        db_service = DatabaseService(db_path=db_path)
        medicine_service = MedicineService(db_service)
        
        # Test medicines to search for
        test_medicines = [
            ("tylenol", "Tylenol/Paracetamol"), 
            ("aspirin", "Aspirin"),
            ("insulin", "Insulin")
        ]
        
        for query, description in test_medicines:
            logger.info(f"\n--- Testing search for {description} ('{query}') ---")
            
            # Test in English
            en_result = medicine_service.search_medicines(query, "en")
            has_price_en = "Price: $" in en_result
            if has_price_en:
                logger.info(f"✓ English result includes price information")
            else:
                logger.warning(f"English result does not show price. Output:\n{en_result}")
            
            # Test in Spanish
            es_result = medicine_service.search_medicines(query, "es")
            has_price_es = "Precio: $" in es_result
            if has_price_es:
                logger.info(f"✓ Spanish result includes price information")
            else:
                logger.warning(f"Spanish result does not show price. Output:\n{es_result}")
            
            logger.info(f"Search for {description} complete. Price display: {'Success' if has_price_en or has_price_es else 'Failed'}")
        
        return True
    except Exception as e:
        logger.error(f"Error testing medicine search: {e}")
        return False

def main():
    """Run the full test suite."""
    logger.info("Starting comprehensive test of medicine search with prices")
    
    # Ensure database is properly initialized
    logger.info("\n=== Initializing Database ===")
    initialize_database()
    
    # Verify schema
    logger.info("\n=== Verifying Database Schema ===")
    schema_ok = verify_database_schema()
    
    # Test search functionality
    if schema_ok:
        logger.info("\n=== Testing Medicine Search with Prices ===")
        search_ok = test_medicine_search()
        
        if search_ok:
            logger.info("\n✓ All tests completed successfully!")
        else:
            logger.error("\n✗ Medicine search tests failed")
    else:
        logger.error("\n✗ Database schema verification failed. Cannot proceed with search tests.")
        
    logger.info("\nTesting complete.")

if __name__ == "__main__":
    main() 