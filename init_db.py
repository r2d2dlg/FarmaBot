"""
Database initialization script for FarmaBot Hugging Face deployment
"""

import sqlite3
import logging
import os
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_database():
    """Initialize the SQLite database with tables and sample data."""
    try:
        # Check if database exists and is locked
        if os.path.exists('farmabot.db'):
            try:
                # Try to connect to check if it's locked
                test_conn = sqlite3.connect('farmabot.db')
                test_conn.close()
                # If we can connect, delete the file
                os.remove('farmabot.db')
                logger.info("Removed existing database")
            except sqlite3.OperationalError:
                logger.error("Database is locked by another process")
                return False
            except PermissionError:
                logger.error("Could not remove database - file is in use")
                return False
        
        # Read the SQL setup script
        with open('setup_database.sql', 'r') as f:
            setup_script = f.read()
            
        # Connect to database
        conn = sqlite3.connect('farmabot.db')
        
        # Execute the entire script at once
        conn.executescript(setup_script)
        
        # Add price column to Medicines table
        try:
            conn.execute("ALTER TABLE Medicines ADD COLUMN price REAL")
            logger.info("Added price column to Medicines table")
        except sqlite3.OperationalError as e:
            if "duplicate column" in str(e).lower():
                logger.info("Price column already exists")
            else:
                raise
        
        # Generate random prices for all medicines
        cursor = conn.cursor()
        cursor.execute("SELECT medicine_id FROM Medicines")
        medicine_ids = [row[0] for row in cursor.fetchall()]
        
        price_updates = []
        for medicine_id in medicine_ids:
            # Generate random price between $0.50 and $4.00
            price = round(random.uniform(0.50, 4.00), 2)
            price_updates.append((price, medicine_id))
        
        # Update prices in batches
        cursor.executemany(
            "UPDATE Medicines SET price = ? WHERE medicine_id = ?", 
            price_updates
        )
        
        conn.commit()
        logger.info("Database initialized successfully")
        
        # Verify the tables were created and prices were added
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        logger.info(f"Created tables: {[table[0] for table in tables]}")
        
        # Check prices
        cursor.execute("SELECT COUNT(*) as count, AVG(price) as avg_price FROM Medicines WHERE price IS NOT NULL")
        price_stats = cursor.fetchone()
        logger.info(f"Added prices for {price_stats[0]} medicines. Average price: ${price_stats[1]:.2f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        return False
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    initialize_database() 