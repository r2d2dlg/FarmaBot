import sqlite3
import os
import logging
from typing import Optional, List, Dict, Any, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SQLiteManager:
    def __init__(self, db_path: str = 'farmabot.db'):
        """Initialize SQLite database manager."""
        logger.info(f"Initializing SQLiteManager with database: {db_path}")
        self.db_path = db_path
        self.setup_logging()
        self.initialize_database()

    def setup_logging(self):
        self.logger = logging.getLogger(__name__)

    def get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        logger.debug(f"Creating new connection to {self.db_path}")
        return sqlite3.connect(self.db_path)

    def initialize_database(self) -> None:
        """Initialize the SQLite database with tables and sample data."""
        try:
            logger.info("Starting database initialization...")
            
            # Read the SQL setup script
            setup_script_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'setup_database.sql')
            logger.info(f"Reading setup script from: {setup_script_path}")
            
            with open(setup_script_path, 'r', encoding='utf-8') as f:
                setup_script = f.read()
                logger.info("Successfully read setup script")

            # Create a new connection and execute the setup script
            conn = self.get_connection()
            cursor = conn.cursor()
            
            try:
                logger.info("Executing setup script...")
                cursor.executescript(setup_script)
                conn.commit()
                logger.info("Database initialized successfully")
                
                # Verify tables were created
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()
                logger.info(f"Created tables: {[table[0] for table in tables]}")
                
            except sqlite3.Error as e:
                logger.error(f"Error executing setup script: {e}")
                conn.rollback()
            finally:
                cursor.close()
                conn.close()

        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise

    def find_medicine(self, medicine_name: str) -> Optional[Dict[str, Any]]:
        """Find medicine by name (generic or brand)."""
        logger.info(f"Searching for medicine: {medicine_name}")
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            query = """
            SELECT 
                m.medicine_id,
                m.generic_name,
                m.brand_name1,
                m.brand_name2,
                m.description,
                m.side_effects,
                m.requires_prescription,
                s.store_id,
                s.name as store_name,
                s.address,
                s.opening_hours,
                s.phone_number,
                i.quantity
            FROM Medicines m
            LEFT JOIN Inventory i ON m.medicine_id = i.medicine_id
            LEFT JOIN Stores s ON i.store_id = s.store_id
            WHERE LOWER(m.generic_name) LIKE LOWER(?) 
               OR LOWER(m.brand_name1) LIKE LOWER(?)
               OR LOWER(m.brand_name2) LIKE LOWER(?)
               OR LOWER(m.brand_name3) LIKE LOWER(?)
               OR LOWER(m.brand_name4) LIKE LOWER(?)
               OR LOWER(m.brand_name5) LIKE LOWER(?)
               OR LOWER(m.brand_name6) LIKE LOWER(?)
            GROUP BY m.medicine_id, s.store_id
            """
            
            search_pattern = f"%{medicine_name}%"
            params = [search_pattern] * 7  # For all brand name columns
            
            cursor.execute(query, params)
            results = cursor.fetchall()
            
            if not results:
                logger.info(f"No medicine found for search term: {medicine_name}")
                return None
            
            # Process results into a structured format
            medicine_info = {
                'medicine_id': results[0][0],
                'generic_name': results[0][1],
                'brand_name1': results[0][2],
                'brand_name2': results[0][3],
                'description': results[0][4],
                'side_effects': results[0][5],
                'requires_prescription': bool(results[0][6]),
                'availability': []
            }
            
            for row in results:
                if row[7]:  # If store_id exists
                    store_info = {
                        'store_id': row[7],
                        'store_name': row[8],
                        'address': row[9],
                        'opening_hours': row[10],
                        'phone_number': row[11],
                        'quantity': row[12]
                    }
                    medicine_info['availability'].append(store_info)
            
            return medicine_info
            
        except sqlite3.Error as e:
            logger.error(f"Database error: {str(e)}")
            return None
        finally:
            cursor.close()
            conn.close()

    def get_store_info(self, store_id: int) -> Optional[Dict[str, Any]]:
        """Get information about a specific store."""
        logger.info(f"Retrieving store information for store ID: {store_id}")
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            query = """
            SELECT store_id, name, address, opening_hours, phone_number
            FROM Stores
            WHERE store_id = ?
            """
            
            cursor.execute(query, (store_id,))
            result = cursor.fetchone()
            
            if result:
                return {
                    'store_id': result[0],
                    'name': result[1],
                    'address': result[2],
                    'opening_hours': result[3],
                    'phone_number': result[4]
                }
            logger.info(f"No store found for store ID: {store_id}")
            return None
            
        except sqlite3.Error as e:
            logger.error(f"Database error: {str(e)}")
            return None
        finally:
            cursor.close()
            conn.close()

    def get_medicine_inventory(self, medicine_id: int) -> List[Dict[str, Any]]:
        """Get inventory information for a specific medicine."""
        logger.info(f"Retrieving inventory for medicine ID: {medicine_id}")
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT s.name, i.quantity
                FROM Inventory i
                JOIN Stores s ON i.store_id = s.store_id
                WHERE i.medicine_id = ?
            """, (medicine_id,))
            
            inventory = []
            for row in cursor.fetchall():
                inventory.append({
                    'store_name': row[0],
                    'quantity': row[1]
                })
            logger.info(f"Found inventory in {len(inventory)} stores")
            return inventory
            
        except sqlite3.Error as e:
            logger.error(f"Database error in get_medicine_inventory: {e}")
            return []
        finally:
            cursor.close()
            conn.close()

    def update_inventory(self, store_id: int, medicine_id: int, quantity: int) -> bool:
        """Update inventory quantity for a medicine at a store."""
        logger.info(f"Updating inventory for store {store_id}, medicine {medicine_id} to {quantity}")
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            query = """
            UPDATE Inventory
            SET quantity = ?
            WHERE store_id = ? AND medicine_id = ?
            """
            
            cursor.execute(query, (quantity, store_id, medicine_id))
            conn.commit()
            
            return True
            
        except sqlite3.Error as e:
            logger.error(f"Database error: {str(e)}")
            conn.rollback()
            return False
        finally:
            cursor.close()
            conn.close()

# Create a global instance of the SQLite manager
db_manager = SQLiteManager() 