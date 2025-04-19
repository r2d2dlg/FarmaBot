"""
Database service for FarmaBot - Handles database connections and operations.
"""

import logging
import sqlite3
from typing import Optional, List, Dict, Any, Union
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseService:
    def __init__(self, db_path: str):
        """Initialize the database service with a SQLite database path."""
        self.db_path = db_path
        self._local = threading.local()
        logger.info(f"Connected to SQLite database: {db_path}")
    
    @property
    def conn(self) -> sqlite3.Connection:
        """Get a thread-local database connection."""
        if not hasattr(self._local, 'conn'):
            self._local.conn = sqlite3.connect(self.db_path)
        return self._local.conn
    
    def execute_query(self, query: str, params: tuple = None) -> Optional[List[tuple]]:
        """Execute a SQL query and return the results."""
        try:
            cursor = self.conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            # For SELECT queries
            if query.strip().upper().startswith('SELECT'):
                return cursor.fetchall()
            
            # For INSERT/UPDATE/DELETE queries
            self.conn.commit()
            return None
            
        except Exception as e:
            logger.error(f"Database error during query execution: {e}", exc_info=True)
            if 'cursor' in locals():
                cursor.close()
            return None
    
    def get_medicine_info(self, medicine_id: int = None, medicine_name: str = None) -> Optional[Dict[str, Any]]:
        """Get medicine information by ID or name."""
        try:
            if medicine_id:
                query = """
                    SELECT m.*, SUM(i.quantity) as total_stock
                    FROM Medicines m
                    LEFT JOIN Inventory i ON m.medicine_id = i.medicine_id
                    WHERE m.medicine_id = ?
                    GROUP BY m.medicine_id
                """
                params = (medicine_id,)
            elif medicine_name:
                query = """
                    SELECT m.*, SUM(i.quantity) as total_stock
                    FROM Medicines m
                    LEFT JOIN Inventory i ON m.medicine_id = i.medicine_id
                    WHERE LOWER(m.generic_name) LIKE LOWER(?)
                    OR LOWER(m.brand_name1) LIKE LOWER(?)
                    GROUP BY m.medicine_id
                """
                search_term = f"%{medicine_name}%"
                params = (search_term, search_term)
            else:
                return None
            
            result = self.execute_query(query, params)
            if not result:
                return None
            
            # Convert to dictionary
            columns = ['medicine_id', 'generic_name', 'brand_name1', 'brand_name2', 
                      'brand_name3', 'brand_name4', 'brand_name5', 'brand_name6',
                      'description', 'side_effects', 'requires_prescription', 'price', 'total_stock']
            return dict(zip(columns, result[0]))
            
        except Exception as e:
            logger.error(f"Error getting medicine info: {e}", exc_info=True)
            return None
    
    def get_store_info(self, store_id: int = None, store_name: str = None) -> Optional[Dict[str, Any]]:
        """Get store information by ID or name."""
        try:
            if store_id:
                query = "SELECT * FROM Stores WHERE store_id = ?"
                params = (store_id,)
            elif store_name:
                query = "SELECT * FROM Stores WHERE LOWER(name) LIKE LOWER(?)"
                search_term = f"%{store_name}%"
                params = (search_term,)
            else:
                return None
            
            result = self.execute_query(query, params)
            if not result:
                return None
            
            # Convert to dictionary
            columns = ['store_id', 'name', 'address', 'opening_hours', 'phone_number']
            return dict(zip(columns, result[0]))
            
        except Exception as e:
            logger.error(f"Error getting store info: {e}", exc_info=True)
            return None
    
    def get_inventory(self, store_id: Optional[int] = None, medicine_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get inventory information, optionally filtered by store or medicine."""
        try:
            query = """
                SELECT s.name as store_name, m.generic_name, m.brand_name1, i.quantity
                FROM Inventory i
                JOIN Stores s ON i.store_id = s.store_id
                JOIN Medicines m ON i.medicine_id = m.medicine_id
                WHERE 1=1
            """
            params = []
            
            if store_id:
                query += " AND i.store_id = ?"
                params.append(store_id)
            if medicine_id:
                query += " AND i.medicine_id = ?"
                params.append(medicine_id)
            
            results = self.execute_query(query, tuple(params))
            if not results:
                return []
            
            # Convert to list of dictionaries
            inventory_list = []
            for row in results:
                inventory_list.append({
                    'store_name': row[0],
                    'generic_name': row[1],
                    'brand_name': row[2],
                    'quantity': row[3]
                })
            return inventory_list
            
        except Exception as e:
            logger.error(f"Error getting inventory: {e}", exc_info=True)
            return []
    
    def __del__(self):
        """Close the database connection when the service is destroyed."""
        if hasattr(self._local, 'conn'):
            self._local.conn.close() 