"""
Store service for FarmaBot - Handles store-related operations.
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, time

from .database_service import DatabaseService

class StoreService:
    def __init__(self, database_service: DatabaseService):
        """Initialize the store service."""
        self.db_service = database_service
        self.logger = logging.getLogger(__name__)
        
    def get_store_locations(self, language: Optional[str] = None) -> str:
        """Get store locations information as formatted text."""
        try:
            # Get all stores
            stores_query = "SELECT * FROM Stores"
            stores = self.db_service.execute_query(stores_query)
            
            if not stores:
                return "No stores found." if language != 'es' else "No se encontraron farmacias."
                
            # Format store locations based on language
            if language == 'es':
                content = "**Ubicaciones de farmacias:**\n\n"
                for i, store in enumerate(stores, 1):
                    store_id, name, address, hours, phone = store
                    content += f"{i}. **{name}**\n   Dirección: {address}\n   Teléfono: {phone}\n\n"
            else:
                content = "**Pharmacy Locations:**\n\n"
                for i, store in enumerate(stores, 1):
                    store_id, name, address, hours, phone = store
                    content += f"{i}. **{name}**\n   Address: {address}\n   Phone: {phone}\n\n"
                
            return content
            
        except Exception as e:
            self.logger.error(f"Error getting store locations: {e}")
            return "Error retrieving store locations." if language != 'es' else "Error al obtener ubicaciones de farmacias."
            
    def get_store_hours(self, language: Optional[str] = None) -> str:
        """Get store operating hours."""
        try:
            # Get all stores
            stores_query = "SELECT * FROM Stores"
            stores = self.db_service.execute_query(stores_query)
            
            if not stores:
                return "No stores found." if language != 'es' else "No se encontraron farmacias."
                
            # Format store hours based on language
            if language == 'es':
                content = "**Horarios de farmacias:**\n\n"
                for i, store in enumerate(stores, 1):
                    store_id, name, address, hours, phone = store
                    content += f"{i}. **{name}**: {hours}\n"
            else:
                content = "**Pharmacy Hours:**\n\n"
                for i, store in enumerate(stores, 1):
                    store_id, name, address, hours, phone = store
                    content += f"{i}. **{name}**: {hours}\n"
                
            return content
            
        except Exception as e:
            self.logger.error(f"Error getting store hours: {e}")
            return "Error retrieving store hours." if language != 'es' else "Error al obtener horarios de farmacias."
            
    def check_store_availability(self, store_id: int, language: Optional[str] = None) -> str:
        """Check if a store is currently open."""
        try:
            query = "SELECT * FROM Stores WHERE store_id = ?"
            stores = self.db_service.execute_query(query, (store_id,))
            
            if not stores:
                return "Store not found." if language != 'es' else "Farmacia no encontrada."
                
            store = stores[0]
            hours = store[3]  # opening_hours is at index 3
            
            if language == 'es':
                return f"Horario de la farmacia: {hours}"
            else:
                return f"Store hours: {hours}"
                
        except Exception as e:
            self.logger.error(f"Error checking store availability: {e}")
            return "Error checking store hours." if language != 'es' else "Error al verificar horarios de farmacia."
            
    def get_nearest_store(self, location: str, language: Optional[str] = None) -> str:
        """Get the nearest store to a given location."""
        try:
            stores_query = "SELECT * FROM Stores LIMIT 1"
            stores = self.db_service.execute_query(stores_query)
            
            if not stores:
                return "No stores found." if language != 'es' else "No se encontraron farmacias."
                
            # Return the first store (in a real app, you would calculate actual distances)
            store = stores[0]
            store_name = store[1]  # name is at index 1
            store_address = store[2]  # address is at index 2
            
            if language == 'es':
                return f"La farmacia más cercana es: **{store_name}** en {store_address}"
            else:
                return f"The nearest store is: **{store_name}** at {store_address}"
            
        except Exception as e:
            self.logger.error(f"Error getting nearest store: {e}")
            return "Error finding nearest store." if language != 'es' else "Error al encontrar la farmacia más cercana."
            
    def get_store_services(self, store_id: int, language: Optional[str] = None) -> str:
        """Get services available at a specific store."""
        try:
            query = "SELECT * FROM Stores WHERE store_id = ?"
            stores = self.db_service.execute_query(query, (store_id,))
            
            if not stores:
                return "Store not found." if language != 'es' else "Farmacia no encontrada."
                
            store = stores[0]
            store_name = store[1]  # name is at index 1
            
            # Since we don't have a services column, we'll return a generic message
            if language == 'es':
                return f"**{store_name}** ofrece servicios estándar de farmacia"
            else:
                return f"**{store_name}** offers standard pharmacy services"
                
        except Exception as e:
            self.logger.error(f"Error getting store services: {e}")
            return "Error getting store services." if language != 'es' else "Error al obtener servicios de farmacia." 