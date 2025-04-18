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
        
    def get_store_locations(self, location: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get store locations information."""
        try:
            return self.db_service.get_store_info(location=location)
        except Exception as e:
            self.logger.error(f"Error getting store locations: {e}")
            raise
            
    def get_store_hours(self, store_id: Optional[int] = None) -> str:
        """Get store operating hours."""
        try:
            stores = self.db_service.get_store_info(store_id=store_id)
            if not stores:
                return "Store not found."
                
            store = stores[0]
            open_time = store.get('OpenTime', '5:00')
            close_time = store.get('CloseTime', '22:00')
            
            return f"Store hours: {open_time} - {close_time} (7 days a week)"
            
        except Exception as e:
            self.logger.error(f"Error getting store hours: {e}")
            return "Error getting store hours."
            
    def check_store_availability(self, store_id: int) -> str:
        """Check if a store is currently open."""
        try:
            stores = self.db_service.get_store_info(store_id=store_id)
            if not stores:
                return "Store not found."
                
            store = stores[0]
            open_time = datetime.strptime(store.get('OpenTime', '5:00'), '%H:%M').time()
            close_time = datetime.strptime(store.get('CloseTime', '22:00'), '%H:%M').time()
            current_time = datetime.now().time()
            
            if open_time <= current_time <= close_time:
                return "The store is currently open."
            else:
                return "The store is currently closed."
                
        except Exception as e:
            self.logger.error(f"Error checking store availability: {e}")
            return "Error checking store availability."
            
    def get_nearest_store(self, location: str) -> str:
        """Get the nearest store to a given location."""
        try:
            stores = self.get_store_locations()
            if not stores:
                return "No stores found."
                
            # This is a simplified version. In a real implementation,
            # you would use geocoding to calculate actual distances
            return f"The nearest store is: {stores[0].get('Location', 'Unknown')}"
            
        except Exception as e:
            self.logger.error(f"Error getting nearest store: {e}")
            return "Error finding nearest store."
            
    def get_store_services(self, store_id: int) -> str:
        """Get services available at a specific store."""
        try:
            stores = self.db_service.get_store_info(store_id=store_id)
            if not stores:
                return "Store not found."
                
            store = stores[0]
            services = store.get('Services', 'Standard pharmacy services')
            return f"Available services: {services}"
            
        except Exception as e:
            self.logger.error(f"Error getting store services: {e}")
            return "Error getting store services."

    def get_store_info(self, location: Optional[str] = None) -> List[Dict[str, Any]]:
        """Alias for backward compatibility, delegates to get_store_locations."""
        return self.get_store_locations(location) 