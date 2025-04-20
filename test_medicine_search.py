"""
Test script for MedicineService functionality.
"""

import logging
from services.database_service import DatabaseService
from services.medicine_service import MedicineService

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def test_medicine_search():
    """Test the medicine search functionality."""
    # Initialize services
    db_service = DatabaseService()
    medicine_service = MedicineService(db_service)
    
    # Test cases
    test_cases = [
        # Exact matches
        ("Aspirin", "en"),
        ("Aspirina", "es"),
        
        # Similar names (misspellings)
        ("aspirn", "en"),
        ("aspirna", "es"),
        ("aspirina forte", "es"),
        
        # Brand names
        ("Bayer", "en"),
        ("Bayer", "es"),
        
        # Complex cases
        ("ibuprofen", "en"),
        ("ibuprofeno", "es"),
        ("ibup", "en"),
    ]
    
    print("\n=== Testing Medicine Search ===\n")
    
    for query, language in test_cases:
        print(f"\nQuery: '{query}' (Language: {language})")
        print("-" * 50)
        
        # Search for medicine
        result = medicine_service.search_medicines(query, language)
        print(result)
        
        # If result is a selection menu, test selection handling
        if "Which of these medicines" in result or "¿A cuál de estos medicamentos" in result:
            # Extract medicine options
            lines = result.split('\n')
            medicines = []
            for line in lines:
                if line.strip().startswith(('1.', '2.', '3.', '4.', '5.')):
                    medicines.append(line.strip())
            
            if medicines:
                # Test selecting the first option
                print("\nTesting selection of first option:")
                selection_result = medicine_service.handle_selection("1", medicines, language)
                print(selection_result)
        
        print("-" * 50)

if __name__ == "__main__":
    setup_logging()
    test_medicine_search() 