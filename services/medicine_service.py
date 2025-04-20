"""
Medicine service for FarmaBot - Handles medicine-related operations.
"""

import logging
from typing import Optional, List, Dict, Any
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.docstore.document import Document
from langchain_core.messages import SystemMessage, HumanMessage
import difflib
import re

from .database_service import DatabaseService

class MedicineService:
    def __init__(self, database_service: DatabaseService, model: str = "gpt-4-turbo"):
        """Initialize the medicine service."""
        self.db_service = database_service
        self.model = model
        # Store last search results for selection context
        self.last_search_results: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(__name__)
        # Adjust similarity thresholds
        self.exact_threshold = 0.9  # For very close matches
        self.high_threshold = 0.7   # For likely matches
        self.low_threshold = 0.5    # For possible matches
        
        # Initialize components
        self.setup_vectorstore()
        
    def setup_vectorstore(self):
        """Set up the vector store for medicine information."""
        try:
            self.embeddings = OpenAIEmbeddings()
            self.vectorstore = Chroma(
                persist_directory="medicines_vectordb_lite",  # Using lightweight version
                embedding_function=self.embeddings
            )
            self.retriever = self.vectorstore.as_retriever()
            self.logger.info("Medicine vector store initialized successfully")
        except Exception as e:
            self.logger.error(f"Error setting up medicine vector store: {e}")
            raise
            
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        return re.sub(r'[^a-zA-Z0-9]', '', text.lower())
        
    def _calculate_similarity(self, query: str, target: str) -> float:
        """Calculate similarity between two strings using multiple methods."""
        if not query or not target:
            return 0.0
            
        # Normalize strings
        query = self._normalize_text(query)
        target = self._normalize_text(target)
        
        # Calculate basic similarity
        basic_similarity = difflib.SequenceMatcher(None, query, target).ratio()
        
        # Calculate partial ratio (for when one string is a substring of the other)
        partial_ratio = difflib.SequenceMatcher(None, query, target).quick_ratio()
        
        # Calculate token set ratio (for when words are in different order)
        query_words = set(query.split())
        target_words = set(target.split())
        token_set_ratio = difflib.SequenceMatcher(None, 
            ' '.join(sorted(query_words)), 
            ' '.join(sorted(target_words))
        ).ratio()
        
        # Return the highest similarity score
        return max(basic_similarity, partial_ratio, token_set_ratio)
        
    def _find_similar_medicines(self, query: str) -> List[Dict[str, Any]]:
        """Find medicines with similar names to the query."""
        try:
            # Fetch generic and all brand columns
            all_medicine_rows = self.db_service.execute_query("""
                SELECT
                    m.MedicineID AS MedicineID,
                    m.[Generic Name] AS GenericName,
                    m.[Brand Name 1] AS BrandName1,
                    m.[Brand Name 2] AS BrandName2,
                    m.[Brand Name 3] AS BrandName3,
                    m.[Brand Name 4] AS BrandName4,
                    m.[Brand Name 5] AS BrandName5,
                    m.[Brand Name 6] AS BrandName6
                FROM dbo.Medicines m
            """)
            
            all_medicines = []
            # Consolidate brand names into list
            for row in all_medicine_rows:
                brands = []
                for i in range(1, 7):
                    bn = row.get(f'BrandName{i}')
                    if bn:
                        brands.append(bn)
                row['BrandNames'] = brands
                all_medicines.append(row)
            
            # Find similar names
            similar_medicines = []
            for medicine in all_medicines:
                generic_name = medicine.get('GenericName', '')
                # Multiple brands
                brand_list = medicine.get('BrandNames', [])
                
                # Calculate similarity scores
                generic_similarity = self._calculate_similarity(query, generic_name)
                # Check similarity across all brands
                brand_similarity = max(
                    (self._calculate_similarity(query, bn) for bn in brand_list),
                    default=0.0
                )
                
                # Get the highest similarity score
                similarity = max(generic_similarity, brand_similarity)
                
                # If similarity is above the low threshold, add the medicine
                if similarity >= self.low_threshold:
                    medicine['Similarity'] = similarity
                    medicine['MatchType'] = 'exact' if similarity >= self.exact_threshold else \
                                         'high' if similarity >= self.high_threshold else 'low'
                    similar_medicines.append(medicine)
            
            # Sort by similarity score and match type
            similar_medicines.sort(key=lambda x: (x['MatchType'] != 'exact', 
                                                x['MatchType'] != 'high', 
                                                -x['Similarity']))
            return similar_medicines[:5]  # Return top 5 matches
            
        except Exception as e:
            self.logger.error(f"Error finding similar medicines: {e}")
            return []
            
    def _format_selection_menu(self, medicines: List[Dict[str, Any]], language: str) -> str:
        """Format a selection menu for similar medicines."""
        if language == "es":
            menu = "¿A cuál de estos medicamentos te refieres?\n\n"
            for i, med in enumerate(medicines, 1):
                menu += f"{i}. {med.get('GenericName')}"
                if med.get('BrandNames'):
                    menu += f" ({', '.join(med.get('BrandNames'))})"
                # Add confidence level
                match_type = med.get('MatchType', '')
                if match_type == 'exact':
                    menu += " [Coincidencia exacta]"
                elif match_type == 'high':
                    menu += " [Coincidencia probable]"
                else:
                    menu += " [Posible coincidencia]"
                menu += "\n"
            menu += "\nPor favor, responde con el número correspondiente."
        else:
            menu = "Which of these medicines are you referring to?\n\n"
            for i, med in enumerate(medicines, 1):
                menu += f"{i}. {med.get('GenericName')}"
                if med.get('BrandNames'):
                    menu += f" ({', '.join(med.get('BrandNames'))})"
                # Add confidence level
                match_type = med.get('MatchType', '')
                if match_type == 'exact':
                    menu += " [Exact match]"
                elif match_type == 'high':
                    menu += " [Probable match]"
                else:
                    menu += " [Possible match]"
                menu += "\n"
            menu += "\nPlease respond with the corresponding number."
            
        return menu
        
    def search_medicines(self, query: str, language: str = "es") -> str:
        """Search for medicines using both brand and generic names with fuzzy matching."""
        try:
            self.logger.info(f"Starting medicine search for query: '{query}'")
            # First, try to find exact matches
            self.logger.info("Attempting exact match search...")
            medicines = self.get_medicine_details(medicine_name=query)
            
            if medicines:
                self.logger.info(f"Exact match found: {medicines[0].get('GenericName')}")
                return self._format_medicine_response(medicines[0], language)
            
            self.logger.info("No exact match found. Attempting fuzzy match search...")
            # If no exact match, look for similar names
            similar_medicines = self._find_similar_medicines(query)
            
            if similar_medicines:
                self.logger.info(f"Fuzzy matches found ({len(similar_medicines)}). Returning selection menu.")
                # Save results for context
                self.last_search_results = similar_medicines
                # Ask the user to choose one
                return self._format_selection_menu(similar_medicines, language)
            
            self.logger.info("No fuzzy matches found. Attempting vector search...")
            # If no similar matches, try vector similarity search
            docs = self.retriever.invoke(query)
            if not docs:
                self.logger.warning(f"No matches found via vector search for query: '{query}'")
                return self._get_not_found_message(language)
                
            self.logger.info("Vector search returned results. Processing best match...")
            # Get the most relevant medicine
            context = docs[0].page_content
            medicine_name = context.split('\n')[0].split(': ')[1]
            
            self.logger.info(f"Best vector match suggests: '{medicine_name}'. Looking up details...")
            # Try to find the medicine in the database
            medicines = self.get_medicine_details(medicine_name=medicine_name)
            if medicines:
                self.logger.info(f"Details found for vector match: {medicines[0].get('GenericName')}")
                return self._format_medicine_response(medicines[0], language)
            
            self.logger.warning(f"Could not find DB details for vector match suggestion: '{medicine_name}'")
            return self._get_not_found_message(language)
            
        except Exception as e:
            self.logger.error(f"Error during medicine search for query '{query}': {e}", exc_info=True)
            return self._get_error_message(language)
            
    def handle_selection(self, selection: str, similar_medicines: List[Dict[str, Any]], language: str) -> str:
        """Handle user selection from similar medicines menu."""
        try:
            # Try to convert selection to integer
            try:
                choice = int(selection)
                if 1 <= choice <= len(similar_medicines):
                    medicine = similar_medicines[choice - 1]
                    return self._format_medicine_response(medicine, language)
            except ValueError:
                pass
                
            # If selection is invalid, return error message
            if language == "es":
                return "Selección inválida. Por favor, elige un número de la lista."
            else:
                return "Invalid selection. Please choose a number from the list."
                
        except Exception as e:
            self.logger.error(f"Error handling selection: {e}")
            return self._get_error_message(language)
            
    def _check_store_stock(self, generic_name: str, brand_names: List[str], inventory_table: str) -> Dict[str, Any]:
        """Check if a medicine (generic and brand names) is available in a specific store using its inventory table."""
        try:
            # Query the store's specific inventory table for generic and brand names
            # Prepare brand names list for SQL IN clause
            brands_list = [bn for bn in brand_names if bn]
            brands_clause = ''
            if brands_list:
                in_list = ', '.join(f"'{bn}'" for bn in brands_list)
                brands_clause = f" OR [Brand Name] IN ({in_list})"
            query = f"""
                SELECT
                    [Generic Name] AS GenericName,
                    [Brand Name] AS BrandName,
                    Inventory
                FROM dbo.[{inventory_table}]
                WHERE [Generic Name] = '{generic_name}'{brands_clause}
            """
            rows = self.db_service.execute_query(query)
            # Determine availability
            generic_available = any(r.get('GenericName') == generic_name and r.get('Inventory', 0) > 0 for r in rows)
            brand_availability = {bn: any(r.get('BrandName') == bn and r.get('Inventory', 0) > 0 for r in rows) for bn in brands_list}
            return {'generic_available': generic_available, 'brand_availability': brand_availability}
        except Exception as e:
            self.logger.error(f"Error checking store stock for store '{inventory_table}': {e}")
            return {'generic_available': False, 'brand_availability': {}}
            
    def _format_medicine_response(self, medicine: Dict[str, Any], language: str) -> str:
        """Format the medicine information response."""
        try:
            # Get store availability
            stores = self.db_service.get_store_info()
            availability_info = []
            # Prepare generic and brand names
            generic_name = medicine.get('GenericName')
            brand_names = medicine.get('BrandNames', [])
            for store in stores:
                inventory_table = store.get('InventoryTableName')
                store_location = store.get('Location')
                stock_info = self._check_store_stock(generic_name, brand_names, inventory_table)
                
                # Format availability information
                if language == "es":
                    store_availability = f"\n{store_location}:\n"
                    if stock_info['generic_available']:
                        store_availability += "  - Genérico: Disponible\n"
                    else:
                        store_availability += "  - Genérico: No disponible\n"
                        
                    if stock_info['brand_availability']:
                        store_availability += "  - Marcas disponibles:\n"
                        for brand, available in stock_info['brand_availability'].items():
                            status = "Disponible" if available else "No disponible"
                            store_availability += f"    * {brand}: {status}\n"
                    else:
                        store_availability += "  - No hay marcas disponibles\n"
                else:
                    store_availability = f"\n{store_location}:\n"
                    if stock_info['generic_available']:
                        store_availability += "  - Generic: Available\n"
                    else:
                        store_availability += "  - Generic: Not available\n"
                        
                    if stock_info['brand_availability']:
                        store_availability += "  - Available brands:\n"
                        for brand, available in stock_info['brand_availability'].items():
                            status = "Available" if available else "Not available"
                            store_availability += f"    * {brand}: {status}\n"
                    else:
                        store_availability += "  - No brands available\n"
                
                availability_info.append(store_availability)
            
            # Format response based on language
            if language == "es":
                # Spanish response without leading spaces or code block formatting
                lines = [
                    "Información del medicamento:",
                    "",
                    f"Nombre genérico: {medicine.get('GenericName', 'No disponible')}",
                    f"Nombre comercial: {', '.join(medicine.get('BrandNames', ['No disponible']))}",
                    f"Requiere receta: {'Sí' if medicine.get('RequiresPrescription') else 'No'}",
                    "",
                    "Disponibilidad en tiendas:",
                ]
                # Add store availability lines
                lines.extend(line.strip() for line in availability_info)
                response = "\n".join(lines)
            else:
                response = f"""
                Medicine Information:
                
                Generic Name: {medicine.get('GenericName', 'Not available')}
                Brand Name: {', '.join(medicine.get('BrandNames', ['Not available']))}
                Requires Prescription: {'Yes' if medicine.get('RequiresPrescription') else 'No'}
                
                Store Availability:
                {''.join(availability_info)}
                """
            
            return response.strip()
            
        except Exception as e:
            self.logger.error(f"Error formatting medicine response: {e}")
            return self._get_error_message(language)
            
    def _get_not_found_message(self, language: str) -> str:
        """Get the 'medicine not found' message in the appropriate language."""
        return (
            "Lo siento, no pude encontrar información sobre ese medicamento. "
            "Por favor, verifica el nombre y vuelve a intentarlo."
            if language == "es" else
            "I'm sorry, I couldn't find information about that medicine. "
            "Please check the name and try again."
        )
        
    def _get_error_message(self, language: str) -> str:
        """Get the error message in the appropriate language."""
        return (
            "Lo siento, tuve un problema al buscar la información del medicamento."
            if language == "es" else
            "I'm sorry, I had trouble finding the medicine information."
        )
            
    def get_medicine_details(self, medicine_id: Optional[int] = None, 
                           medicine_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get detailed information about a specific medicine."""
        try:
            query = """
            SELECT
                m.MedicineID AS MedicineID,
                m.[Generic Name] AS GenericName,
                m.[Brand Name 1] AS BrandName1,
                m.[Brand Name 2] AS BrandName2,
                m.[Brand Name 3] AS BrandName3,
                m.[Brand Name 4] AS BrandName4,
                m.[Brand Name 5] AS BrandName5,
                m.[Brand Name 6] AS BrandName6,
                m.Prescription AS RequiresPrescription,
                m.[Side Effects (Common)] AS CommonSideEffects,
                m.[Side Effects (Rare)] AS RareSideEffects
            FROM dbo.Medicines m
            """
            
            if medicine_id:
                query += f" WHERE m.MedicineID = {medicine_id}"
            elif medicine_name:
                query += f" WHERE m.[Generic Name] LIKE '%{medicine_name}%' OR m.[Brand Name 1] LIKE '%{medicine_name}%'"
                
            rows = self.db_service.execute_query(query)
            # Combine multiple brand columns into a list
            for row in rows:
                brand_list = []
                for i in range(1, 7):
                    bn = row.get(f'BrandName{i}')
                    if bn and isinstance(bn, str) and bn.strip():
                        brand_list.append(bn.strip())
                row['BrandNames'] = brand_list
            return rows
        except Exception as e:
            self.logger.error(f"Error getting medicine details: {e}")
            raise
            
    def check_prescription_status(self, medicine_name: str) -> str:
        """Check if a medicine requires a prescription."""
        try:
            medicines = self.get_medicine_details(medicine_name=medicine_name)
            if not medicines:
                return "Medicine not found."
                
            medicine = medicines[0]
            return f"Prescription required: {medicine.get('RequiresPrescription', 'Unknown')}"
            
        except Exception as e:
            self.logger.error(f"Error checking prescription status: {e}")
            return "Error checking prescription status."
            
    def get_side_effects(self, medicine_name: str) -> str:
        """Get side effects information for a medicine."""
        try:
            medicines = self.get_medicine_details(medicine_name=medicine_name)
            if not medicines:
                return "Medicine not found."
                
            medicine = medicines[0]
            common_effects = medicine.get('CommonSideEffects', 'No common side effects listed.')
            rare_effects = medicine.get('RareSideEffects', 'No rare side effects listed.')
            
            return f"Common side effects: {common_effects}\nRare side effects: {rare_effects}"
            
        except Exception as e:
            self.logger.error(f"Error getting side effects: {e}")
            return "Error getting side effects information." 