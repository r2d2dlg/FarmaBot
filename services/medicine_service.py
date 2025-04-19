"""
Medicine service for FarmaBot - Handles medicine-related operations.
"""

import logging
from typing import Optional, List, Dict, Any, Union
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.docstore.document import Document
from langchain_core.messages import SystemMessage, HumanMessage
import difflib
import re
import random

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
        
        # Context tracking
        self.current_medicine_context = None
        self.available_stores = []
        
        # Initialize components
        self.setup_vectorstore()
        
    def setup_vectorstore(self):
        """Set up the vector store for medicine information."""
        try:
            self.embeddings = OpenAIEmbeddings()
            self.vectorstore = Chroma(
                persist_directory="medicines_vectordb",
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
                    m.medicine_id,
                    m.generic_name,
                    m.brand_name1,
                    m.brand_name2,
                    m.brand_name3,
                    m.brand_name4,
                    m.brand_name5,
                    m.brand_name6
                FROM Medicines m
            """)
            
            if not all_medicine_rows:
                self.logger.error("Database error fetching all medicines.")
                return []
            
            all_medicines = []
            # Consolidate brand names into list
            for row in all_medicine_rows:
                medicine = {
                    'medicine_id': row[0],
                    'generic_name': row[1],
                    'brand_name1': row[2],
                    'brand_name2': row[3],
                    'brand_name3': row[4],
                    'brand_name4': row[5],
                    'brand_name5': row[6],
                    'brand_name6': row[7]
                }
                all_medicines.append(medicine)
            
            # Find similar names
            similar_medicines = []
            for medicine in all_medicines:
                generic_name = medicine.get('generic_name', '')
                brand_names = [
                    medicine.get('brand_name1'),
                    medicine.get('brand_name2'),
                    medicine.get('brand_name3'),
                    medicine.get('brand_name4'),
                    medicine.get('brand_name5'),
                    medicine.get('brand_name6')
                ]
                brand_names = [bn for bn in brand_names if bn]  # Remove None values
                
                # Calculate similarity scores
                generic_similarity = self._calculate_similarity(query, generic_name)
                brand_similarity = max(
                    (self._calculate_similarity(query, bn) for bn in brand_names),
                    default=0.0
                )
                
                # Get the highest similarity score
                similarity = max(generic_similarity, brand_similarity)
                
                # If similarity is above the low threshold, add the medicine
                if similarity >= self.low_threshold:
                    medicine['similarity'] = similarity
                    medicine['match_type'] = 'exact' if similarity >= self.exact_threshold else \
                                         'high' if similarity >= self.high_threshold else 'low'
                    similar_medicines.append(medicine)
            
            # Sort by similarity score and match type
            similar_medicines.sort(key=lambda x: (x['match_type'] != 'exact', 
                                                x['match_type'] != 'high', 
                                                -x['similarity']))
            return similar_medicines[:5]  # Return top 5 matches
            
        except Exception as e:
            self.logger.error(f"Error finding similar medicines: {e}", exc_info=True)
            return []
            
    def _format_selection_menu(self, medicines: List[Dict[str, Any]], language: str) -> str:
        """Format a selection menu for similar medicines."""
        if language == "es":
            menu = "¿A cuál de estos medicamentos te refieres?\n\n"
            for i, med in enumerate(medicines, 1):
                # Get all brand names
                brand_names = [name for name in [
                    med.get('brand_name1'), 
                    med.get('brand_name2'),
                    med.get('brand_name3'),
                    med.get('brand_name4'),
                    med.get('brand_name5'),
                    med.get('brand_name6')
                ] if name]
                
                menu += f"{i}. {med.get('generic_name')}"
                if brand_names:
                    menu += f" ({', '.join(brand_names)})"
                
                # Add confidence level
                match_type = med.get('match_type', '')
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
                # Get all brand names
                brand_names = [name for name in [
                    med.get('brand_name1'), 
                    med.get('brand_name2'),
                    med.get('brand_name3'),
                    med.get('brand_name4'),
                    med.get('brand_name5'),
                    med.get('brand_name6')
                ] if name]
                
                menu += f"{i}. {med.get('generic_name')}"
                if brand_names:
                    menu += f" ({', '.join(brand_names)})"
                
                # Add confidence level
                match_type = med.get('match_type', '')
                if match_type == 'exact':
                    menu += " [Exact match]"
                elif match_type == 'high':
                    menu += " [Probable match]"
                else:
                    menu += " [Possible match]"
                menu += "\n"
            menu += "\nPlease respond with the corresponding number."
            
        return menu
        
    def _generate_and_add_price(self, medicine_dict: Dict) -> Dict:
        """Retrieves the stored price from the database or generates a new one if not found."""
        # Try to get price from the database first
        medicine_id = medicine_dict.get('medicine_id')
        if medicine_id:
            price_query = "SELECT price FROM Medicines WHERE medicine_id = ?"
            result = self.db_service.execute_query(price_query, (medicine_id,))
            
            if result and result[0][0] is not None:
                # Format to 2 decimal places
                formatted_price = f"{result[0][0]:.2f}"
                medicine_dict['Price'] = formatted_price
                self.logger.info(f"Retrieved stored price ${formatted_price} for MedicineID {medicine_id}")
                return medicine_dict
        
        # Fallback to generating a price if not found in DB
        price = random.uniform(0.50, 4.00)
        # Format to 2 decimal places
        formatted_price = f"{price:.2f}"
        medicine_dict['Price'] = formatted_price
        self.logger.info(f"Generated price ${formatted_price} for MedicineID {medicine_dict.get('medicine_id')}")
        return medicine_dict

    def search_medicines(self, query: str, language: str) -> str:
        """Search for medicines and return formatted results."""
        try:
            # Find similar medicines
            similar_medicines = self._find_similar_medicines(query)
            
            if not similar_medicines:
                return self._get_not_found_message(language)
            
            # If we have an exact match (similarity >= 0.9), show inventory directly
            exact_matches = [m for m in similar_medicines if m.get('similarity', 0) >= self.exact_threshold]
            if exact_matches:
                medicine = exact_matches[0]
                # Get inventory for all stores
                inventory = self.db_service.execute_query("""
                    SELECT s.name, i.quantity
                    FROM Inventory i 
                    JOIN Stores s ON i.store_id = s.store_id 
                    WHERE i.medicine_id = ?
                    ORDER BY s.name
                """, (medicine['medicine_id'],))
                
                # Get price for display
                price_query = "SELECT price FROM Medicines WHERE medicine_id = ?"
                price_result = self.db_service.execute_query(price_query, (medicine['medicine_id'],))
                
                price = None
                if price_result and price_result[0][0] is not None:
                    price = f"{price_result[0][0]:.2f}"
                    medicine['Price'] = price
                
                # Format response based on language
                if language == "es":
                    response = f"Encontré {medicine['generic_name']}"
                    if medicine['brand_name1']:
                        response += f" ({medicine['brand_name1']})"
                    if price:
                        response += f"\nPrecio: ${price}"
                    response += "\n\nDisponible en las siguientes farmacias:\n\n"
                    for idx, (store_name, quantity) in enumerate(inventory, 1):
                        response += f"{idx}. {store_name}: {quantity} unidades\n"
                    response += "\nPor favor, escriba el número de la farmacia donde desea comprar el medicamento."
                else:
                    response = f"I found {medicine['generic_name']}"
                    if medicine['brand_name1']:
                        response += f" ({medicine['brand_name1']})"
                    if price:
                        response += f"\nPrice: ${price}"
                    response += "\n\nAvailable at the following pharmacies:\n\n"
                    for idx, (store_name, quantity) in enumerate(inventory, 1):
                        response += f"{idx}. {store_name}: {quantity} units\n"
                    response += "\nPlease type the number of the pharmacy where you would like to purchase the medicine."
                
                # Store current medicine context for store selection
                self.current_medicine_context = medicine
                self.available_stores = [store[0] for store in inventory]
                return response
            
            # If no exact match, show selection menu
            self.last_search_results = similar_medicines
            return self._format_selection_menu(similar_medicines, language)
            
        except Exception as e:
            self.logger.error(f"Error searching for medicine '{query}': {e}", exc_info=True)
            return self._get_error_message(language)
            
    def handle_selection(self, selection: str, language: str = "es") -> str:
        """Handle user selection from a list, check aggregate stock."""
        self.logger.info(f"Handling selection: {selection} from last results count: {len(self.last_search_results)}")
        
        if not self.last_search_results:
            self.logger.warning("handle_selection called without prior search results.")
            return self._get_error_message(language)
            
        try:
            index = int(selection) - 1
            if 0 <= index < len(self.last_search_results):
                selected_medicine = self.last_search_results[index]
                selected_medicine_generic_name = selected_medicine.get('generic_name')
                # Get brand names from the selected dict as well
                selected_brand_names = [selected_medicine.get(f'brand_name{i}') for i in range(1, 7) if selected_medicine.get(f'brand_name{i}')]
                self.logger.info(f"User selected medicine: {selected_medicine_generic_name}")
                
                # Generate price for the selected medicine
                selected_medicine = self._generate_and_add_price(selected_medicine)
                
                # Check aggregate stock using the *name* and *brands*
                aggregate_stock = self._check_store_stock(generic_name=selected_medicine_generic_name, brand_names=selected_brand_names)
                
                if isinstance(aggregate_stock, int) and aggregate_stock > 0:
                    self.logger.info(f"Found aggregate stock ({aggregate_stock}) for selected medicine.")
                    # Update context - no store selection needed
                    self.current_medicine_context = {'type': 'medicine_selected_stock', 'medicine': selected_medicine}
                    # Use simplified format response
                    return self._format_store_availability_response(selected_medicine, aggregate_stock, language)
                else:
                     self.logger.info("No aggregate stock found for selected medicine.")
                     # Update context - info provided, no stock
                     self.current_medicine_context = {'type': 'medicine_info_no_stock', 'medicine': selected_medicine}
                     # Use general info format (will state no availability)
                     return self._format_medicine_response(selected_medicine, language)
            else:
                self.logger.warning(f"Invalid selection index: {index}")
                # Re-present the menu
                return self._format_selection_menu(self.last_search_results, language)
        except ValueError:
            self.logger.warning(f"Invalid selection format: {selection}")
            # Re-present the menu on bad input
            return self._format_selection_menu(self.last_search_results, language)
        except Exception as e:
            self.logger.error(f"Error handling selection: {e}", exc_info=True)
            return self._get_error_message(language)
            
    def _check_store_stock(self, generic_name: str, brand_names: List[str]) -> Union[int, str]:
        """Check stock by medicine ID in the 'Inventory' table."""
        try:
            # Get the medicine_id from Medicines table first
            medicine_ids = []
            
            # Search by generic name
            if generic_name:
                query = """
                    SELECT medicine_id
                    FROM Medicines
                    WHERE LOWER(generic_name) LIKE ?
                """
                result = self.db_service.execute_query(query, (f"%{generic_name.lower().strip()}%",))
                if result:
                    medicine_ids.extend([row[0] for row in result])
            
            # Search by brand names
            if brand_names:
                for brand in brand_names:
                    if not brand:
                        continue
                    query = """
                        SELECT medicine_id
                        FROM Medicines
                        WHERE 
                            LOWER(brand_name1) LIKE ? OR
                            LOWER(brand_name2) LIKE ? OR
                            LOWER(brand_name3) LIKE ? OR
                            LOWER(brand_name4) LIKE ? OR
                            LOWER(brand_name5) LIKE ? OR
                            LOWER(brand_name6) LIKE ?
                    """
                    brand_param = f"%{brand.lower().strip()}%"
                    result = self.db_service.execute_query(query, (brand_param,) * 6)
                    if result:
                        medicine_ids.extend([row[0] for row in result])
            
            # Remove duplicates
            medicine_ids = list(set(medicine_ids))
            
            if not medicine_ids:
                self.logger.warning(f"No medicine IDs found for {generic_name} or any brand names")
                return "Not available"
            
            self.logger.info(f"Found medicine IDs: {medicine_ids}")
            
            # Now check inventory for these medicine IDs
            total_stock = 0
            for medicine_id in medicine_ids:
                query = """
                    SELECT SUM(quantity) as total
                    FROM Inventory
                    WHERE medicine_id = ?
                """
                result = self.db_service.execute_query(query, (medicine_id,))
                if result and result[0][0] is not None:
                    total_stock += result[0][0]
            
            if total_stock > 0:
                self.logger.info(f"Total stock for {generic_name}: {total_stock}")
                return total_stock
            else:
                self.logger.info(f"No stock available for {generic_name}")
                return "Not available"
            
        except Exception as e:
            self.logger.error(f"Error checking stock: {e}", exc_info=True)
            return "Error checking stock"

    def _format_medicine_response(self, medicine: Dict, language: str = "es") -> str:
        """Format the response for a found medicine, including price and total availability."""
        medicine_id = medicine.get('medicine_id') # Still useful for potential future lookups
        generic_name = medicine.get('generic_name') or medicine.get('generic_name', 'N/A')

        # Get other details (brands, description, prescription) from the medicine dict
        brand_names_list = [name for name in (
            medicine.get('brand_name1'), 
            medicine.get('brand_name2'),
            medicine.get('brand_name3'), 
            medicine.get('brand_name4'),
            medicine.get('brand_name5'), 
            medicine.get('brand_name6')
        ) if name]
        
        # Get aggregate stock using the *name* and *brands*
        total_stock = self._check_store_stock(generic_name=generic_name, brand_names=brand_names_list) 

        if 'Description' not in medicine or 'RequiresPrescription' not in medicine:
            # Fetch if needed - _get_medicine_details still uses ID primarily
            if medicine_id:
                full_details = self._get_medicine_details(medicine_id=medicine_id)
                if full_details:
                    medicine.update(full_details)
            else: # Fallback if ID somehow missing
                full_details = self._get_medicine_details(medicine_name=generic_name)
                if full_details:
                    medicine.update(full_details)

        description = medicine.get('Description', '')
        prescription = medicine.get('RequiresPrescription', False)
        
        # Get price from database directly
        price = None
        if medicine_id:
            try:
                price_query = "SELECT price FROM Medicines WHERE medicine_id = ?"
                price_result = self.db_service.execute_query(price_query, (medicine_id,))
                if price_result and price_result[0][0] is not None:
                    price = f"{price_result[0][0]:.2f}"
            except Exception as e:
                self.logger.error(f"Error fetching price: {e}")
        
        # Use cached price as fallback
        if not price:
            price = medicine.get('Price')

        # Format response including the aggregate stock
        if language == "es":
            response = f"**Medicamento Encontrado:**\n" \
                       f"- **Nombre Genérico:** {generic_name}\n"
            if brand_names_list:
                response += f"- **Nombres de Marca:** {', '.join(brand_names_list)}\n"
            if description:
                 response += f"- **Descripción:** {description}\n"
            response += f"- **Requiere Receta:** {'Sí' if prescription else 'No'}\n"
            if price:
                 response += f"- **Precio:** ${price}\n"
            # Add aggregate stock info
            if isinstance(total_stock, int) and total_stock > 0:
                response += f"- **Disponibilidad Total:** {total_stock} unidades en todas las tiendas.\n"
            else: # Handles 0, "Not available", or "Error checking stock"
                 response += f"- **Disponibilidad Total:** No disponible actualmente.\n"
        else: # English
            response = f"**Medicine Found:**\n" \
                       f"- **Generic Name:** {generic_name}\n"
            if brand_names_list:
                response += f"- **Brand Names:** {', '.join(brand_names_list)}\n"
            if description:
                 response += f"- **Description:** {description}\n"
            response += f"- **Requires Prescription:** {'Yes' if prescription else 'No'}\n"
            if price:
                 response += f"- **Price:** ${price}\n"
            # Add aggregate stock info
            if isinstance(total_stock, int) and total_stock > 0:
                 response += f"- **Total Availability:** {total_stock} units across all stores.\n"
            else: # Handles 0, "Not available", or "Error checking stock"
                 response += f"- **Total Availability:** Currently not available.\n"

        return response.strip()
        
    def _format_store_availability_response(self, medicine: Dict, store_stock: int, language: str = "es") -> str:
        """Format response for medicine with aggregate stock (no store list)."""
        generic_name = medicine.get('generic_name') or medicine.get('generic_name', 'N/A')
        # Corrected brand name generation
        brand_names_list = [name for name in (
            medicine.get('brand_name1'), 
            medicine.get('brand_name2'),
            medicine.get('brand_name3'), 
            medicine.get('brand_name4'),
            medicine.get('brand_name5'), 
            medicine.get('brand_name6')
        ) if name]
        brand_names = f" ({', '.join(brand_names_list)})" if brand_names_list else ""

        # Get price from database directly
        price = None
        medicine_id = medicine.get('medicine_id')
        if medicine_id:
            price_query = "SELECT price FROM Medicines WHERE medicine_id = ?"
            price_result = self.db_service.execute_query(price_query, (medicine_id,))
            if price_result and price_result[0][0] is not None:
                price = f"{price_result[0][0]:.2f}"
        
        # Use cached price as fallback
        if not price:
            price = medicine.get('Price')

        # Check if stock is available (int > 0)
        if not isinstance(store_stock, int) or store_stock <= 0:
            # This medicine shouldn't have reached here if stock <= 0, but handle defensively
            if language == "es":
                return f"Lo siento, {generic_name}{brand_names} no está disponible actualmente." 
            else:
                return f"Sorry, {generic_name}{brand_names} is currently not available."

        # Format response showing total stock
        if language == "es":
            response = f"Encontré **{generic_name}{brand_names}**. "
            if price:
                 response += f"Precio: ${price}. "
            response += f"Hay una disponibilidad total de **{store_stock} unidades** en todas las tiendas.\n"
            response += "¿Necesitas información sobre otro medicamento?" # Changed follow-up
        else: # English
            response = f"Found **{generic_name}{brand_names}**. "
            if price:
                 response += f"Price: ${price}. "
            response += f"There is a total availability of **{store_stock} units** across all stores.\n"
            response += "Do you need information about another medicine?" # Changed follow-up

        return response.strip()
        
    def _get_not_found_message(self, language: str) -> str:
        """Return a message indicating the medicine was not found."""
        if language == "es":
            return "Lo siento, no pude encontrar información sobre ese medicamento. ¿Podrías verificar el nombre e intentarlo de nuevo?"
        else:
            return "I'm sorry, I couldn't find information about that medicine. Could you please check the name and try again?"

    def _get_error_message(self, language: str) -> str:
        """Return a generic error message."""
        if language == "es":
            return "Lo siento, ocurrió un error al procesar tu solicitud. Por favor, intenta de nuevo más tarde."
        else:
            return "I'm sorry, an error occurred while processing your request. Please try again later."

    def _get_medicine_details(self, medicine_id: int = None, medicine_name: str = None) -> dict:
        """Get detailed information about a medicine using named parameters."""
        if not medicine_id and not medicine_name:
            self.logger.warning("Attempted to get medicine details without ID or name.")
            return {}

        try:
            params = {}
            if medicine_id:
                query = "SELECT * FROM Medicines WHERE MedicineID = :med_id"
                params = {"med_id": medicine_id}
            elif medicine_name:
                 query = """
                    SELECT * FROM Medicines 
                    WHERE LOWER([Generic Name]) = :name 
                       OR LOWER([Brand Name 1]) = :name OR LOWER([Brand Name 2]) = :name OR LOWER([Brand Name 3]) = :name
                       OR LOWER([Brand Name 4]) = :name OR LOWER([Brand Name 5]) = :name OR LOWER([Brand Name 6]) = :name
                    LIMIT 1 
                 """
                 params = {"name": medicine_name.lower().strip()}
            else:
                return {} # Should not happen due to initial check
                
            result = self.db_service.execute_query(query, params, fetch_one=True)

            if result:
                self.logger.info(f"Retrieved details for medicine ID: {result.get('MedicineID')}")
                return result # Already a dict
            else:
                 # Fallback to LIKE search if exact match failed for name
                 if medicine_name and not medicine_id:
                     query_like = """
                        SELECT * FROM Medicines 
                        WHERE LOWER([Generic Name]) LIKE :like_name 
                           OR LOWER([Brand Name 1]) LIKE :like_name OR LOWER([Brand Name 2]) LIKE :like_name OR LOWER([Brand Name 3]) LIKE :like_name
                           OR LOWER([Brand Name 4]) LIKE :like_name OR LOWER([Brand Name 5]) LIKE :like_name OR LOWER([Brand Name 6]) LIKE :like_name
                        LIMIT 1
                    """
                     params_like = {"like_name": f"%{medicine_name.lower().strip()}%"}
                     result_like = self.db_service.execute_query(query_like, params_like, fetch_one=True)
                     if result_like:
                         self.logger.info(f"Retrieved details via LIKE search for medicine ID: {result_like.get('MedicineID')}")
                         return result_like

            self.logger.warning(f"No details found for medicine ID {medicine_id} or name '{medicine_name}'")
            return {}
            
        except Exception as e:
            self.logger.error(f"Error getting medicine details: {e}", exc_info=True)
            return {}

    def check_prescription_status(self, medicine_name: str) -> str:
        """Check if a medicine requires a prescription."""
        details = self._get_medicine_details(medicine_name=medicine_name)
        if details:
            requires_prescription = details.get('RequiresPrescription', False)
            generic_name = details.get('Generic Name', medicine_name)
            if requires_prescription:
                return f"{generic_name} requires a prescription."
            else:
                return f"{generic_name} does not require a prescription."
        else:
            return f"Sorry, I couldn't find information about {medicine_name}."

    def get_side_effects(self, medicine_name: str) -> str:
        """Retrieve common side effects for a medicine (placeholder)."""
        # In a real application, this would query a database or external API.
        # For now, return a placeholder based on the medicine name.
        details = self._get_medicine_details(medicine_name=medicine_name)
        if details:
            generic_name = details.get('Generic Name', medicine_name)
            # Example placeholder logic
            if "ibuprofen" in generic_name.lower():
                return "Common side effects of Ibuprofen may include upset stomach, nausea, vomiting, headache, diarrhea, constipation, dizziness, or drowsiness."
            elif "paracetamol" in generic_name.lower() or "acetaminophen" in generic_name.lower():
                return "Acetaminophen (Paracetamol) is generally well-tolerated, but side effects can include nausea or rash. Severe liver damage can occur with overdose."
            else:
                return f"Common side effects for {generic_name} can vary. Please consult a pharmacist or doctor for specific information."
        else:
             return f"Sorry, I couldn't find information about {medicine_name}."

    def find_medicine(self, name: str) -> List[Dict]:
        """Find medicine by name using Python filtering."""
        try:
            self.logger.info(f"Executing find_medicine (Python Filter) for: '{name}'")
            
            # Get all medicines from database
            query = """
                SELECT 
                    m.medicine_id,
                    m.generic_name,
                    m.brand_name1,
                    m.brand_name2,
                    m.brand_name3,
                    m.brand_name4,
                    m.brand_name5,
                    m.brand_name6,
                    m.description,
                    m.side_effects,
                    m.requires_prescription
                FROM Medicines m
            """
            
            all_medicines = self.db_service.execute_query(query)
            if not all_medicines:
                self.logger.error("Database error fetching all medicines.")
                return []
            
            # Convert to list of dictionaries
            medicines = []
            for row in all_medicines:
                medicine = {
                    'medicine_id': row[0],
                    'generic_name': row[1],
                    'brand_name1': row[2],
                    'brand_name2': row[3],
                    'brand_name3': row[4],
                    'brand_name4': row[5],
                    'brand_name5': row[6],
                    'brand_name6': row[7],
                    'description': row[8],
                    'side_effects': row[9],
                    'requires_prescription': row[10]
                }
                medicines.append(medicine)
            
            # Common brand name mappings
            brand_mappings = {
                "tylenol": "acetaminophen",
                "panadol": "acetaminophen",
                "advil": "ibuprofen",
                "motrin": "ibuprofen",
                "benadryl": "diphenhydramine",
                "allegra": "fexofenadine",
                "claritin": "loratadine",
                "zyrtec": "cetirizine"
            }
            
            # Normalize search term
            search_term = name.lower().strip()
            if search_term in brand_mappings:
                search_term = brand_mappings[search_term]
                self.logger.info(f"Mapped brand name '{name}' to generic name '{search_term}'")
            
            # Find exact matches first
            exact_matches = []
            like_matches = []
            
            for med in medicines:
                generic_name = med['generic_name'].lower() if med['generic_name'] else ''
                brand_names = [
                    med['brand_name1'].lower() if med['brand_name1'] else '',
                    med['brand_name2'].lower() if med['brand_name2'] else '',
                    med['brand_name3'].lower() if med['brand_name3'] else '',
                    med['brand_name4'].lower() if med['brand_name4'] else '',
                    med['brand_name5'].lower() if med['brand_name5'] else '',
                    med['brand_name6'].lower() if med['brand_name6'] else ''
                ]
                brand_names = [bn for bn in brand_names if bn]  # Remove empty strings
                
                # Check for exact matches
                if search_term == generic_name:
                    med['match_type'] = 'exact_generic'
                    exact_matches.append(med)
                elif any(search_term == bn for bn in brand_names):
                    med['match_type'] = 'exact_brand'
                    exact_matches.append(med)
                # Check for LIKE matches
                elif search_term in generic_name:
                    med['match_type'] = 'like_generic'
                    like_matches.append(med)
                elif any(search_term in bn for bn in brand_names):
                    med['match_type'] = 'like_brand'
                    like_matches.append(med)
                
            # If we have exact matches, return those
            if exact_matches:
                self.logger.info(f"Found {len(exact_matches)} exact matches")
                return exact_matches
            
            # If we have only one LIKE match, treat it as a good match
            if len(like_matches) == 1:
                like_matches[0]['match_type'] = 'single_like_result'
                self.logger.info("Found single LIKE match")
                return like_matches
            
            # If we have multiple LIKE matches, return all of them
            if like_matches:
                self.logger.info(f"Found {len(like_matches)} LIKE matches")
                return like_matches
            
            # No matches found
            self.logger.info(f"No medicine found for query: {name}")
            return []
            
        except Exception as e:
            self.logger.error(f"Error during Python filtering find_medicine for '{name}': {e}", exc_info=True)
            return []

    def handle_store_selection(self, selection: str, language: str = "es") -> str:
        """Handle user selection of a store, then ask for quantity."""
        try:
            index = int(selection) - 1
            if 0 <= index < len(self.available_stores):
                selected_store = self.available_stores[index]
                medicine_name = self.current_medicine_context.get('generic_name', 'the medicine')
                
                # Store the selected store index for later use
                self.current_medicine_context['selected_store_index'] = index
                
                if language == "es":
                    return f"Ha seleccionado la farmacia: {selected_store}\n\n¿Cuántas unidades de {medicine_name} desea comprar?"
                else:
                    return f"You've selected the pharmacy: {selected_store}\n\nHow many units of {medicine_name} would you like to purchase?"
            else:
                if language == "es":
                    return "Número de farmacia inválido. Por favor, seleccione un número de la lista."
                else:
                    return "Invalid pharmacy number. Please select a number from the list."
        except ValueError:
            if language == "es":
                return "Por favor, ingrese un número válido."
            else:
                return "Please enter a valid number."
        except Exception as e:
            self.logger.error(f"Error handling store selection: {e}", exc_info=True)
            return self._get_error_message(language)
            
    def handle_quantity_selection(self, quantity: str, language: str = "es") -> str:
        """Handle user selection of quantity, then ask to add to order."""
        try:
            quantity_num = int(quantity)
            if quantity_num <= 0:
                if language == "es":
                    return "Por favor, ingrese una cantidad válida mayor que cero."
                else:
                    return "Please enter a valid quantity greater than zero."
                    
            medicine_name = self.current_medicine_context.get('generic_name', 'the medicine')
            
            # Get the price of the medicine
            medicine_id = self.current_medicine_context.get('medicine_id')
            price_per_unit = 0.0
            
            if medicine_id:
                price_query = "SELECT price FROM Medicines WHERE medicine_id = ?"
                result = self.db_service.execute_query(price_query, (medicine_id,))
                
                if result and result[0][0] is not None:
                    price_per_unit = float(result[0][0])
            
            # Check if the quantity is available in the selected store
            if not self.available_stores:
                if language == "es":
                    return "Error: No hay farmacias seleccionadas. Por favor, vuelva a buscar el medicamento."
                else:
                    return "Error: No pharmacies selected. Please search for the medicine again."
                
            store_index = self.current_medicine_context.get('selected_store_index', 0)
            selected_store = self.available_stores[store_index]
            
            # Get the available quantity in the selected store
            available_query = """
                SELECT i.quantity
                FROM Inventory i
                JOIN Stores s ON i.store_id = s.store_id
                WHERE i.medicine_id = ? AND s.name = ?
            """
            result = self.db_service.execute_query(available_query, (medicine_id, selected_store))
            
            if not result:
                if language == "es":
                    return "Error al verificar el inventario. Por favor, intente nuevamente."
                else:
                    return "Error checking inventory. Please try again."
                    
            available_quantity = result[0][0]
            
            if quantity_num > available_quantity:
                if language == "es":
                    return f"Lo sentimos, solo hay {available_quantity} unidades disponibles en {selected_store}. Por favor, ingrese una cantidad menor o igual a {available_quantity}."
                else:
                    return f"Sorry, there are only {available_quantity} units available at {selected_store}. Please enter a quantity less than or equal to {available_quantity}."
            
            # Calculate total price for this purchase
            total_price = price_per_unit * quantity_num
            
            # Store selected store information for cart
            self.current_medicine_context['selected_store'] = selected_store
            self.current_medicine_context['selected_quantity'] = quantity_num
            
            if language == "es":
                return f"Se han agregado {quantity_num} unidades de {medicine_name} a su carrito.\n" + \
                       f"Farmacia: {selected_store}\n" + \
                       f"Precio unitario: ${price_per_unit:.2f}\n" + \
                       f"Total para este medicamento: ${total_price:.2f}\n\n" + \
                       f"¿Desea agregar otro medicamento o finalizar la compra?\n\n" + \
                       f"1. Agregar otro medicamento\n2. Finalizar compra"
            else:
                return f"Added {quantity_num} units of {medicine_name} to your cart.\n" + \
                       f"Pharmacy: {selected_store}\n" + \
                       f"Unit price: ${price_per_unit:.2f}\n" + \
                       f"Total for this medicine: ${total_price:.2f}\n\n" + \
                       f"Would you like to add another medicine or checkout?\n\n" + \
                       f"1. Add another medicine\n2. Checkout"
        except ValueError:
            if language == "es":
                return "Por favor, ingrese una cantidad válida en números."
            else:
                return "Please enter a valid quantity as a number."
        except Exception as e:
            self.logger.error(f"Error handling quantity selection: {e}", exc_info=True)
            return self._get_error_message(language)

    def check_medicine_availability(self, medicine_id: int) -> Dict[str, Any]:
        """Check medicine availability across all stores."""
        try:
            # Get aggregate inventory information
            aggregate_info = self.db_service.get_aggregate_inventory(medicine_id=medicine_id)
            
            if not aggregate_info:
                self.logger.info(f"No aggregate inventory found for medicine_id={medicine_id}")
                return {
                    'available': False,
                    'total_quantity': 0,
                    'stores': []
                }
            
            # Get individual store quantities
            store_quantities = self.db_service.execute_query(
                """
                SELECT s.name, i.quantity
                FROM Inventory i
                JOIN Stores s ON i.store_id = s.store_id
                WHERE i.medicine_id = :medicine_id
                AND i.quantity > 0
                """,
                {'medicine_id': medicine_id}
            )
            
            return {
                'available': aggregate_info['total_quantity'] > 0,
                'total_quantity': aggregate_info['total_quantity'],
                'stores': [{'name': row['name'], 'quantity': row['quantity']} for row in store_quantities]
            }
            
        except Exception as e:
            self.logger.error(f"Error checking medicine availability: {e}", exc_info=True)
            return {
                'available': False,
                'total_quantity': 0,
                'stores': []
            } 