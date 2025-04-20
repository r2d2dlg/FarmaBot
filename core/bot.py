"""
Main FarmaBot class - Core chatbot functionality.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.agents import create_sql_agent
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import re
import random
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
import uuid
import json
from reportlab.lib import colors

from .language_manager import LanguageManager
from services.medicine_service import MedicineService
from services.store_service import StoreService

class FarmaBot:
    def __init__(self, medicine_service: MedicineService, store_service: StoreService, model: str = "gpt-4-turbo"):
        """Initialize the FarmaBot with core components."""
        self.model = model
        
        # Initialize LLM based on model name
        if model.startswith("gemini"):
            google_api_key = os.getenv("GOOGLE_API_KEY")
            if not google_api_key:
                raise ValueError("GOOGLE_API_KEY is required for Gemini models but not found in environment.")
            self.llm = ChatGoogleGenerativeAI(model=model, google_api_key=google_api_key, convert_system_message_to_human=True)
            # TODO: Consider switching to Google embeddings later if needed
        else:
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError("OPENAI_API_KEY is required for non-Gemini models but not found in environment.")
            self.llm = ChatOpenAI(model=model, openai_api_key=openai_api_key)
        
        # Keep OpenAI Embeddings for now, as changing requires vector store rebuild
        self.embeddings = OpenAIEmbeddings()
        self.language_manager = LanguageManager(model)
        self.current_context = None  # Track current conversation context
        # For order flow: item and quantity
        self.order_context = None
        
        # Initialize order logging
        self.session_id = str(uuid.uuid4())
        self.orders_log_file = f"orders_{self.session_id}.json"
        self._initialize_orders_log()
        
        # Store pre-initialized services
        if not medicine_service or not store_service:
            raise ValueError("MedicineService and StoreService instances are required.")
        self.medicine_service = medicine_service
        self.store_service = store_service
        
        # Initialize components
        self.setup_logging()
        self.setup_vectorstore()
        
    def setup_logging(self):
        """Set up logging configuration."""
        logging.basicConfig(
            filename='farmabot_logs.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_vectorstore(self):
        """Set up the vector store for medicine information."""
        try:
            self.vectorstore = Chroma(
                persist_directory="medicines_vectordb_lite",  # Using lightweight version
                embedding_function=self.embeddings
            )
            self.retriever = self.vectorstore.as_retriever()
            self.logger.info("Vector store initialized successfully")
        except Exception as e:
            self.logger.error(f"Error setting up vector store: {e}")
            raise
            
    def process_message(self, message: str, language: str) -> str:
        """Process a user message and return a response."""
        try:
            # Log the interaction
            self.logger.info(f"Processing message in {language}: {message}")
            
            # Check for exit commands
            text_lower = message.lower().strip()
            if text_lower in ['gracias', 'thank you', 'thanks']:
                if language == 'es':
                    return "¡Gracias por usar FarmaBot! ¡Hasta pronto! <exit>"
                else:
                    return "Thank you for using FarmaBot! Goodbye! <exit>"
            
            # Handle pending contexts first
            if self.current_context:
                context_type = self.current_context.get('type')
                
                if context_type == 'order_branch':
                    return self.process_order_branch(message, language)
                elif context_type == 'order_quantity':
                    return self.process_order_quantity(message, language)
                elif context_type == 'post_order':
                    return self.process_post_order_choice(message, language)
            
            # Check for checkout command when there are existing orders
            checkout_terms = ['checkout', 'check out', 'finalizar', 'terminar', 'pagar']
            if any(term in text_lower for term in checkout_terms) and self.current_context and self.current_context.get('orders'):
                return self.process_post_order_choice("2", language)
            
            # If the message is just a medicine name or contains medicine-related terms, process as medicine query
            medicine_terms = ['tienen', 'have', 'busco', 'looking for', 'medicamento', 'medicine', 'drug']
            if (any(term in text_lower for term in medicine_terms) or
                text_lower in ['tylenol', 'panadol', 'aspirin', 'ibuprofeno', 'paracetamol', 'vicodin', 'azprazolam', 'metilfenidato']):
                return self.process_medicine_query(message, language)
            
            # Check for symptom-based medicine advice requests
            symptom_terms = ["dolor", "fiebre", "tos", "náusea", "vomito", "pain", "fever", "cough", "headache"]
            action_terms = ["tomar", "recomendar", "recommend", "take", "advise"]
            if any(sym in text_lower for sym in symptom_terms) and any(act in text_lower for act in action_terms):
                # Disclaim legal advice
                return ("Solo los médicos están autorizados por ley para dar recomendaciones de medicamentos basadas en síntomas. "
                        "Por favor, consulta a un profesional de la salud." if language == "es" else
                        "Only doctors are legally permitted to give medical recommendations based on symptoms. "
                        "Please consult a healthcare professional.")
            
            # Handle store info queries
            if any(keyword in text_lower for keyword in ["tienda", "store", "location", "ubicaciones", "horario", "hours"]):
                return self.process_store_query(message, language)
            
            # If we get here and the message is not empty, treat it as a medicine query
            if message.strip():
                return self.process_medicine_query(message, language)
            
            # If message is empty, provide guidance
            if language == "es":
                return ("Disculpa, no entendí tu solicitud. Puedes preguntarme de las siguientes formas:\n"
                        "- Para disponibilidad de un medicamento: escribe el nombre del medicamento\n"
                        "- Para horarios de atención: escribe 'horarios'\n"
                        "- Para ubicaciones: escribe 'ubicaciones' o 'dónde están nuestras tiendas'")
            else:
                return ("Sorry, I didn't understand that. You can ask me:\n"
                        "- For medication availability: type the medication name\n"
                        "- For opening hours: type 'hours'\n"
                        "- For store locations: type 'locations' or 'where are your stores'")
                
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            return ("Lo siento, tuve un problema al procesar tu mensaje. Por favor, intenta reformular tu pregunta o pregunta sobre otro medicamento."
                     if language == "es" else
                     "I'm sorry, I had trouble processing your message. Please try rephrasing your question or ask about a different medicine.")
            
    def process_medicine_query(self, query: str, language: str) -> str:
        """Process queries related to medicines."""
        try:
            # Clean query: strip Spanish question prefixes to isolate medicine name
            q = query.strip()
            # Remove leading verbs like 'tienen', 'tiene', 'hay', 'tengo' (Spanish)
            if language == 'es':
                q = re.sub(r'(?i)^(tienen|tiene|hay|tengo|busco)\s+', '', q)
            else:
                q = re.sub(r'(?i)^(have|has|looking for)\s+', '', q)
            # Strip punctuation
            q = q.rstrip('?.! ')
            
            # Map brand names to generic names (based on inventory table)
            brand_to_generic = {
                'tylenol': 'Paracetamol',
                'panadol': 'Paracetamol',
                'aspirin': 'Acetylsalicylic Acid',
                'ibuprofeno': 'Ibuprofen',
                'vicodin': 'Hydrocodone',
                'ritalin': 'Methylphenidate',
                'ritalina': 'Methylphenidate',
                'concerta': 'Methylphenidate',
                'adderall': 'Amphetamine',
                'xanax': 'Alprazolam',
                'klonopin': 'Clonazepam',
                'valium': 'Diazepam',
                'nizoral': 'Ketoconazole',
                'nexium': 'Esomeprazole',
                'nasonex': 'Mometasone',
                'neosporin': 'Neomycin',
                'norflex': 'Orphenadrine',
                'norvasc': 'Amlodipine',
                'neurontin': 'Gabapentin',
                'nitro-dur': 'Nitroglycerin',
                'nix': 'Permethrin',
                'nolvadex': 'Tamoxifen',
                'norco': 'Hydrocodone',
                'nordette': 'Levonorgestrel',
                'norpace': 'Disopyramide',
                # Add more mappings as needed
            }
            
            # Map Spanish generic names to English if needed
            if language == 'es':
                spanish_to_english = {
                    'ibuprofeno': 'Ibuprofen',
                    'paracetamol': 'Paracetamol',
                    'aspirina': 'Acetylsalicylic Acid',
                    'metilfenidato': 'Methylphenidate',
                    'alprazolam': 'Alprazolam',
                    'clonazepam': 'Clonazepam',
                    'diazepam': 'Diazepam',
                    'ketoconazol': 'Ketoconazole',
                    # add more mappings as needed
                }
                if q.lower() in spanish_to_english:
                    q = spanish_to_english[q.lower()]
            
            # Check availability in all stores
            stores = self.store_service.db_service.get_store_info()
            availability_info = []
            medicine_found = False
            generic_name = None
            brand_name = None
            
            # First, try to find the medicine in the inventory
            try:
                # Search by both generic name and brand name
                rows = self.store_service.db_service.execute_query(
                    f"SELECT DISTINCT [Generic Name], [Brand Name] FROM dbo.Inventory " +
                    f"WHERE [Generic Name] = '{q}' OR [Brand Name] = '{q}'"
                )
                if rows:
                    medicine_found = True
                    generic_name = rows[0].get('Generic Name')
                    brand_name = rows[0].get('Brand Name')
                else:
                    # If not found directly, check brand_to_generic mapping
                    q_lower = q.lower()
                    if q_lower in brand_to_generic:
                        generic_name = brand_to_generic[q_lower]
                        rows = self.store_service.db_service.execute_query(
                            f"SELECT DISTINCT [Brand Name] FROM dbo.Inventory WHERE [Generic Name] = '{generic_name}'"
                        )
                        if rows:
                            medicine_found = True
                            brand_name = rows[0].get('Brand Name')
            except Exception as e:
                self.logger.error(f"Error querying inventory: {e}")
            
            if not medicine_found:
                # Try to find similar medicines
                similar_medicines = []
                try:
                    # Search for similar names in both generic and brand names
                    rows = self.store_service.db_service.execute_query(
                        f"SELECT DISTINCT [Generic Name], [Brand Name] FROM dbo.Inventory " +
                        f"WHERE [Generic Name] LIKE '%{q}%' OR [Brand Name] LIKE '%{q}%'"
                    )
                    for row in rows:
                        gen = row.get('Generic Name')
                        brand = row.get('Brand Name')
                        if gen and gen not in similar_medicines:
                            similar_medicines.append(gen)
                        if brand and brand not in similar_medicines:
                            similar_medicines.append(brand)
                except Exception as e:
                    self.logger.error(f"Error searching similar medicines: {e}")
                
                if similar_medicines:
                    # Store the similar medicines in context for selection
                    self.current_context = {
                        'type': 'medicine_selection',
                        'medicines': similar_medicines
                    }
                    
                    if language == 'es':
                        response = f"Lo siento, no encontré '{q}'. ¿Quizás quisiste decir uno de estos medicamentos?\n"
                        for i, med in enumerate(similar_medicines, 1):
                            response += f"{i}. {med}\n"
                        response += "\nPor favor, selecciona el número del medicamento que deseas o escribe el nombre correcto."
                    else:
                        response = f"Sorry, I couldn't find '{q}'. Did you mean one of these medications?\n"
                        for i, med in enumerate(similar_medicines, 1):
                            response += f"{i}. {med}\n"
                        response += "\nPlease select the number of the medication you want or type the correct name."
                    
                    return response
                
                # If no similar medicines found
                if self.current_context and self.current_context.get('orders'):
                    if language == 'es':
                        return (f"Lo siento, no tenemos '{q}' disponible. ¿Qué deseas hacer?\n"
                               "1. Buscar otro medicamento\n"
                               "2. Finalizar y generar cotización con los pedidos actuales")
                    else:
                        return (f"Sorry, we don't have '{q}' available. What would you like to do?\n"
                               "1. Look for another medication\n"
                               "2. Checkout and generate quote with current orders")
                else:
                    if language == 'es':
                        return (f"Lo siento, no tenemos '{q}' disponible. ¿Deseas buscar otro medicamento? "
                               "Por favor, dime el nombre del medicamento que te interesa.")
                    else:
                        return (f"Sorry, we don't have '{q}' available. Would you like to look for another medication? "
                               "Please tell me the name of the medicine you're interested in.")
            
            # Check availability in all stores using the found generic name
            available_stores = []
            for store in stores:
                table = store.get('InventoryTableName')
                try:
                    rows = self.store_service.db_service.execute_query(
                        f"SELECT Inventory FROM dbo.[{table}] WHERE [Generic Name] = '{generic_name}'"
                    )
                    inventory = rows[0].get('Inventory', 0) if rows else 0
                    if inventory > 0:
                        store_info = {
                            'location': store.get('Location'),
                            'inventory': inventory
                        }
                        available_stores.append(store_info)
                        availability_info.append(f"- {store.get('Location')}: {inventory} unidades")
                except Exception:
                    continue
            
            if not availability_info:
                if language == 'es':
                    return f"Lo siento, no tenemos '{q}' disponible en ninguna sucursal."
                else:
                    return f"Sorry, we don't have '{q}' available in any store."
            
            # If medicine is found, start the order flow
            self.current_context = {
                'type': 'order_branch',
                'generic': generic_name,
                'available_stores': available_stores  # Store the available stores in context
            }
            
            # Display both generic and brand names in the response
            display_name = f"{brand_name} ({generic_name})" if brand_name and brand_name != generic_name else generic_name
            
            # Format response with numbered store options
            if language == 'es':
                response = f"Disponibilidad de {display_name} en nuestras sucursales:\n\n"
                # Add availability info with numbered options
                for i, store in enumerate(available_stores, 1):
                    response += f"{i}. {store['location']}: {store['inventory']} unidades\n"
                response += "\n(Escribe el número de la sucursal para seleccionarla)"
            else:
                response = f"Availability of {display_name} in our stores:\n\n"
                # Add availability info with numbered options
                for i, store in enumerate(available_stores, 1):
                    response += f"{i}. {store['location']}: {store['inventory']} units\n"
                response += "\n(Type the store number to select it)"
            
            # Save the numbered options in context for validation
            self.current_context.update({
                'store_options': {str(i): store['location'] 
                                for i, store in enumerate(available_stores, 1)}
            })
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing medicine query: {e}")
            if language == "es":
                return "Lo siento, tuve un problema al buscar información sobre ese medicamento. Por favor, verifica el nombre e intenta nuevamente."
            else:
                return "I'm sorry, I had trouble finding information about that medicine. Please check the name and try again."
            
    def process_store_query(self, query: str, language: str) -> str:
        """Process queries related to store information."""
        try:
            lower_q = query.lower()
            stores_all = self.store_service.db_service.get_store_info()
            # Handle specific branch location queries using short store names
            if any(kw in lower_q for kw in ["donde", "dónde"]):
                # Build mapping of short names to full location
                short_map = {}
                for store in stores_all:
                    loc = store.get('Location', '')
                    address = store.get('Address', '')  # Get the address
                    # Use text after '-' as short name or full loc if no separator
                    parts = loc.split('-')
                    short = parts[-1].strip().lower() if len(parts) > 1 else loc.strip().lower()
                    short_map[short] = {'location': loc, 'address': address}
                # Find which branch was asked
                for short, info in short_map.items():
                    if short in lower_q:
                        if language == "es":
                            return f"La tienda de {info['location']} está ubicada en {info['address']}. ¡Puedes consultarla en el mapa interactivo!"
                        else:
                            return f"The {info['location']} store is located at {info['address']}. You can view it on the interactive map below!"
            # Retrieve all store info
            stores = stores_all
            if not stores:
                return ("No pude encontrar información de las tiendas." if language == "es"
                        else "I could not find information about stores.")
            # 1. Operating hours
            if any(kw in lower_q for kw in ["horario", "horarios", "hours"]):
                lines = []
                if language == "es":
                    lines.append("Horarios de atención de nuestras tiendas:")
                else:
                    lines.append("Here are our stores' operating hours:")
                for store in stores:
                    loc = store.get('Location', 'Ubicación Desconocida' if language == 'es' else 'Unknown Location')
                    open_t = store.get('OpenTime', '5:00')
                    close_t = store.get('CloseTime', '22:00')
                    if language == "es":
                        lines.append(f"- {loc}: Abierto de {open_t} a {close_t}, todos los días")
                    else:
                        lines.append(f"- {loc}: Open from {open_t} to {close_t}, every day")
                return "\n".join(lines)
            # 2. Store locations (text only)
            if any(kw in lower_q for kw in ["ubicaciones", "locations"]):
                lines = []
                if language == "es":
                    lines.append("Aquí están las ubicaciones de nuestras tiendas:")
                else:
                    lines.append("Here are our store locations:")
                for store in stores:
                    loc = store.get('Location', 'Ubicación Desconocida' if language == 'es' else 'Unknown Location')
                    address = store.get('Address', 'Dirección no disponible' if language == 'es' else 'Address not available')
                    if language == "es":
                        lines.append(f"- {loc}\n  Dirección: {address}")
                    else:
                        lines.append(f"- {loc}\n  Address: {address}")
                lines.append("\n" + ("¡Puedes ver todas las ubicaciones en el mapa interactivo!" if language == "es" 
                                   else "You can view all locations on the interactive map!"))
                return "\n".join(lines)
            # 3. Clarify or services (future)
            if language == "es":
                return ("¿Qué información necesitas sobre nuestras tiendas? Puedo ayudarte con:\n"
                        "- Horarios de atención\n- Ubicaciones\n- Servicios disponibles")
            else:
                return ("What information do you need about our stores? I can help you with:\n"
                        "- Opening hours\n- Locations\n- Available services")
        except Exception as e:
            self.logger.error(f"Error processing store query: {e}")
            if language == "es":
                return "Lo siento, tuve un problema al obtener la información de las tiendas. Por favor, intenta nuevamente más tarde."
            else:
                return "I'm sorry, I had trouble getting store information. Please try again later."
            
    def process_general_query(self, query: str, language: str) -> str:
        """Process general queries."""
        try:
            system_prompt = f"""You are a helpful pharmacy assistant. Answer the question in a friendly and professional manner.
            If you don't understand the question or it's not clear what the user is asking about, ask for clarification.
            Respond in {'Spanish' if language == 'es' else 'English'}.
            """
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=query)
            ]
            
            return self.llm.invoke(messages).content
            
        except Exception as e:
            self.logger.error(f"Error processing general query: {e}")
            if language == "es":
                return "Lo siento, no pude entender tu pregunta. ¿Podrías reformularla o ser más específico?"
            else:
                return "I'm sorry, I couldn't understand your question. Could you rephrase it or be more specific?"
                
    def _format_medicine_info(self, medicine: Dict[str, Any], language: str) -> str:
        """Format medicine information for display."""
        if language == "es":
            # Spanish branch without side effect information
            return f"""
            Información del medicamento:
            
            Nombre genérico: {medicine.get('GenericName', 'No disponible')}
            Nombre comercial: {medicine.get('BrandName', 'No disponible')}
            Requiere receta: {'Sí' if medicine.get('RequiresPrescription') else 'No'}
            
            Disponibilidad:
            {medicine.get('Availability', 'Información no disponible')}
            """
        else:
            return f"""
            Medicine Information:
            
            Generic Name: {medicine.get('GenericName', 'Not available')}
            Brand Name: {medicine.get('BrandName', 'Not available')}
            Requires Prescription: {'Yes' if medicine.get('RequiresPrescription') else 'No'}
            
            Availability:
            {medicine.get('Availability', 'Information not available')}
            
            Common Side Effects:
            {medicine.get('CommonSideEffects', 'Not available')}
            
            Rare Side Effects:
            {medicine.get('RareSideEffects', 'Not available')}
            """

    def process_medicine_availability(self, message: str, language: str) -> str:
        """Handle queries specifying medicine, dosage form, and branch in one phrase."""
        # Known dosage terms and store locations
        dosage_terms = ["pastilla", "tableta", "gotas", "capsula", "jarabe"]
        store_names = [s.get('Location').lower() for s in self.store_service.db_service.get_store_info()]
        text = message.lower()
        # Extract dosage
        dosage = next((term for term in dosage_terms if term in text), None)
        # Extract branch
        branch = next((loc for loc in store_names if loc in text), None)
        # Extract medicine name by removing keywords
        med_name = message
        if dosage:
            med_name = re.sub(f"(?i) en {dosage}", "", med_name)
        if branch:
            med_name = re.sub(f"(?i) en {branch}", "", med_name)
        # Remove question marks and leading verbs
        med_name = re.sub(r"[¿?]", "", med_name)
        med_name = re.sub(r"(?i)^(tienen|tiene|quiero|me gustaria|busco|buscaria)\s+", "", med_name).strip()
        # Get medicine details
        details = self.medicine_service.get_medicine_details(medicine_name=med_name)
        if not details:
            return (f"No pude encontrar información sobre '{med_name}'" if language == "es"
                    else f"I could not find information about '{med_name}'")
        med = details[0]
        # Get branch info
        stores = self.store_service.db_service.get_store_info(location=branch)
        if not stores:
            return (f"No pude encontrar la sucursal '{branch.title()}'" if language == "es"
                    else f"I could not find branch '{branch}'")
        store = stores[0]
        # Check stock
        generic = med.get('GenericName')
        # Use full list of brand names
        brands = med.get('BrandNames', [])
        stock = self.medicine_service._check_store_stock(generic, brands, store.get('InventoryTableName'))
        # Build response
        if language == "es":
            res = f"Disponibilidad de {generic} en {dosage or 'formato desconocido'} en {store.get('Location')}:\n"
            res += f"- Genérico: {'Disponible' if stock.get('generic_available') else 'No disponible'}\n"
            if stock.get('brand_availability'):
                res += "- Marcas disponibles:\n"
                for b, ok in stock['brand_availability'].items():
                    res += f"  * {b}: {'Disponible' if ok else 'No disponible'}\n"
            return res
        else:
            res = f"Availability of {generic} in {dosage or 'unknown form'} at {store.get('Location')}:\n"
            res += f"- Generic: {'Available' if stock.get('generic_available') else 'Not available'}\n"
            if stock.get('brand_availability'):
                res += "- Available brands:\n"
                for b, ok in stock['brand_availability'].items():
                    res += f"  * {b}: {'Available' if ok else 'Not available'}\n"
            return res 

    def process_store_stock_query(self, message: str, language: str) -> str:
        """Check stock availability of a medicine across all stores."""
        import re
        # Remove Spanish stock query prefix
        q = message.strip()
        q = re.sub(r"(?i)^(tienen|tiene|hay)\s+", "", q)
        q = q.rstrip('?.! ')
        # First, try brand name lookup via aggregated inventory view
        try:
            rows = self.store_service.db_service.execute_query(
                f"SELECT Store, [Brand Name] AS BrandName, Inventory FROM v_inventory "
                f"WHERE LOWER([Brand Name]) LIKE LOWER('%{q}%')"
            )
            if rows:
                lines = []
                if language == 'es':
                    lines.append(f"Disponibilidad de {q} en nuestras tiendas:")
                else:
                    lines.append(f"Availability of {q} in our stores:")
                for row in rows:
                    store = row.get('Store')
                    inv = row.get('Inventory', 0)
                    if language == 'es':
                        status = 'Disponible' if inv > 0 else 'No disponible'
                        lines.append(f"- {store}: {status} ({inv} unidades)")
                    else:
                        status = 'Available' if inv > 0 else 'Not available'
                        lines.append(f"- {store}: {status} ({inv} units)")
                return "\n".join(lines)
        except Exception:
            pass
        # Finally, out-of-stock message when no inventory rows found
        if language == 'es':
            return f"Lo siento, actualmente no tenemos {q} en stock en ninguna sucursal."
        else:
            return f"Sorry, we currently don't have {q} in stock in any store." 

    def process_quote_generate(self, message: str, language: str) -> str:
        """Process price quote for a list of medications."""
        # Parse comma-separated medication names
        meds = [m.strip() for m in message.split(',') if m.strip()]
        if not meds:
            return ("No se detectaron medicamentos para cotizar." if language == 'es'
                    else "No medications detected for quote.")
        # Assign random prices between $1 and $3
        prices = {med: round(random.uniform(1, 3), 2) for med in meds}
        lines = []
        total = 0
        for med, price in prices.items():
            total += price
            lines.append(f"- {med}: ${price}")
        # Build header and footer
        if language == 'es':
            header = "Cotización de precios:"
            footer = f"Precio total estimado: ${round(total, 2)}"
        else:
            header = "Price quote:"
            footer = f"Estimated total price: ${round(total, 2)}"
        # Clear ordering context
        self.current_context = None
        # Return full quote with logo
        logo_md = "![Farma AI Logo](images/farma_AI_logo.png)"
        return f"{logo_md}\n\n{header}\n" + "\n".join(lines) + f"\n{footer}" 

    def process_order_item(self, message: str, language: str) -> str:
        """Start order by parsing medicine and branch, confirm stock, ask for quantity."""
        import re
        # Clean message and extract medicine name
        text = message.strip()
        med_name = re.sub(r"(?i)^(comprar|quiero comprar|purchase|buy)\s+", "", text).strip()
        # Lookup generic name
        details = self.medicine_service.get_medicine_details(medicine_name=med_name)
        if not details:
            return (f"No pude encontrar información sobre '{med_name}'" if language=='es'
                    else f"I could not find '{med_name}' in our catalog.")
        generic = details[0].get('GenericName')
        # Now detect branch if mentioned
        stores = self.store_service.db_service.get_store_info()
        branch = None
        lower = text.lower()
        for store in stores:
            loc = store.get('Location','')
            loc_lower = loc.lower()
            short = loc_lower.split('-')[-1].strip()
            if short in lower or loc_lower in lower:
                branch = store
                break
        # If branch not found, save context and ask for branch
        if not branch:
            self.current_context = {'type': 'order_branch', 'generic': generic}
            return (f"¿En qué sucursal deseas recoger {generic}?" if language=='es'
                    else f"Which store would you like to pick up {generic}?" )
        # Branch found: remove branch phrase and proceed to inventory check
        # Check inventory
        table = branch.get('InventoryTableName')
        try:
            rows = self.store_service.db_service.execute_query(
                f"SELECT Inventory FROM dbo.[{table}] WHERE [Generic Name] = '{generic}'"
            )
            inventory = rows[0].get('Inventory',0) if rows else 0
        except Exception:
            inventory = 0
        if inventory <= 0:
            return (f"Lo siento, no hay '{generic}' disponible en {branch.get('Location')}" if language=='es'
                    else f"Sorry, '{generic}' is not available at {branch.get('Location')}.")
        # Prompt for quantity
        self.current_context = {
            'type': 'order_quantity',
            'generic': generic,
            'branch': branch.get('Location'),
            'inventory': inventory
        }
        return (f"Tenemos {inventory} unidades de {generic} en {branch.get('Location')}. ¿Cuántas deseas comprar?"
                if language=='es' else
                f"We have {inventory} units of {generic} at {branch.get('Location')}. How many would you like to purchase?")

    def _generate_pdf_quote(self, order_details: dict, language: str) -> str:
        """Generate a PDF quote for the order."""
        pdf_path = f"{os.getcwd()}/{uuid.uuid4()}.pdf"
        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Add logo with better positioning and size
        try:
            logo_path = "images/farma_AI_logo.png"
            if os.path.exists(logo_path):
                # Create a table for the logo to control its placement better
                logo = Image(logo_path)
                # Set absolute width and height while maintaining aspect ratio
                logo.drawWidth = 2.5*inch  # Slightly smaller width
                logo.drawHeight = 1.2*inch  # Maintain aspect ratio
                
                # Add white background and center alignment
                logo_table = Table([[logo]], colWidths=[6*inch])
                logo_table.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('BACKGROUND', (0, 0), (-1, -1), colors.white),
                    ('LEFTPADDING', (0, 0), (-1, -1), 0),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 0),
                    ('TOPPADDING', (0, 0), (-1, -1), 20),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 20),
                ]))
                
                story.append(logo_table)
                story.append(Spacer(1, 0.3*inch))  # Add space after logo
            else:
                self.logger.warning(f"Logo file not found at {logo_path}")
        except Exception as e:
            self.logger.error(f"Error adding logo to PDF: {e}")

        # Add title with custom style
        title_style = styles['Heading1']
        title_style.alignment = 1  # Center alignment
        title_style.textColor = colors.HexColor('#1a237e')  # Dark blue color
        title = "Cotización de Pedido" if language == 'es' else "Order Quotation"
        story.append(Paragraph(title, title_style))
        story.append(Spacer(1, 0.3*inch))

        # Add order details with improved formatting
        details_map = {
            'es': {
                'Order': 'Pedido',
                'Medication': 'Medicamento',
                'Store': 'Sucursal',
                'Quantity': 'Cantidad',
                'Unit Price': 'Precio Unitario',
                'Subtotal': 'Subtotal',
                'Total': 'Total',
                'Running Total': 'Total Acumulado'
            },
            'en': {
                'Order': 'Order',
                'Medication': 'Medication',
                'Store': 'Store',
                'Quantity': 'Quantity',
                'Unit Price': 'Unit Price',
                'Subtotal': 'Subtotal',
                'Total': 'Total',
                'Running Total': 'Running Total'
            }
        }
        lang_details = details_map[language]

        # Create styles for details
        detail_style = styles['Normal']
        detail_style.leftIndent = 50
        detail_style.spaceAfter = 10
        detail_style.textColor = colors.HexColor('#424242')  # Dark gray for better readability

        header_style = styles['Heading3']
        header_style.leftIndent = 30
        header_style.spaceAfter = 12
        header_style.textColor = colors.HexColor('#1976d2')  # Blue color for headers

        # Add multiple orders
        orders = order_details.get('orders', [])
        running_total = 0
        for i, order in enumerate(orders, 1):
            # Add order number with background
            order_header = Table([[Paragraph(f"{lang_details['Order']} {i}", header_style)]], 
                               colWidths=[520])
            order_header.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f5f5f5')),  # Light gray background
                ('LEFTPADDING', (0, 0), (-1, -1), 30),
                ('TOPPADDING', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ]))
            story.append(order_header)
            story.append(Spacer(1, 0.1*inch))
            
            # Add order details
            story.append(Paragraph(f"{lang_details['Medication']}: {order['generic']}", detail_style))
            story.append(Paragraph(f"{lang_details['Store']}: {order['branch']}", detail_style))
            story.append(Paragraph(f"{lang_details['Quantity']}: {order['count']}", detail_style))
            story.append(Paragraph(f"{lang_details['Unit Price']}: ${order['price']:.2f}", detail_style))
            story.append(Paragraph(f"{lang_details['Subtotal']}: ${order['total']:.2f}", detail_style))
            story.append(Spacer(1, 0.2*inch))
            
            running_total += order['total']

        # Add final total with bold style and background
        total_style = styles['Heading2']
        total_style.textColor = colors.HexColor('#1a237e')  # Dark blue for total
        
        total_table = Table([[Paragraph(f"{lang_details['Total']}: ${running_total:.2f}", total_style)]], 
                           colWidths=[520])
        total_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#e3f2fd')),  # Light blue background
            ('ALIGN', (0, 0), (-1, -1), 'RIGHT'),
            ('LEFTPADDING', (0, 0), (-1, -1), 50),
            ('RIGHTPADDING', (0, 0), (-1, -1), 50),
            ('TOPPADDING', (0, 0), (-1, -1), 15),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 15),
        ]))
        
        story.append(Spacer(1, 0.3*inch))
        story.append(total_table)
        story.append(Spacer(1, 0.5*inch))

        # Add footer message with italic style
        footer_style = styles['Italic']
        footer_style.alignment = 1  # Center alignment
        footer_style.textColor = colors.HexColor('#1976d2')  # Blue color for footer
        footer = "¡Gracias por su compra!" if language == 'es' else "Thank you for your purchase!"
        story.append(Paragraph(footer, footer_style))

        # Build the PDF with a frame
        doc.build(story)
        self.logger.info(f"Generated PDF quote at: {pdf_path}")
        return pdf_path

    def _initialize_orders_log(self):
        """Initialize the orders log file for this session."""
        try:
            with open(self.orders_log_file, 'w') as f:
                json.dump({'orders': [], 'total': 0.0}, f)
        except Exception as e:
            self.logger.error(f"Error initializing orders log: {e}")

    def _add_to_orders_log(self, order: dict):
        """Add an order to the log file."""
        try:
            with open(self.orders_log_file, 'r') as f:
                data = json.load(f)
            
            data['orders'].append(order)
            data['total'] = sum(order['total'] for order in data['orders'])
            
            with open(self.orders_log_file, 'w') as f:
                json.dump(data, f)
            
            return data['total']
        except Exception as e:
            self.logger.error(f"Error adding to orders log: {e}")
            return 0.0

    def _get_orders_from_log(self):
        """Get all orders from the log file."""
        try:
            with open(self.orders_log_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error reading orders log: {e}")
            return {'orders': [], 'total': 0.0}

    def _cleanup_orders_log(self):
        """Remove the temporary orders log file."""
        try:
            if os.path.exists(self.orders_log_file):
                os.remove(self.orders_log_file)
        except Exception as e:
            self.logger.error(f"Error cleaning up orders log: {e}")

    def process_order_quantity(self, message: str, language: str) -> str:
        """Finalize order with quantity, generate PDF quote, clear context."""
        ctx = self.current_context or {}
        try:
            count = int(message.strip())
        except ValueError:
            return ("Por favor ingrese un número válido." if language == 'es'
                    else "Please enter a valid number.")
        if count <= 0:
             return ("Por favor ingrese una cantidad positiva." if language == 'es'
                     else "Please enter a positive quantity.")
        inventory = ctx.get('inventory', 0)
        generic = ctx.get('generic', '')
        branch = ctx.get('branch', '')
        price = ctx.get('price', 0)  # Get the price that was already generated
        
        if count > inventory:
            return (f"Solo tenemos {inventory} unidades disponibles. Intente con una cantidad menor." if language == 'es'
                    else f"We only have {inventory} units available. Please choose a smaller quantity.")
        
        total = round(price * count, 2)
        
        # Create order details
        order = {
            'generic': generic,
            'branch': branch,
            'count': count,
            'price': price,
            'total': total
        }
        
        # Add to orders log
        total_amount = self._add_to_orders_log(order)
        
        # Update context type for post-order choice
        self.current_context = {'type': 'post_order'}
        
        # Format the quote details
        if language == 'es':
            quote_details = (
                f"Resumen del pedido:\n"
                f"Medicamento: {generic}\n"
                f"Sucursal: {branch}\n"
                f"Cantidad: {count}\n"
                f"Precio por unidad: ${price:.2f}\n"
                f"Total: ${total:.2f}\n"
                f"\nTotal acumulado: ${total_amount:.2f}\n\n"
                f"¿Qué deseas hacer?\n"
                f"1. Agregar otro medicamento\n"
                f"2. Generar cotización final"
            )
        else:
            quote_details = (
                f"Order summary:\n"
                f"Medicine: {generic}\n"
                f"Store: {branch}\n"
                f"Quantity: {count}\n"
                f"Price per unit: ${price:.2f}\n"
                f"Total: ${total:.2f}\n"
                f"\nRunning total: ${total_amount:.2f}\n\n"
                f"What would you like to do?\n"
                f"1. Add another medication\n"
                f"2. Generate final quote"
            )
        
        return quote_details

    def process_post_order_choice(self, message: str, language: str) -> str:
        """Handle the choice after adding an order."""
        choice = message.strip()
        
        if choice == "1":
            # Clear current context but keep the orders log
            self.current_context = {'type': None}
            return ("Por favor, dime el nombre del medicamento que te interesa." if language == 'es'
                    else "Please tell me the name of the medicine you're interested in.")
        elif choice == "2":
            # Get all orders from the log
            data = self._get_orders_from_log()
            orders = data.get('orders', [])
            
            if not orders:
                return ("No hay pedidos para generar una cotización." if language == 'es'
                        else "There are no orders to generate a quote.")
            
            # Generate PDF with all orders
            pdf_path = self._generate_pdf_quote(data, language)
            
            # Clean up the orders log
            self._cleanup_orders_log()
            
            # Clear the entire context
            self.current_context = None
            
            return pdf_path
        else:
            return ("Por favor selecciona 1 para agregar otro medicamento o 2 para generar la cotización." if language == 'es'
                    else "Please select 1 to add another medication or 2 to generate the quote.")

    def process_order_branch(self, message: str, language: str) -> str:
        """Handle branch selection step: check inventory and ask quantity."""
        text = message.strip()
        stores = self.store_service.db_service.get_store_info()
        branch = None
        
        # Get store options from context
        ctx = self.current_context or {}
        store_options = ctx.get('store_options', {})
        
        # First, try to match against numbered options
        if text in store_options:
            store_location = store_options[text]
            # Find the full store info
            for store in stores:
                if store.get('Location') == store_location:
                    branch = store
                    break
        
        # If no match found with numbers, try other matching methods
        if not branch:
            # Map simple location names to full store names (case insensitive)
            location_map = {
                'chorrera': 'Panamá Oeste - La Chorrera',
                'la chorrera': 'Panamá Oeste - La Chorrera',
                'david': 'Chiriquí - David',
                'costa del este': 'Panama City - Costa del Este',
                'el dorado': 'Panama City - El Dorado',
                'dorado': 'Panama City - El Dorado',
                'san francisco': 'Panama City - San Francisco',
                'francisco': 'Panama City - San Francisco'
            }
            
            text_lower = text.lower()
            # First try exact matches from the location map
            if text_lower in location_map:
                full_name = location_map[text_lower]
                for store in stores:
                    if store.get('Location') == full_name:
                        branch = store
                        break
            
            # If no exact match, try partial matches from the location map
            if not branch:
                for simple_name, full_name in location_map.items():
                    if simple_name in text_lower:
                        for store in stores:
                            if store.get('Location') == full_name:
                                branch = store
                                break
                        if branch:
                            break

            # If still no match, try matching against full store names
            if not branch:
                for store in stores:
                    loc = store.get('Location', '').lower()
                    if loc in text_lower or text_lower in loc:
                        branch = store
                        break

        if not branch:
            # Show available stores as numbered options
            if store_options:
                store_list = "\n".join([f"{num}. {loc}" for num, loc in store_options.items()])
                return (f"Por favor, selecciona una sucursal válida escribiendo su número:\n\n{store_list}" if language=='es'
                        else f"Please select a valid store by typing its number:\n\n{store_list}")
            else:
                # If no store options in context, show all stores
                valid_stores = [store.get('Location') for store in stores]
                store_list = "\n".join([f"{i+1}. {store}" for i, store in enumerate(valid_stores)])
                return (f"Por favor, selecciona una sucursal válida escribiendo su número:\n\n{store_list}" if language=='es'
                        else f"Please select a valid store by typing its number:\n\n{store_list}")

        # Retrieve order details
        generic = ctx.get('generic','')
        
        # Check inventory at selected branch
        table = branch.get('InventoryTableName')
        try:
            rows = self.store_service.db_service.execute_query(
                f"SELECT Inventory FROM dbo.[{table}] WHERE [Generic Name] = '{generic}'"
            )
            inventory = rows[0].get('Inventory',0) if rows else 0
        except Exception:
            inventory = 0

        if inventory <= 0:
            return (f"Lo siento, no hay '{generic}' disponible en {branch.get('Location')}" if language=='es'
                    else f"Sorry, '{generic}' is not available at {branch.get('Location')}.")

        # Generate random price
        import random
        price = round(random.uniform(1, 3), 2)

        # Save branch, inventory and price in context
        self.current_context.update({
            'type': 'order_quantity',
            'generic': generic,
            'branch': branch.get('Location'),
            'inventory': inventory,
            'price': price
        })
        
        return (f"Tenemos {inventory} unidades de {generic} en {branch.get('Location')}.\nPrecio: ${price:.2f} por unidad\n¿Cuántas deseas comprar?"
                if language=='es' else
                f"We have {inventory} units of {generic} at {branch.get('Location')}.\nPrice: ${price:.2f} per unit\nHow many would you like to purchase?") 