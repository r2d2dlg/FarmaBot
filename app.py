"""
FarmaBot Hugging Face Spaces Deployment
This script creates a simplified version of the FarmaBot interface 
with SQLite database for deployment on Hugging Face.
"""

import os
from dotenv import load_dotenv
import logging
import sqlite3
import gradio as gr
import uuid
from datetime import datetime
from typing import List, Dict

# Load environment variables
load_dotenv()

from core.bot import FarmaBot
from core.language_manager import LanguageManager
from services.database_service import DatabaseService
from services.medicine_service import MedicineService
from services.store_service import StoreService
from init_db import initialize_database

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('farmabot.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Initialize the database
initialize_database()

# Global conversation state dictionary
conversation_states = {}

def create_chatbot_interface():
    """
    Create a Gradio interface for the chatbot.
    """
    # Initialize services with SQLite database
    db_path = "farmabot.db"
    
    # Select model based on available API keys
    openai_api_key = os.getenv("OPENAI_API_KEY")
    google_api_key = os.getenv("GOOGLE_API_KEY")
    
    if google_api_key:
        model_name = "gemini-2.5-flash"
    elif openai_api_key:
        model_name = os.getenv("MODEL", "gpt-3.5-turbo")
    else:
        logger.warning("No API keys found. Using Gemini as default but expect errors.")
        model_name = "gemini-2.5-flash"
    
    try:
        # Initialize services
        logger.info("Initializing services...")
        db_service = DatabaseService(db_path=db_path)
        medicine_service = MedicineService(db_service)
        store_service = StoreService(db_service)
        
        # Initialize language manager with the same model
        language_manager = LanguageManager(model=model_name)
        
        # Initialize bot with services
        logger.info(f"Initializing FarmaBot with model: {model_name}...")
        bot = FarmaBot(
            medicine_service=medicine_service, 
            store_service=store_service,
            model=model_name
        )
        
        def handle_user_message(message: str, history, state) -> dict:
            try:
                logger.info(f"Current state: {state}")
                
                # If this is the first message (empty history), show language selection
                if not history:
                    return {
                        "role": "assistant",
                        "content": "Please select your language / Por favor seleccione su idioma:\n1. Español\n2. English"
                    }
                
                # Ensure cart exists in state
                if 'cart' not in state:
                    state['cart'] = []
                
                # Language selection
                if state['stage'] == 'language_selection':
                    if message.lower() in ['1', 'es', 'spanish', 'español']:
                        state['language'] = 'es'
                        state['stage'] = 'main_menu'
                        logger.info("Selected Spanish, moving to main menu")
                        return {
                            "role": "assistant",
                            "content": "¿Qué te gustaría hacer?\n\n1. Buscar medicamentos\n2. Ver ubicaciones\n3. Ver horarios"
                        }
                    elif message.lower() in ['2', 'en', 'english']:
                        state['language'] = 'en'
                        state['stage'] = 'main_menu'
                        logger.info("Selected English, moving to main menu")
                        return {
                            "role": "assistant",
                            "content": "What would you like to do?\n\n1. Search medicines\n2. View locations\n3. View hours"
                        }
                    else:
                        logger.info("Invalid language selection")
                        return {
                            "role": "assistant",
                            "content": "Please select your language / Por favor seleccione su idioma:\n1. Español\n2. English"
                        }
                
                # Main menu selection
                if state['stage'] == 'main_menu':
                    logger.info(f"Processing main menu selection: {message}")
                    if message == '1':
                        state['stage'] = 'medicine_search'
                        logger.info("Moving to medicine search")
                        content = "¿Qué medicamento estás buscando?" if state['language'] == 'es' else "What medicine are you looking for?"
                        return {
                            "role": "assistant",
                            "content": content
                        }
                    elif message == '2':
                        # Handle locations
                        logger.info("Showing locations")
                        content = store_service.get_store_locations(state['language'])
                        if state['language'] == 'es':
                            content += "\n\n¿Qué te gustaría hacer?\n\n1. Buscar medicamentos\n2. Ver ubicaciones\n3. Ver horarios"
                        else:
                            content += "\n\nWhat would you like to do?\n\n1. Search medicines\n2. View locations\n3. View hours"
                        return {
                            "role": "assistant",
                            "content": content
                        }
                    elif message == '3':
                        # Handle hours
                        logger.info("Showing hours")
                        content = store_service.get_store_hours(state['language'])
                        if state['language'] == 'es':
                            content += "\n\n¿Qué te gustaría hacer?\n\n1. Buscar medicamentos\n2. Ver ubicaciones\n3. Ver horarios"
                        else:
                            content += "\n\nWhat would you like to do?\n\n1. Search medicines\n2. View locations\n3. View hours"
                        return {
                            "role": "assistant",
                            "content": content
                        }
                    else:
                        # Invalid selection
                        logger.info("Invalid main menu selection")
                        if state['language'] == 'es':
                            content = "Opción inválida. ¿Qué te gustaría hacer?\n\n1. Buscar medicamentos\n2. Ver ubicaciones\n3. Ver horarios"
                        else:
                            content = "Invalid option. What would you like to do?\n\n1. Search medicines\n2. View locations\n3. View hours"
                        return {
                            "role": "assistant",
                            "content": content
                        }
                
                # Medicine search
                if state['stage'] == 'medicine_search':
                    logger.info(f"Processing medicine search: {message}")
                    # Check if the input is a menu option
                    if message in ['1', '2', '3']:
                        content = "Por favor, ingrese el nombre del medicamento que está buscando." if state['language'] == 'es' else "Please enter the name of the medicine you are looking for."
                        return {
                            "role": "assistant",
                            "content": content
                        }
                    
                    response = medicine_service.search_medicines(message, state['language'])
                    if "Available at" in response or "Disponible en" in response:
                        state['stage'] = 'store_selection'
                        state['current_medicine'] = message
                    return {
                        "role": "assistant",
                        "content": response
                    }
                
                # Store selection
                if state['stage'] == 'store_selection':
                    logger.info(f"Processing store selection: {message}")
                    response = medicine_service.handle_store_selection(message, state['language'])
                    # If response is asking for quantity, update stage
                    if "¿Cuántas unidades" in response or "How many units" in response:
                        state['stage'] = 'quantity_selection'
                    return {
                        "role": "assistant",
                        "content": response
                    }
                
                # Quantity selection
                if state['stage'] == 'quantity_selection':
                    logger.info(f"Processing quantity selection: {message}")
                    response = medicine_service.handle_quantity_selection(message, state['language'])
                    
                    # If quantity was successfully added to cart
                    if "agregado" in response or "Added" in response:
                        try:
                            quantity = int(message)
                            medicine_name = medicine_service.current_medicine_context.get('generic_name', 'Unknown medicine')
                            
                            # Get price information
                            medicine_id = medicine_service.current_medicine_context.get('medicine_id')
                            price_per_unit = 0.0
                            
                            if medicine_id:
                                price_query = "SELECT price FROM Medicines WHERE medicine_id = ?"
                                result = db_service.execute_query(price_query, (medicine_id,))
                                
                                if result and result[0][0] is not None:
                                    price_per_unit = float(result[0][0])
                            
                            # Calculate total for this item
                            item_total = price_per_unit * quantity
                            
                            # Get selected store from medicine service context
                            selected_store = medicine_service.current_medicine_context.get('selected_store', 'Unknown pharmacy')
                            
                            # Add to cart with store information
                            cart_item = {
                                'medicine': medicine_name,
                                'quantity': quantity,
                                'price_per_unit': price_per_unit,
                                'item_total': item_total,
                                'pharmacy': selected_store,
                                'timestamp': datetime.now().isoformat()
                            }
                            state['cart'].append(cart_item)
                            logger.info(f"Added to cart: {cart_item}")
                        except ValueError:
                            logger.warning(f"Failed to parse quantity: {message}")
                    
                    if "add another medicine" in response.lower() or "agregar otro medicamento" in response.lower():
                        state['stage'] = 'purchase_option'
                    
                    return {
                        "role": "assistant",
                        "content": response
                    }
                
                # Purchase option
                if state['stage'] == 'purchase_option':
                    logger.info(f"Processing purchase option: {message}")
                    if message == '1':
                        state['stage'] = 'medicine_search'
                        content = "¿Qué medicamento estás buscando?" if state['language'] == 'es' else "What medicine are you looking for?"
                        return {
                            "role": "assistant",
                            "content": content
                        }
                    elif message == '2':
                        state['stage'] = 'checkout'
                        # Show cart summary immediately
                        cart_items = state.get('cart', [])
                        
                        if not cart_items:
                            if state['language'] == 'es':
                                content = "Su carrito está vacío. ¿Desea realizar una búsqueda?\n1. Sí\n2. No"
                            else:
                                content = "Your cart is empty. Would you like to make a search?\n1. Yes\n2. No"
                            return {
                                "role": "assistant", 
                                "content": content
                            }
                        
                        # Format cart summary
                        if state['language'] == 'es':
                            content = "**Resumen de su compra:**\n\n"
                            total_cart_value = 0.0
                            for i, item in enumerate(cart_items, 1):
                                item_total = item.get('item_total', 0.0)
                                price_per_unit = item.get('price_per_unit', 0.0)
                                pharmacy = item.get('pharmacy', 'Farmacia desconocida')
                                total_cart_value += item_total
                                content += f"{i}. {item['medicine']} - {item['quantity']} unidades - ${price_per_unit:.2f}/unidad - Farmacia: {pharmacy} - Total: ${item_total:.2f}\n"
                            content += f"\n**Total de la compra: ${total_cart_value:.2f}**\n"
                            
                            # Generate receipt summary instead of PDF
                            receipt_id = str(uuid.uuid4())[:8]
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            content += f"\nNúmero de orden: {receipt_id}"
                            content += f"\nFecha: {timestamp}"
                            content += "\n\nSu compra ha sido procesada. ¿Desea realizar otra búsqueda?\n1. Sí\n2. No"
                        else:
                            content = "**Purchase Summary:**\n\n"
                            total_cart_value = 0.0
                            for i, item in enumerate(cart_items, 1):
                                item_total = item.get('item_total', 0.0)
                                price_per_unit = item.get('price_per_unit', 0.0)
                                pharmacy = item.get('pharmacy', 'Unknown pharmacy')
                                total_cart_value += item_total
                                content += f"{i}. {item['medicine']} - {item['quantity']} units - ${price_per_unit:.2f}/unit - Pharmacy: {pharmacy} - Total: ${item_total:.2f}\n"
                            content += f"\n**Total purchase value: ${total_cart_value:.2f}**\n"
                            
                            # Generate receipt summary instead of PDF
                            receipt_id = str(uuid.uuid4())[:8]
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            content += f"\nOrder number: {receipt_id}"
                            content += f"\nDate: {timestamp}"
                            content += "\n\nYour purchase has been processed. Would you like to make another search?\n1. Yes\n2. No"
                        
                        # Mark that we've shown the summary
                        state['checkout_summary_shown'] = True
                        return {
                            "role": "assistant",
                            "content": content
                        }
                    else:
                        if state['language'] == 'es':
                            content = "Opción inválida. Por favor seleccione:\n1. Agregar otro medicamento\n2. Finalizar compra"
                        else:
                            content = "Invalid option. Please select:\n1. Add another medicine\n2. Checkout"
                        return {
                            "role": "assistant",
                            "content": content
                        }
                
                # Checkout completion
                if state['stage'] == 'checkout':
                    logger.info(f"Processing checkout: {message}")
                    
                    # Show cart summary if coming from store selection and not already shown
                    if message == '2' and not state.get('checkout_summary_shown'):
                        cart_items = state.get('cart', [])
                        
                        if not cart_items:
                            if state['language'] == 'es':
                                content = "Su carrito está vacío. ¿Desea realizar una búsqueda?\n1. Sí\n2. No"
                            else:
                                content = "Your cart is empty. Would you like to make a search?\n1. Yes\n2. No"
                            return {
                                "role": "assistant",
                                "content": content
                            }
                        
                        # Format cart summary
                        if state['language'] == 'es':
                            content = "**Resumen de su compra:**\n\n"
                            total_cart_value = 0.0
                            for i, item in enumerate(cart_items, 1):
                                item_total = item.get('item_total', 0.0)
                                price_per_unit = item.get('price_per_unit', 0.0)
                                pharmacy = item.get('pharmacy', 'Farmacia desconocida')
                                total_cart_value += item_total
                                content += f"{i}. {item['medicine']} - {item['quantity']} unidades - ${price_per_unit:.2f}/unidad - Farmacia: {pharmacy} - Total: ${item_total:.2f}\n"
                            content += f"\n**Total de la compra: ${total_cart_value:.2f}**\n"
                            
                            # Generate receipt summary instead of PDF
                            receipt_id = str(uuid.uuid4())[:8]
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            content += f"\nNúmero de orden: {receipt_id}"
                            content += f"\nFecha: {timestamp}"
                            content += "\n\nSu compra ha sido procesada. ¿Desea realizar otra búsqueda?\n1. Sí\n2. No"
                        else:
                            content = "**Purchase Summary:**\n\n"
                            total_cart_value = 0.0
                            for i, item in enumerate(cart_items, 1):
                                item_total = item.get('item_total', 0.0)
                                price_per_unit = item.get('price_per_unit', 0.0)
                                pharmacy = item.get('pharmacy', 'Unknown pharmacy')
                                total_cart_value += item_total
                                content += f"{i}. {item['medicine']} - {item['quantity']} units - ${price_per_unit:.2f}/unit - Pharmacy: {pharmacy} - Total: ${item_total:.2f}\n"
                            content += f"\n**Total purchase value: ${total_cart_value:.2f}**\n"
                            
                            # Generate receipt summary instead of PDF
                            receipt_id = str(uuid.uuid4())[:8]
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            content += f"\nOrder number: {receipt_id}"
                            content += f"\nDate: {timestamp}"
                            content += "\n\nYour purchase has been processed. Would you like to make another search?\n1. Yes\n2. No"
                        
                        # Mark that we've shown the summary
                        state['checkout_summary_shown'] = True
                        return {
                            "role": "assistant",
                            "content": content
                        }
                    
                    if message == '1':
                        state['stage'] = 'main_menu'
                        # Reset cart after checkout
                        state['cart'] = []
                        state['checkout_summary_shown'] = False
                        if state['language'] == 'es':
                            content = "¿Qué te gustaría hacer?\n\n1. Buscar medicamentos\n2. Ver ubicaciones\n3. Ver horarios"
                        else:
                            content = "What would you like to do?\n\n1. Search medicines\n2. View locations\n3. View hours"
                        return {
                            "role": "assistant",
                            "content": content
                        }
                    elif message == '2':
                        # Reset conversation state including cart
                        state.update({
                            'stage': 'language_selection',
                            'language': None,
                            'current_medicine': None,
                            'cart': [],
                            'checkout_summary_shown': False
                        })
                        return {
                            "role": "assistant",
                            "content": "Please select your language / Por favor seleccione su idioma:\n1. Español\n2. English"
                        }
                    else:
                        if state['language'] == 'es':
                            content = "Opción inválida. ¿Deseas realizar otra búsqueda?\n1. Sí\n2. No"
                        else:
                            content = "Invalid option. Would you like to make another search?\n1. Yes\n2. No"
                        return {
                            "role": "assistant",
                            "content": content
                        }
                
                # Handle unknown state
                logger.info("Unknown state or message")
                content = "Lo siento, no entiendo." if state['language'] == 'es' else "Sorry, I don't understand."
                return {
                    "role": "assistant",
                    "content": content
                }
                
            except Exception as e:
                logger.error(f"Error handling message: {e}", exc_info=True)
                return {
                    "role": "assistant",
                    "content": "An error occurred. Please try again. / Ocurrió un error. Por favor, intente de nuevo."
                }
        
        # Create the Gradio interface
        interface = gr.ChatInterface(
            fn=handle_user_message,
            title="FarmaBot",
            description="Farma AI - Tu Farmacia Inteligente",
            examples=None,
            theme="soft",
            chatbot=gr.Chatbot(
                value=[{
                    "role": "assistant",
                    "content": "Please select your language / Por favor seleccione su idioma:\n1. Español\n2. English"
                }],
                type="messages"
            ),
            additional_inputs=[gr.State({'stage': 'language_selection', 'language': None, 'current_medicine': None, 'cart': []})],
            additional_outputs=None,
            type="messages"
        )
        
        return interface
        
    except Exception as e:
        logger.error(f"Error creating chatbot interface: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        demo = create_chatbot_interface()
        demo.launch()
    except Exception as e:
        logger.critical(f"Failed to launch Gradio interface: {e}", exc_info=True)
        raise 