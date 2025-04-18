"""
Gradio interface for FarmaBot - Provides a user-friendly chat interface.
"""

import gradio as gr
from typing import List, Dict, Any, Tuple, Optional
import logging
from datetime import datetime
import os
import folium
import html
from dotenv import load_dotenv

from core.bot import FarmaBot
from services.database_service import DatabaseService
from services.medicine_service import MedicineService
from services.store_service import StoreService

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Coordinates for stores (name -> lat, lon)
STORE_COORDS: Dict[str, Tuple[float, float]] = {
    "Panamá Oeste - La Chorrera": (8.997, -79.752),
    "Panama City - Costa del Este": (8.967, -79.545),
    "Chiriquí - David": (8.433, -82.433),
    "Panama City - El Dorado": (8.983, -79.516),
    "Panama City - San Francisco": (8.990, -79.518),
}

def create_interface(bot: FarmaBot) -> gr.Blocks:
    """Create the Gradio interface for the chatbot."""
    
    # Helper to generate interactive map HTML of store locations
    def generate_map_html(language: str) -> str:
        """Generate a full map of all store locations."""
        # Center on Panama
        m = folium.Map(location=[8.5, -79.9], zoom_start=7)
        for name, (lat, lon) in STORE_COORDS.items():
            folium.Marker(location=[lat, lon], popup=name).add_to(m)
        full_html = m.get_root().render()
        escaped = html.escape(full_html)
        iframe = f'<iframe srcdoc="{escaped}" width="100%" height="500" style="border:none;"></iframe>'
        # Interactive directions link
        origin = f"{next(iter(STORE_COORDS.values()))[0]},{next(iter(STORE_COORDS.values()))[1]}"
        waypoints = "|".join(f"{lat},{lon}" for lat, lon in STORE_COORDS.values())
        maps_link = f"https://www.google.com/maps/dir/?api=1&origin={origin}&waypoints={waypoints}"
        link_text = "Abrir mapa interactivo" if language == "es" else "Open interactive map"
        link_tag = f'<a href="{maps_link}" target="_blank">{link_text}</a>'
        return iframe + "<br>" + link_tag
    
    # Custom CSS for a modern look
    css = """
    .gradio-container {
        max-width: 800px;
        margin: auto;
    }
    .chatbot {
        min-height: 400px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .user-message {
        background-color: #e3f2fd;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
    .bot-message {
        background-color: #f5f5f5;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
    .language-selector {
        margin-bottom: 20px;
    }
    .submit-button {
        background-color: #2196f3;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
    }
    .submit-button:hover {
        background-color: #1976d2;
    }
    .menu-button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
        margin: 5px;
        width: 200px;
    }
    .menu-button:hover {
        background-color: #45a049;
    }
    .file-output {
        margin: 10px 0;
        text-align: center;
    }
    .file-output a {
        display: inline-block;
        background-color: #2196f3;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        text-decoration: none;
        margin: 5px;
    }
    .file-output a:hover {
        background-color: #1976d2;
    }
    """
    
    with gr.Blocks(css=css) as interface:
        # Display Farma AI logo at quarter container width
        gr.Image(value="images/farma_AI_logo.png", interactive=False, show_label=False, width=200)
        gr.Markdown("# Farma AI Bot - A Tu Servicio")
        
        # Add instruction box above chat
        instruction_box = gr.Markdown(
            """<div style="padding: 10px; margin-bottom: 10px; background-color: #e3f2fd; border-radius: 5px; text-align: center;">
            🔍 Escribe 'Gracias' para terminar / Type 'Thank you' to exit
            </div>"""
        )
        
        # Add a message box to guide users
        gr.Markdown("""
        # ¡Bienvenido a Farma AI Bot! 👋 \\ Welcome to Farma AI Bot! 👋
        
        Por favor, selecciona una de las opciones a continuación antes de comenzar tu conversación:
        
        Please select one of the options below before starting your conversation:
        """)
        
        # Initial menu buttons
        with gr.Row():
            medicine_btn = gr.Button(
                "Medicine Information / Información de Medicamentos",
                elem_classes=["menu-button"]
            )
            store_btn = gr.Button(
                "Store Hours & Locations / Horarios y Ubicaciones",
                elem_classes=["menu-button"]
            )
            
        # Language selector
        language = gr.Radio(
            choices=["es", "en"],
            value="es",
            label="Language / Idioma",
            elem_classes=["language-selector"]
        )
        
        # Prefill chat with welcome message
        chatbot = gr.Chatbot(
            label="Chat",
            elem_classes=["chatbot"],
            value=[{"role": "assistant", "content": "¡Bienvenido a FarmaBot! Por favor, selecciona una opción:\n\n1. Información de Medicamentos - Para consultar disponibilidad, efectos secundarios y más\n2. Horarios y Ubicaciones - Para conocer nuestras tiendas"}],
            height=500,
            type="messages"
        )
        
        # Message input
        msg = gr.Textbox(
            label="Your message",
            placeholder="Type your message here...",
            lines=2,
            visible=True
        )
        
        # Submit button
        submit = gr.Button(
            "Send",
            elem_classes=["submit-button"],
            visible=True
        )
        
        # Simple file output for PDF downloads
        file_output = gr.File(visible=False)
        
        # Map HTML for locations
        map_html = gr.HTML("", visible=False)
        
        # Clear button
        clear = gr.Button("Clear Chat / Limpiar Chat")
        
        def handle_language_change(language: str) -> List[Dict[str, str]]:
            """Handle language change and show welcome message."""
            welcome_msg = {
                "es": "¡Bienvenido a FarmaBot! Por favor, selecciona una opción:\n\n1. Información de Medicamentos - Para consultar disponibilidad, efectos secundarios y más\n2. Horarios y Ubicaciones - Para conocer nuestras tiendas",
                "en": "Welcome to FarmaBot! Please select an option:\n\n1. Medicine Information - To check availability, side effects and more\n2. Store Hours & Locations - To learn about our stores"
            }
            return [{"role": "assistant", "content": welcome_msg[language]}]
        
        def handle_medicine_click(chat_history: List[Dict[str, str]], language: str) -> List[Dict[str, str]]:
            """Handle medicine information button click."""
            prompt = {
                "es": "Por favor, dime el nombre del medicamento que te interesa. Puedo buscar por nombre genérico o marca comercial.",
                "en": "Please tell me the name of the medicine you're interested in. I can search by generic name or brand name."
            }
            chat_history.append({"role": "assistant", "content": prompt[language]})
            return chat_history
        
        def handle_store_click(chat_history: List[Dict[str, str]], language: str) -> List[Dict[str, str]]:
            """Handle store information button click."""
            prompt = {
                "es": "¿Qué información necesitas sobre nuestras tiendas? Puedo ayudarte con:\n- Horarios de atención\n- Ubicaciones\n- Servicios disponibles",
                "en": "What information do you need about our stores? I can help you with:\n- Opening hours\n- Locations\n- Available services"
            }
            chat_history.append({"role": "assistant", "content": prompt[language]})
            return chat_history
        
        def handle_submit(user_message: str, chat_history: List[Dict[str, str]], language: str) -> Tuple[List[Dict[str, str]], Any, Any, Any]:
            """Handle message submission and generate response."""
            # If user attempts to message before selecting an option, prompt to use menu
            if len(chat_history) == 1:
                prompt = (
                    "Por favor, selecciona primero una opción: 'Información de Medicamentos' o 'Horarios y Ubicaciones'."
                    if language == 'es' else
                    "Please first select an option: 'Medicine Information' or 'Store Hours & Locations'."
                )
                chat_history.append({"role": "assistant", "content": prompt})
                return chat_history, gr.update(visible=False), gr.update(visible=False), None

            # If input is empty, do nothing
            if not user_message.strip():
                return chat_history, gr.update(visible=False), gr.update(visible=False), None

            # Log the user message and chat history
            chat_history.append({"role": "user", "content": user_message})
            
            # Call the bot's process_message method
            response = bot.process_message(user_message, language)
            
            # Check if this is an exit command
            if isinstance(response, str) and "<exit>" in response:
                clean_response = response.replace("<exit>", "")
                chat_history.append({"role": "assistant", "content": clean_response})
                return chat_history, gr.update(visible=False), gr.update(visible=False), None
            
            # Check if the response is a PDF path
            if isinstance(response, str) and response.endswith('.pdf'):
                confirm_msg = (
                    "¡Pedido confirmado! Tu cotización está lista para descargar." 
                    if language == "es" 
                    else "Order confirmed! Your quote is ready for download."
                )
                chat_history.append({"role": "assistant", "content": confirm_msg})
                return chat_history, gr.update(visible=False), gr.update(value=response, visible=True), None
            
            # Otherwise, append the bot's response to the chat history
            chat_history.append({"role": "assistant", "content": response})
            
            # If this is a location query, show interactive map
            if any(kw in user_message.lower() for kw in ["ubicaciones", "locations", "donde", "dónde"]):
                map_html_str = generate_map_html(language)
                return chat_history, gr.update(visible=True, value=map_html_str), gr.update(visible=False), None
            
            return chat_history, gr.update(visible=False), gr.update(visible=False), None
        
        def clear_history() -> List[Dict[str, str]]:
            """Clear the chat history."""
            return []
            
        def show_input_components():
            """Show the message input and submit button."""
            return gr.update(visible=True), gr.update(visible=True)
            
        # Set up event handlers
        language.change(
            handle_language_change,
            inputs=[language],
            outputs=[chatbot]
        )
        
        medicine_btn.click(
            handle_medicine_click,
            inputs=[chatbot, language],
            outputs=[chatbot]
        ).then(
            show_input_components,
            outputs=[msg, submit]
        )
        
        store_btn.click(
            handle_store_click,
            inputs=[chatbot, language],
            outputs=[chatbot]
        ).then(
            show_input_components,
            outputs=[msg, submit]
        )
        
        submit.click(
            handle_submit,
            inputs=[msg, chatbot, language],
            outputs=[chatbot, map_html, file_output]
        ).then(
            lambda: "",  # Clear message input
            outputs=[msg]
        )
        
        clear.click(
            clear_history,
            outputs=[chatbot]
        ).then(
            lambda: (gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)),  # Hide input components
            outputs=[msg, submit, map_html]
        )
        
        # Allow submitting with Enter key
        msg.submit(
            handle_submit,
            inputs=[msg, chatbot, language],
            outputs=[chatbot, map_html, file_output]
        ).then(
            lambda: "",  # Clear message input
            outputs=[msg]
        )
        
    return interface
    
def run_interface(bot: FarmaBot, share: bool = False) -> None:
    """Run the Gradio interface."""
    interface = create_interface(bot)
    interface.launch(share=share) 