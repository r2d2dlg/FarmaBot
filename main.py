"""
Main script to run FarmaBot.
"""

import os
from dotenv import load_dotenv
import logging

# Load environment variables FIRST, before other imports
load_dotenv()

from core.bot import FarmaBot
from services.database_service import DatabaseService
from services.medicine_service import MedicineService
from services.store_service import StoreService
from interface.gradio_interface import create_interface, run_interface

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('farmabot.log'),
            logging.StreamHandler()
        ]
    )

def main():
    """Initialize and run the chatbot."""
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Get environment variables
    db_connection_string = os.getenv("DB_CONNECTION_STRING")
    model_name = os.getenv("MODEL", "gpt-4-turbo") # Default to gpt-4-turbo if not set
    google_api_key = os.getenv("GOOGLE_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY") # Needed if using OpenAI

    if not db_connection_string:
        logger.error("DB_CONNECTION_STRING environment variable not set.")
        raise ValueError("Database connection string is missing.")

    # Validate API keys based on selected model
    if model_name.startswith("gemini") and not google_api_key:
        logger.error("MODEL is set to Gemini, but GOOGLE_API_KEY is missing in .env")
        raise ValueError("Google API Key is required for Gemini models.")
    elif not model_name.startswith("gemini") and not openai_api_key:
        logger.error("MODEL is not Gemini, but OPENAI_API_KEY is missing in .env")
        raise ValueError("OpenAI API Key is required for non-Gemini models.")

    try:
        # Initialize services
        logger.info("Initializing services...")
        # Extract db_path from connection string for SQLite
        db_path = db_connection_string.replace("sqlite:///", "")
        db_service = DatabaseService(db_path=db_path)
        medicine_service = MedicineService(db_service)
        store_service = StoreService(db_service)
        
        # Initialize bot with services
        logger.info(f"Initializing FarmaBot with model: {model_name}...")
        bot = FarmaBot(
            medicine_service=medicine_service, 
            store_service=store_service,
            model=model_name # Pass the model name here
        )
        
        # Create and run the interface
        logger.info("Starting Gradio interface...")
        interface = create_interface(bot)
        interface.launch(share=True)  # Set share=True to get a public URL
        
    except Exception as e:
        logger.error(f"Error running FarmaBot: {e}")
        raise

if __name__ == "__main__":
    main() 