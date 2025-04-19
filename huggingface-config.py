"""
Hugging Face Spaces configuration for FarmaBot.
This file is required for proper configuration of the app in Hugging Face Spaces.
"""

# Configuration for Hugging Face Spaces
# You can set your API keys here or in environment variables in the Hugging Face Spaces interface

# LLM configuration
# Set at least one of these API keys in your Space settings under "Repository secrets"
# OPENAI_API_KEY: Required for using OpenAI models
# GOOGLE_API_KEY: Required for using Google's Gemini models (preferred)

# Default model configuration
DEFAULT_MODEL = "gemini-2.5-flash"  # Using Gemini 2.5 Flash for better performance

# If neither key is set in the environment, the app will attempt to load with a default
# but will show errors when querying about medications

# Database configuration
# The app uses SQLite which is automatically created if it doesn't exist
DB_CONNECTION_STRING = "sqlite:///farmabot.db"

# Gradio app configuration
APP_TITLE = "FarmaBot - Asistente Farmacéutico"
APP_DESCRIPTION = "Asistente virtual para consultas sobre medicamentos y farmacias"
APP_THEME = "soft"
ENABLE_QUEUE = True 