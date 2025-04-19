"""
Language management for FarmaBot - Handles language detection and translation.
"""

from typing import Dict, Optional
import os
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

class LanguageManager:
    def __init__(self, model: str = "gpt-4-turbo"):
        """Initialize the language manager with a specific model."""
        # Initialize LLM based on model name
        if model.startswith("gemini"):
            google_api_key = os.getenv("GOOGLE_API_KEY")
            if not google_api_key:
                raise ValueError("GOOGLE_API_KEY is required for Gemini models but not found in environment.")
            # Convert system messages to human messages for Gemini models
            self.llm = ChatGoogleGenerativeAI(model=model, google_api_key=google_api_key, convert_system_message_to_human=True)
        else:
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError("OPENAI_API_KEY is required for non-Gemini models but not found in environment.")
            self.llm = ChatOpenAI(model=model, openai_api_key=openai_api_key)
            
        self.supported_languages = {"en": "English", "es": "Spanish"}
        self.default_language = "es"
        
    def detect_language(self, text: str) -> str:
        """Detect the language of the input text."""
        try:
            # For shorter and more reliable detection, use a rule-based approach first
            if self._check_language_indicators(text):
                return self._check_language_indicators(text)
                
            # Fall back to LLM detection for more complex cases
            messages = [
                SystemMessage(content="Detect the language of the following text. Respond with 'en' for English or 'es' for Spanish."),
                HumanMessage(content=text)
            ]
            response = self.llm.invoke(messages).content.strip().lower()
            return response if response in self.supported_languages else self.default_language
        except Exception as e:
            print(f"Error detecting language: {e}")
            return self.default_language
    
    def _check_language_indicators(self, text: str) -> Optional[str]:
        """Use rule-based approach to detect language based on common words."""
        text_lower = text.lower()
        
        # Common Spanish words and patterns
        es_indicators = ["que", "como", "para", "por", "con", "los", "las", "el", "la", 
                       "tienen", "donde", "cual", "cuanto", "medicina", "receta", "farmacia"]
        
        # Common English words and patterns
        en_indicators = ["the", "for", "with", "what", "how", "are", "is", "to", "where", 
                         "have", "which", "medicine", "prescription", "pharmacy"]
        
        # Count occurrences of indicator words
        es_count = sum(1 for word in es_indicators if f" {word} " in f" {text_lower} " or text_lower.startswith(f"{word} "))
        en_count = sum(1 for word in en_indicators if f" {word} " in f" {text_lower} " or text_lower.startswith(f"{word} "))
        
        # If there's a clear winner, return it
        if es_count > en_count:
            return "es"
        elif en_count > es_count:
            return "en"
        
        # No clear winner, return None to fall back to LLM detection
        return None
            
    def translate(self, text: str, target_language: str) -> str:
        """Translate text to the target language."""
        if target_language not in self.supported_languages:
            raise ValueError(f"Unsupported language: {target_language}")
            
        try:
            messages = [
                SystemMessage(content=f"Translate the following text to {self.supported_languages[target_language]}. Keep the meaning and tone exactly the same."),
                HumanMessage(content=text)
            ]
            return self.llm.invoke(messages).content
        except Exception as e:
            print(f"Error translating text: {e}")
            return text
            
    def get_language_specific_response(self, response_type: str, language: str) -> str:
        """Get language-specific responses for common queries."""
        responses = {
            "greeting": {
                "en": "Hello! How can I help you today?",
                "es": "¡Hola! ¿En qué puedo ayudarte hoy?"
            },
            "error": {
                "en": "I'm sorry, I had trouble processing your request.",
                "es": "Lo siento, tuve problemas al procesar tu solicitud."
            },
            "farewell": {
                "en": "Goodbye! Have a great day!",
                "es": "¡Adiós! ¡Que tengas un buen día!"
            }
        }
        
        return responses.get(response_type, {}).get(language, responses[response_type][self.default_language]) 