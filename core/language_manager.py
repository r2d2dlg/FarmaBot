"""
Language management for FarmaBot - Handles language detection and translation.
"""

from typing import Dict, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

class LanguageManager:
    def __init__(self, model: str = "gpt-4-turbo"):
        """Initialize the language manager with a specific model."""
        self.llm = ChatOpenAI(model=model)
        self.supported_languages = {"en": "English", "es": "Spanish"}
        self.default_language = "es"
        
    def detect_language(self, text: str) -> str:
        """Detect the language of the input text."""
        try:
            messages = [
                SystemMessage(content="Detect the language of the following text. Respond with 'en' for English or 'es' for Spanish."),
                HumanMessage(content=text)
            ]
            response = self.llm.invoke(messages).content.strip().lower()
            return response if response in self.supported_languages else self.default_language
        except Exception as e:
            print(f"Error detecting language: {e}")
            return self.default_language
            
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