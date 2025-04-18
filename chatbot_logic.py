from logging_utils import log_interaction

def detect_language(text):
    es_indicators = ["que", "como", "para", "por", "con", "los", "las", "el", "la"]
    en_indicators = ["the", "for", "with", "what", "how", "are", "is", "to", "where"]
    text_lower = text.lower()
    es_count = sum(1 for word in es_indicators if f" {word} " in f" {text_lower} ")
    en_count = sum(1 for word in en_indicators if f" {word} " in f" {text_lower} ")
    return "es" if es_count >= en_count else "en"

def handle_error(error, context="general", language="es"):
    error_messages = {
        "db_connection": {
            "es": "No pude conectarme a la base de datos. Por favor, intenta más tarde.",
            "en": "I couldn't connect to the database. Please try again later."
        },
        "default": {
            "es": "Ocurrió un error inesperado. Por favor, intenta con otra pregunta.",
            "en": "An unexpected error occurred. Please try with another question."
        }
    }
    return error_messages.get(context, error_messages["default"]).get(language, error_messages["default"]["en"])