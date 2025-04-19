# FarmaBot - Asistente Farmacéutico

FarmaBot es un asistente virtual basado en IA que proporciona información sobre medicamentos y farmacias. Ayuda a los usuarios a encontrar información sobre medicamentos, verificar si requieren receta, conocer sus efectos secundarios y ubicar farmacias cercanas.

## Características

- 🔍 **Búsqueda de medicamentos** por nombre genérico o marca comercial
- ⚠️ **Información sobre efectos secundarios** de medicamentos
- 💊 **Verificación de receta médica** para medicamentos controlados
- 🏥 **Localización de farmacias** y horarios de atención
- 📝 **Gestión de órdenes** y consultas de disponibilidad
- 🤖 **Potenciado por Gemini 2.5 Flash** para respuestas rápidas y precisas

## Cómo usar

1. Escribe tu consulta en el cuadro de texto y presiona Enter
2. Ejemplos de consultas:
   - "¿Tienen paracetamol?"
   - "¿Dónde está la farmacia más cercana?"
   - "¿Requiere receta el Lorazepam?"
   - "¿Cuáles son los efectos secundarios del omeprazol?"

## Configuración técnica

Esta aplicación utiliza:
- **Gradio**: Para la interfaz de usuario
- **LangChain**: Para la integración con modelos de lenguaje
- **SQLite**: Como base de datos para almacenar información de medicamentos y farmacias
- **Gemini 2.5 Flash**: Modelo de IA de Google para procesamiento de lenguaje natural

### Configurar API Keys

Para un funcionamiento completo, se requiere al menos una de las siguientes claves API:
- **Google API Key**: Para utilizar Gemini 2.5 Flash (recomendado)
- **OpenAI API Key**: Alternativa para utilizar modelos como GPT-3.5-Turbo o GPT-4

Estas claves deben configurarse en los ajustes de Hugging Face Spaces bajo "Repository secrets".

## Aviso importante

FarmaBot proporciona información general y no sustituye el consejo médico profesional. Siempre consulta a un profesional de la salud para recomendaciones médicas personalizadas. 