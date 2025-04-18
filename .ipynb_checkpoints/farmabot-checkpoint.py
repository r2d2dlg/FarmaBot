#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import re
import random
import traceback
import logging
import time # Although time wasn't used in the final snippets, it was in logging setup
from datetime import datetime
from dotenv import load_dotenv

import gradio as gr
import pandas as pd
import numpy as np # Although numpy wasn't used in final snippets, keep if needed for Chroma internals etc.
from sqlalchemy import create_engine

# LangChain Core Imports
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.callbacks import StdOutCallbackHandler # Used in one of the LLM initializations

# LangChain Community Imports
from langchain_community.utilities import SQLDatabase
from langchain_community.chat_message_histories import ChatMessageHistory # Used for Gradio state potentially

# LangChain OpenAI Imports
from langchain_openai import OpenAIEmbeddings, ChatOpenAI, OpenAI # OpenAI used for classification example

# LangChain Integrations / Main Package Imports
from langchain.docstore.document import Document
from langchain_chroma import Chroma
from langchain.agents import create_sql_agent
from langchain.chains import ConversationalRetrievalChain # Used in one example chain
from langchain.memory import ChatMessageHistory as LangchainChatMessageHistory # Used in one example


# In[2]:


# Load environment variables in a file called .env

load_dotenv(override=True)
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')


# In[3]:


# --- Database Connection Details ---
# Model & DB Configuration
MODEL = "gpt-4-turbo" # Or "gpt-4" as used in some later cells
DB_NAME = "Medicines" # This wasn't used directly, DATABASE_NAME is used
SERVER_NAME = "localhost"
DATABASE_NAME = "ChatbotFarmacia"
DRIVER = "ODBC Driver 17 for SQL Server"
SCHEMA_NAME = "dbo" # Used in SQLDatabase and document metadata
TABLE_NAME = "Medicines" # Used in document metadata
VECTOR_DB_PATH = "medicines_vectordb" # Renamed from db_name for clarity
LOG_FILE = 'farmabot_logs.log'

# Included tables for SQL Agent
INCLUDE_TABLES = [
    "Medicines", "inventory", "inventory_chorrera",
    "inventory_costa_del_este", "inventory_david", "inventory_el_dorado",
    "inventory_san_francisco", "Stores"
]

# Store names for keyword matching (ensure consistency with DB/inventory tables)
STORE_NAMES_ES = ["chorrera", "costa del este", "david", "el dorado", "san francisco"]
STORE_NAMES_EN = ["chorrera", "costa del este", "david", "el dorado", "san francisco"]



# In[4]:


# ==============================================================================
# Logging Setup
# ==============================================================================
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def log_interaction(query, response, query_type, duration_ms=0, error=None):
    """Log a single user interaction to file and console."""
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "query_type": query_type,
        "response_length": len(str(response)), # Ensure response is string for len()
        "duration_ms": duration_ms,
        "error": str(error) if error else None
    }
    log_message = f"INTERACTION: {log_data}"
    print(f"LOG: {log_message}") # Console log
    logging.info(log_message) # File log


# In[5]:


# Database Connection & Initial Data Load
# ==============================================================================
connection_string = f"mssql+pyodbc://{SERVER_NAME}/{DATABASE_NAME}?driver={DRIVER}"
engine = None
db = None
df_medicines = pd.DataFrame() # Use a more descriptive name
chunks = []
stores_df = pd.DataFrame() # For the get_store_count tool if kept

try:
    print(f"Attempting to connect to {DATABASE_NAME} on {SERVER_NAME}...")
    engine = create_engine(connection_string)

    # Test the connection
    with engine.connect() as connection:
        print(f"Successfully connected to database '{DATABASE_NAME}' on '{SERVER_NAME}'.")

    # Load initial Medicines data
    sql_query_medicines = f"SELECT * FROM [{SCHEMA_NAME}].[{TABLE_NAME}]"
    print(f"Loading medicine data...")
    df_medicines = pd.read_sql(sql_query_medicines, engine)
    print(f"Successfully loaded {len(df_medicines)} medicine rows.")

    # Split DataFrame into Chunks
    chunks = [df_medicines.iloc[i:i+5] for i in range(0, len(df_medicines), 5)]
    print(f"Medicine data split into {len(chunks)} chunks.")

    # Load Stores data (potentially for get_store_count tool)
    sql_query_stores = f"SELECT StoreID, StoreName, Location FROM {SCHEMA_NAME}.Stores"
    print("Loading stores data...")
    stores_df = pd.read_sql(sql_query_stores, engine)
    print(f"Successfully loaded {len(stores_df)} stores into DataFrame.")

    # Setup LangChain SQLDatabase interface
    db = SQLDatabase(engine=engine, schema=SCHEMA_NAME, include_tables=INCLUDE_TABLES)
    print("SQLDatabase interface created.")
    # Optional: print(db.get_table_info())

except Exception as e:
    print(f"FATAL ERROR: Error connecting to database or loading initial data: {e}")
    log_interaction(query="DB Setup", response="Error", query_type="db_setup_error", error=e)
    # Handle error appropriately - maybe exit or prevent chatbot launch
    engine = None
    db = None


# In[6]:


# Use one consistent LLM for the main chat and agent, unless specific temps are needed
llm = ChatOpenAI(model=MODEL, temperature=0.5) # Adjusted temperature slightly
embeddings = OpenAIEmbeddings()
print("LLM and Embeddings models initialized.")
# --- Updated SQL Agent Creation ---



# In[7]:


# ==============================================================================
# Vector Store Setup
# ==============================================================================
vectorstore = None
if not df_medicines.empty:
    docs = []
    print("Starting document conversion for vector store...")
    for i, chunk_df in enumerate(chunks):
        for index, row in chunk_df.iterrows():
            try:
                status_flag = int(row.get('Prescription', -1))
                status_text = "Requires Prescription" if status_flag == 1 else "Over-the-Counter" if status_flag == 0 else "Unknown"
                page_content = f"Medicine: {row['Generic Name']}\nUses: {row['Uses']}\nPrescription Status: {status_text}"
                metadata = {
                    "source_db_table": f"{SCHEMA_NAME}.{TABLE_NAME}",
                    "chunk_index": i,
                    "prescription_required_flag": status_flag,
                    "uses": row.get('Uses', ""),
                    "side_effects_common": row.get('Side Effects (Common)', ""),
                    "side_effects_rare": row.get('Side Effects (Rare)', ""),
                    "similar_drugs": row.get('Similar Drugs', ""),
                    "brand_name_1": row.get('Brand Name 1', ""),
                    # Add other relevant metadata fields...
                }
                docs.append(Document(page_content=page_content, metadata=metadata))
            except KeyError as e:
                print(f"KeyError processing row {index} for vector store: {e} - Check column names!")
            except Exception as e:
                 print(f"Error processing row {index} for vector store: {e}")
    print(f"Created {len(docs)} Document objects.")

    # Delete old vector store if exists
    if os.path.exists(VECTOR_DB_PATH):
        try:
            print(f"Attempting to delete existing vector store collection in '{VECTOR_DB_PATH}'...")
            Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embeddings).delete_collection()
            print(f"Deleted existing collection in '{VECTOR_DB_PATH}'.")
            # Consider removing the directory itself if needed: shutil.rmtree(VECTOR_DB_PATH)
        except Exception as e:
            print(f"Could not delete collection in '{VECTOR_DB_PATH}', may attempt overwrite: {e}")

    # Create new vector store
    try:
        print("Creating new vector store...")
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=VECTOR_DB_PATH
        )
        print(f"Vector store created with {vectorstore._collection.count()} documents in '{VECTOR_DB_PATH}'.")
    except Exception as e:
        print(f"FATAL ERROR: Could not create vector store: {e}")
        log_interaction(query="Vector Store Setup", response="Error", query_type="vectorstore_error", error=e)
        vectorstore = None # Ensure it's None if creation fails
else:
    print("WARNING: Medicine DataFrame is empty, skipping vector store creation.")

# Setup retriever (only if vectorstore was created)
retriever = None
if vectorstore:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5}) # Retrieve top 5 chunks
    print("Retriever created from vector store.")
else:
    print("WARNING: Vector store not available, RAG functionality will be disabled.")


# In[8]:


# ==============================================================================
# SQL Agent & Tools Setup
# ==============================================================================
sql_agent = None
# Define a placeholder for the tool even if agent creation fails initially
sql_query_tool_placeholder = None

if db and llm: # Ensure db and llm objects were created successfully
    try:
        # --- Pre-calculate table info to avoid complex f-string evaluation issues ---
        print("Fetching table info for SQL Agent prompt...")
        stores_table_info = db.get_table_info(['Stores'])
        medicines_table_info = db.get_table_info(['Medicines'])
        print("Table info fetched successfully.")

        # --- Define the agent prefix using the pre-calculated info ---
        agent_prefix = f"""You are an expert SQL agent for a pharmacy system.

        You have access to tables including 'Stores' (schema: {stores_table_info}) which contains store location information,
        and various 'inventory_...' tables (like 'inventory_chorrera', 'inventory_costa_del_este', etc.) containing stock levels for medicines.
        The primary medicine information is in the 'Medicines' table (schema: {medicines_table_info}).

        When asked about stores, store counts, or general locations, query the 'Stores' table.
        When asked "how many stores", run 'SELECT COUNT(*) FROM {SCHEMA_NAME}.Stores'.

        IMPORTANT: When asked about inventory, stock, quantity, or availability for a specific medicine:
        1. Identify the medicine name precisely.
        2. Identify the specific store location if mentioned (e.g., 'Chorrera', 'Costa del Este'). Try to map these common names to the relevant inventory table (like '{SCHEMA_NAME}.inventory_chorrera') or filter based on store name if querying a unified inventory table.
        3. Query the appropriate table(s) to find the current quantity for that medicine at that location (or overall if no location is specified).
        4. Return the quantity clearly as a number if possible. If reporting for a specific store, mention it. Example response: "There are 48 units of Tylenol in stock at the Costa del Este branch."

        Always check the schema carefully before answering and provide clear, concise responses based ONLY on the database information.
        Do not make up information. If you cannot find the information, say so clearly.
        """
        # --- Create the SQL Agent ---
        print("Creating SQL Agent...")
        sql_agent = create_sql_agent(
            llm=llm,
            db=db, # Pass the db object itself here
            agent_type="openai-tools",
            verbose=True,
            prefix=agent_prefix # Use the pre-formatted prefix
        )
        print("SQL Agent created successfully.")

    except Exception as e:
        print(f"ERROR: Could not create SQL Agent: {e}")
        # Log the full traceback for better debugging if the issue persists
        import traceback
        print("Traceback:")
        print(traceback.format_exc())
        log_interaction(query="SQL Agent Setup", response="Error", query_type="sql_agent_error", error=traceback.format_exc()) # Log full traceback
        sql_agent = None # Ensure agent is None if creation fails
else:
    print("WARNING: SQL Database interface or LLM not available, SQL Agent cannot be created.")

# --- Define SQL Query Tool ---
# Define the tool function regardless of whether the agent was created successfully.
# It will check for the agent's existence internally when called.
@tool
def sql_query(query: str) -> str:
    """Execute a SQL query against the store and inventory database. Use this for questions about store locations, store counts, and specific medicine inventory/stock levels."""
    if not sql_agent:
         print("WARNING: sql_query tool called, but SQL Agent is not available.")
         # Provide a user-friendly message if the agent isn't ready
         # Consider checking language preference here if possible, otherwise default
         return "Lo siento, no puedo acceder a la base de datos de la tienda en este momento. / Sorry, I cannot access the store database right now."
    try:
        print(f"DEBUG: Sending query to SQL Agent: {query}") # Debug print
        result = sql_agent.invoke({"input": query})
        # Handle potential variations in agent output structure
        output = result.get("output", result.get("result", "No specific output found."))
        print(f"DEBUG: Received output from SQL Agent: {output}") # Debug print
        return output
    except Exception as e:
        print(f"Error invoking SQL agent for query '{query}': {e}")
        import traceback
        log_interaction(query=f"SQL Agent Query: {query}", response="Error", query_type="sql_agent_query_error", error=traceback.format_exc())
        # Provide a user-friendly error message
        return f"Error al consultar la base de datos. / Error querying database."

# Assign the function to the placeholder variable (optional, but can be useful)
sql_query_tool_placeholder = sql_query

# (Optional) Define get_store_count tool - Commented out as likely redundant
# @tool
# def get_store_count() -> str:
#     # ... (implementation) ...

def chat(message_list):
    """
    Handles messages, routes requests, respects language, focuses on pharmacy topics,
    handles inventory/orders, attempts context carry-over, translates SQL responses,
    and includes enhanced debugging for AttributeError.
    """
    print("DEBUG: Received message list:", message_list)

    # --- Outer Try-Except for general errors ---
    try:
        if not message_list:
            return "Say something!"

        # --- Language Detection & Last Messages ---
        language = "es" # Default
        # Ensure message_list[0] exists and is a dictionary before accessing
        if message_list and isinstance(message_list[0], dict) and \
           message_list[0].get('role') == 'system' and 'User prefers:' in message_list[0].get('content', ''):
             language = "en" if "en" in message_list[0].get('content') else "es"

        user_content = ""
        # Ensure user_content is always a string
        if message_list and isinstance(message_list[-1], dict) and message_list[-1].get('role') == 'user':
            content_val = message_list[-1].get('content', '')
            if isinstance(content_val, str):
                 user_content = content_val
            else:
                 print(f"WARNING: User message content is not a string: {type(content_val)}. Converting.")
                 user_content = str(content_val)

        last_bot_message = ""
        if len(message_list) > 1 and isinstance(message_list[-2], dict) and message_list[-2].get('role') == 'assistant':
             bot_content_val = message_list[-2].get('content', '')
             if isinstance(bot_content_val, str):
                  last_bot_message = bot_content_val
             else:
                  print(f"WARNING: Bot message content is not a string: {type(bot_content_val)}. Converting.")
                  last_bot_message = str(bot_content_val)

        if not user_content:
            return "Por favor, dime algo." if language == "es" else "Please say something."

        # --- Keyword Lists ---
        # Define locally or ensure global definitions are accessible
        # Using global lists defined outside the function is assumed here
        location_keywords_es = ["tienda", "tiendas", "farmacia", "farmacias", "sucursal", "sucursales", "ubicación", "ubicaciones", "cuántas", "donde", "dónde", "local", "locales"] + STORE_NAMES_ES
        location_keywords_en = ["store", "stores", "pharmacy", "pharmacies", "location", "locations", "how many", "where", "branch", "branches"] + STORE_NAMES_EN
        hours_keywords_es = ["hora", "horas", "horario", "horarios", "abierto", "cierra", "abre", "disponible", "atención", "atienden", "cuando"]
        hours_keywords_en = ["hour", "hours", "schedule", "open", "close", "opens", "closes", "availability", "available", "when", "time", "timing"]
        medication_keywords_es = ["medicamento", "medicina", "medicinas", "droga", "drogas", "pastilla", "efecto", "efectos", "secundarios", "alternativa", "dosis", "uso", "usos", "para qué sirve", "receta"] + known_medicines
        medication_keywords_en = ["medication", "medicine", "drug", "drugs", "pill", "effect", "effects", "side effect", "alternative", "dose", "use", "uses", "what is it for", "prescription"] + known_medicines
        inventory_keywords_es = ["inventario", "stock", "cantidad", "cuantos", "cuántos", "disponible", "hay", "tienen"]
        inventory_keywords_en = ["inventory", "stock", "quantity", "how many", "available", "are there", "do you have", "have"]
        order_keywords_es = ["ordenar", "pedir", "comprar", "quiero", "necesito"]
        order_keywords_en = ["order", "buy", "purchase", "i want", "i need"]
        thanks_keywords_es = ["gracias", "muchas gracias", "te lo agradezco", "agradecido", "gracie"]
        thanks_keywords_en = ["thank", "thanks", "thank you", "appreciated", "grateful", "thx"]


        # --- Intent Detection & Context Handling ---
        medicine_context = None
        location_context = None
        intent = "unknown"

        try: # Specific try block for intent/context section
            user_content_lower = user_content.lower() # Should be safe now
            is_thanks = any(keyword in user_content_lower for keyword in (thanks_keywords_es if language == "es" else thanks_keywords_en))
            is_hours = any(keyword in user_content_lower for keyword in (hours_keywords_es if language == "es" else hours_keywords_en))

            # --- Helper Functions ---
            # Ensure these access the globally defined lists correctly
            def extract_medicine(text):
                if not isinstance(text, str): return None
                text_lower = text.lower()
                for med in known_medicines: # Uses global known_medicines
                    if med in text_lower: return med
                return None

            def extract_location(text):
                if not isinstance(text, str): return None
                text_lower = text.lower()
                for loc in STORE_NAMES_ES: # Uses global STORE_NAMES_ES
                    if loc in text_lower: return loc
                return None
            # --- End Helper Functions ---

            current_medicine = extract_medicine(user_content)
            current_location = extract_location(user_content)

            # Context Recovery Logic
            if not current_medicine and len(message_list) >= 3:
                current_mentions_inventory = any(kw in user_content_lower for kw in inventory_keywords_es + inventory_keywords_en + order_keywords_es + order_keywords_en)
                if current_location and current_mentions_inventory:
                    prev_user_content = message_list[-3].get('content', '')
                    if isinstance(prev_user_content, str):
                        medicine_context = extract_medicine(prev_user_content)
                        if medicine_context:
                            print(f"DEBUG: Context Recovery: Using medicine '{medicine_context}' from previous turn.")
                            current_medicine = medicine_context
                    else:
                        print(f"WARNING: Previous user message content was not a string: {type(prev_user_content)}")


            # Determine Intent (Simplified)
            is_inventory_kw = any(kw in user_content_lower for kw in inventory_keywords_es + inventory_keywords_en)
            is_order_kw = any(kw in user_content_lower for kw in order_keywords_es + order_keywords_en)
            is_location_kw = any(kw in user_content_lower for kw in location_keywords_es + location_keywords_en)
            is_medication_kw = any(kw in user_content_lower for kw in medication_keywords_es + medication_keywords_en)

            if current_medicine and is_order_kw: intent = "order_request"
            elif current_medicine and is_inventory_kw: intent = "inventory_check"
            elif is_hours: intent = "hours_info"
            elif is_location_kw and not current_medicine: intent = "location_info"
            elif is_medication_kw: intent = "medication_info"
            else: # Fallback context check
                if current_location and not current_medicine and len(message_list) >=3:
                     prev_bot_msg_lower = last_bot_message.lower() # Safe now
                     if 'en stock' in prev_bot_msg_lower or 'in stock' in prev_bot_msg_lower:
                         intent = "inventory_check"
                         if not medicine_context:
                              prev_user_content = message_list[-3].get('content', '')
                              if isinstance(prev_user_content, str):
                                   medicine_context = extract_medicine(prev_user_content)
                                   if medicine_context: current_medicine = medicine_context
                              else:
                                   print(f"WARNING: Previous user message content was not a string during fallback: {type(prev_user_content)}")

            if intent == "unknown": intent = "off_topic"
            print(f"DEBUG: Intent={intent}, Medicine={current_medicine}, Location={current_location}")

        except AttributeError as ae:
            print(f"ERROR: AttributeError during Intent/Context processing: {ae}")
            print(traceback.format_exc())
            log_interaction(query=user_content, response="AttributeError during intent processing", query_type="intent_attr_error", error=ae)
            raise # Re-raise error
        except NameError as ne: # Specifically catch NameError here
             print(f"ERROR: NameError during Intent/Context processing - likely missing global list (known_medicines or STORE_NAMES_ES?): {ne}")
             print(traceback.format_exc())
             log_interaction(query=user_content, response="NameError during intent processing", query_type="intent_name_error", error=ne)
             raise # Re-raise error


        # --- ROUTING BASED ON DETERMINED INTENT ---
        # (The rest of the routing logic remains the same as the previous version)
        # ... (Handle Thanks) ...
        # ... (Handle Order Follow-up) ...
        # ... (Handle Inventory/Order) ...
        # ... (Handle Location Info) ...
        # ... (Handle Hours) ...
        # ... (Handle RAG Medication Info - with its own debugging) ...
        # ... (Handle Off-Topic) ...
        # ... (Fallback Warning) ...

        # Placeholder for the rest of the routing logic from the previous version
        # Ensure each path returns a response string
        if intent == "hours_info":
             response = "Todas nuestras farmacias están abiertas de 5 AM a 10 PM..." if language == "es" else "All our pharmacies are open from 5 AM to 10 PM..."
             log_interaction(query=user_content, response=response, query_type="hours_fixed", duration_ms=0)
             return response
        # ... Add other intent handling blocks here, ensuring they return ...

        # Make sure the RAG block is here
        if intent == "medication_info":
            print(f"DEBUG: Routing to RAG system...")
            if not retriever:
                 print("ERROR: Retriever is not available.")
                 return "Lo siento, no puedo buscar información sobre medicamentos en este momento." if language == "es" else "Sorry, I cannot look up medication information right now."
            try: # Specific try for RAG block
                # ... (Detailed RAG logic with debugging from previous step) ...
                 if isinstance(response_content, str):
                    response = response_content
                    # ... (log and return response) ...
                    log_interaction(query=user_content, response=response, query_type="medicine_rag", duration_ms=0)
                    return response
                 else:
                    # ... (handle error) ...
                    raise TypeError(f"Unexpected LLM response format. Expected string content, got {type(response_content)}")
            # ... (except blocks for RAG) ...
            except AttributeError as ae: # Catch AttributeError specifically within RAG
                 print(f"ERROR: AttributeError during RAG processing: {ae}")
                 print(traceback.format_exc())
                 log_interaction(query=user_content, response="AttributeError during RAG processing", query_type="rag_attr_error", error=ae)
                 raise # Re-raise to be caught by outer handler
            except Exception as e: # Catch other errors during RAG
                 error_msg = "Lo siento, tuve un problema al buscar información sobre ese medicamento." if language == "es" else "I'm sorry, I had trouble finding information about that medication."
                 print(f"ERROR: Exception during RAG processing: {str(e)}")
                 print(traceback.format_exc())
                 log_interaction(query=user_content, response=error_msg, query_type="medicine_rag_error", duration_ms=0, error=str(e))
                 return error_msg # Return specific RAG error message


        # Default/Off-topic handling if no other intent matched or returned
        print(f"DEBUG: Handling as off-topic or fallback.")
        response = "Lo siento, solo puedo ayudarte con preguntas sobre..." if language == "es" else "Sorry, I can only help with questions about..." # Truncated
        log_interaction(query=user_content, response=response, query_type="off_topic", duration_ms=0)
        return response


    # --- Outer Exception Handler ---
    except Exception as e:
        print(f"ERROR: Uncaught exception in chat function: {e}")
        print("Traceback:")
        print(traceback.format_exc()) # Print detailed traceback
        log_interaction(query=user_content, response="Generic Error Fallback", query_type="chat_uncaught_error", error=traceback.format_exc()) # Log full traceback
        # Return the user-facing message, including the specific error string
        return f"I encountered an error: {str(e)}. Please try rephrasing your question."


# In[9]:


def debug_message_structure(message_list):
    """
    Función de diagnóstico que muestra la estructura completa del message_list
    """
    import json
    print("==== DEBUG: MESSAGE STRUCTURE ====")
    print(f"Type of message_list: {type(message_list)}")
    print(f"Length of message_list: {len(message_list) if isinstance(message_list, list) else 'Not a list'}")
    
    # Intenta serializar a JSON para una visualización fácil
    try:
        print(f"Content: {json.dumps(message_list, indent=2)}")
    except:
        # Si no se puede serializar, muestra elemento por elemento
        if isinstance(message_list, list):
            for i, item in enumerate(message_list):
                print(f"Item {i}:")
                print(f"  Type: {type(item)}")
                print(f"  Content: {str(item)[:100]}...")
        else:
            print(f"Raw content: {str(message_list)[:100]}...")
    
    print("==== END DEBUG ====")
    
    # Retorna un mensaje genérico para facilitar la depuración
    return "Diagnóstico completado. Por favor revisa los logs para ver la estructura del mensaje."

def minimal_chat(message_list):
    """
    Versión mínima de la función chat que solo maneja consultas sobre fiebre.
    Elimina la mayoría de la complejidad para enfocarse en resolver el problema principal.
    """
    import traceback
    
    try:
        # Diagnóstico de la estructura del mensaje
        debug_message_structure(message_list)
        
        # Retorna un mensaje fijo para pruebas
        RESPUESTA_FIEBRE = """
        Para la fiebre, tenemos los siguientes medicamentos:
        
        1. Paracetamol (Acetaminofén) - Tylenol, Tempra
           - Reduce la fiebre y alivia el dolor
           - Disponible en tabletas, jarabe y gotas
        
        2. Ibuprofeno - Advil, Motrin
           - Antiinflamatorio que reduce la fiebre y el dolor
           - Disponible en tabletas y suspensión
        
        3. Naproxeno - Aleve
           - Antiinflamatorio de acción prolongada
           - Para adultos
        
        4. Ácido Acetilsalicílico - Aspirina
           - Reduce la fiebre y la inflamación
           - Solo para adultos, no recomendado para niños
        
        ¿Deseas información más específica sobre alguno de estos medicamentos?
        """
        
        return RESPUESTA_FIEBRE

    except Exception as e:
        print(f"ERROR EN MINIMAL_CHAT: {str(e)}")
        print(traceback.format_exc())
        return f"Error diagnosticado: {str(e)}"

def chat(message_list):
    """
    Función chat modificada que intercepta el comando directamente para probar
    una respuesta fija sin procesar el message_list.
    """
    try:
        # Si no hay mensaje, devolver mensaje genérico
        if not message_list:
            return "Por favor, dime algo."
            
        # Intentar extraer el mensaje del usuario
        user_message = ""
        try:
            # Buscar el mensaje del usuario en la lista
            if isinstance(message_list, list):
                for msg in reversed(message_list):
                    if isinstance(msg, dict) and msg.get('role') == 'user':
                        user_content = msg.get('content', '')
                        if isinstance(user_content, str):
                            user_message = user_content
                            break
        except Exception as e:
            print(f"Error al extraer mensaje del usuario: {str(e)}")
        
        # Si encontramos el mensaje y contiene "fiebre"
        user_message_lower = user_message.lower() if isinstance(user_message, str) else ""
        
        # Responder a consultas sobre fiebre ignorando todo el procesamiento complejo
        if "fiebre" in user_message_lower:
            return minimal_chat(message_list)
        else:
            # Para otros mensajes, enviar respuesta genérica
            return f"Recibí tu mensaje. Para consultas sobre medicamentos específicos como los de fiebre, por favor menciona el síntoma o tipo de medicamento que buscas."
            
    except Exception as e:
        import traceback
        print(f"ERROR EN CHAT PRINCIPAL: {str(e)}")
        print(traceback.format_exc())
        return f"Ocurrió un error al procesar tu mensaje: {str(e)}. Por favor, intenta con otra pregunta."

# Implementación de chat alternativa si la principal falla
def fallback_chat(message_list):
    """
    Implementación alternativa extremadamente simple para garantizar alguna respuesta
    """
    try:
        # Mensaje fijo para solución inmediata
        return """
        Para la fiebre, tenemos varios medicamentos como Paracetamol (Tylenol), 
        Ibuprofeno (Advil), Naproxeno (Aleve) y Aspirina. 
        Todos están disponibles en nuestras farmacias.
        """
    except:
        return "Lo siento, estamos experimentando dificultades técnicas. Por favor, intenta más tarde."


# In[10]:


# Simple LangChain implementation without typing

# Session storage for maintaining conversation history
session_histories = {}

# Define the models/LLMs
# Assuming MODEL is defined elsewhere in your code
llm = ChatOpenAI(model=MODEL, temperature=0.7)
agent_llm = ChatOpenAI(model=MODEL, temperature=0)

# Load the vectorstore
if 'vectorstore' not in locals() or vectorstore is None:
    print("Loading existing vectorstore from disk...")
    embeddings = OpenAIEmbeddings()
    try:
        vectorstore = Chroma(persist_directory=db_name, embedding_function=embeddings)
        print(f"Loaded vectorstore with {vectorstore._collection.count()} documents")
    except Exception as e:
        print(f"Error loading vectorstore: {e}")

# Set up the retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Function to get or create session history
def get_session_history(session_id):
    if session_id not in session_histories:
        session_histories[session_id] = ChatMessageHistory()
    return session_histories[session_id]

# Create a RAG chain
def create_rag_chain():
    # Define the prompt template
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are an expert pharmaceutical assistant. Use the following context to answer the question.
        
If you're asked about side effects, focus on the information in the 'Side Effects (Common)' and 'Side Effects (Rare)' fields.
If you're asked about stores or inventory, explain that this information needs to be queried from the database.
Answer the question based only on the provided context. If the information isn't available, say so clearly."""),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessage(content="{question}"),
        SystemMessage(content="Context: {context}")
    ])
    
    # Create the RAG chain
    return (
        {"context": retriever, "question": RunnablePassthrough(), "chat_history": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

# Create the RAG chain
rag_chain = create_rag_chain()

# Define the SQL query tool
@tool
def sql_query(query):
    """Execute a SQL query against the store and inventory database"""
    try:
        # Assuming sql_agent is defined elsewhere
        return sql_agent.invoke({"input": query})["output"]
    except Exception as e:
        return f"Error querying database: {str(e)}"

# Vector search function for direct access to RAG
def vector_search(query, session_id="default"):
    try:
        # Get history and pass it explicitly
        history = get_session_history(session_id)
        result = rag_chain.invoke({"question": query, "chat_history": history.messages})
        
        # Record the exchange
        history.add_user_message(query)
        history.add_ai_message(result)
        return result
    except Exception as e:
        print(f"Error in vector_search: {e}")
        # Fallback to basic query without history
        return rag_chain.invoke({"question": query, "chat_history": []})

# Simple agent without LangGraph
def simple_agent(query):
    """A simple agent implementation that doesn't use LangGraph"""
    # Create a prompt for the agent
    agent_prompt = f"""You are a helpful assistant that can answer questions about medicines and store inventory.

Question: {query}

If this is about store inventory, locations, or similar store-related information, use the SQL database.
If this is about medicine properties, side effects, or drug information, use the medicine information database.
Otherwise, answer directly.

Respond with your final answer."""

    # Get a response from the LLM
    response = agent_llm.invoke(agent_prompt)
    
    # Extract the content
    if hasattr(response, "content"):
        return response.content
    else:
        return str(response)

# The integrated chat function
def chat(question, session_id="default"):
    try:
        # For store-related questions, directly route to SQL agent
        if any(keyword in question.lower() for keyword in ["store", "stores", "location", "locations", "inventory", "stock", "how many"]):
            try:
                print(f"Routing to SQL agent: {question}")
                result = sql_query(question)
                
                # Record the exchange in history
                history = get_session_history(session_id)
                history.add_user_message(question)
                history.add_ai_message(result)
                return result
            except Exception as e:
                print(f"SQL direct routing failed: {e}, falling back to simple agent")
        
        # For medicine-related questions about side effects, use RAG
        if any(keyword in question.lower() for keyword in ["side effect", "medicine", "drug", "medication"]):
            try:
                print(f"Routing to RAG chain: {question}")
                return vector_search(question, session_id)
            except Exception as e:
                print(f"RAG chain failed: {e}, falling back to simple agent")
        
        # For general questions, use the simple agent
        try:
            print(f"Using simple agent: {question}")
            result = simple_agent(question)
            
            # Record the exchange in history
            history = get_session_history(session_id)
            history.add_user_message(question)
            history.add_ai_message(result)
            
            return result
        except Exception as e:
            print(f"Agent failed: {e}, falling back to direct LLM")
            try:
                response = llm.invoke(question)
                content = response.content if hasattr(response, "content") else str(response)
                return content
            except Exception as llm_err:
                return f"I encountered several errors processing your request. Please try rephrasing your question. Error: {str(llm_err)}"
                
    except Exception as e:
        return f"I encountered an error: {str(e)}. Please try rephrasing your question."


# In[11]:


# Optionally, view the first chunk
print(len(chunks))


# In[12]:


# Let's investigate the vectors

collection = vectorstore._collection
count = collection.count()

sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
dimensions = len(sample_embedding)
print(f"There are {count:,} vectors with {dimensions:,} dimensions in the vector store")


# In[ ]:


def create_consolidated_interface():
    """Creates the Gradio interface with language selection"""
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Farma AI Panama")
        gr.Markdown("Asistente virtual para consultas sobre farmacias, horarios, medicamentos e inventario.")

        language_state = gr.State("asking")
        initial_greeting = "¡Hola! Soy el asistente virtual de Farma AI Panama. Puedo ayudarte con información sobre nuestras ubicaciones, horarios (5 AM a 10 PM), medicamentos e inventario. ¿Prefieres hablar en español o en inglés? / Hello! I'm Farma AI Panama's virtual assistant. I can help with info about our locations, hours (5 AM to 10 PM), medications, and inventory. Do you prefer Spanish or English?"

        chatbot = gr.Chatbot(
            height=500,
            type="messages",
            value=[{"role": "assistant", "content": initial_greeting}]
        )

        with gr.Row():
            msg = gr.Textbox(placeholder="Escribe aquí...", show_label=False, scale=4)
            submit_button = gr.Button("Enviar", variant="primary", scale=1)

        clear_button = gr.Button("Borrar Conversación")

        def handle_submit(user_message, chat_history_list, current_language):
            """Handles message submission, language selection, and calls the main chat logic."""
            if not user_message.strip():
                return "", chat_history_list, current_language

            # Append user message dictionary
            chat_history_list.append({"role": "user", "content": user_message})

            response_text = ""
            new_language = current_language

            if current_language == "asking":
                lower_msg = user_message.lower()
                if any(word in lower_msg for word in ["español", "espanol", "spanish"]):
                    new_language = "es"
                    response_text = "Has seleccionado español. ¿En qué puedo ayudarte hoy?"
                elif any(word in lower_msg for word in ["english", "inglés", "ingles"]):
                    new_language = "en"
                    response_text = "You've selected English. How can I help you today?"
                else:
                    response_text = "No entendí tu preferencia. Por favor indica 'español' o 'inglés' / I didn't catch your preference. Please indicate 'Spanish' or 'English'."
                    new_language = "asking" # Remain in asking state
            else:
                # Regular chat processing
                # Prepare message list for the chat function (add system prompt)
                system_prompt = {"role": "system", "content": f"User prefers: {new_language}. Respond in {'Spanish' if new_language == 'es' else 'English'}."}
                messages_for_chat_func = [system_prompt] + chat_history_list

                try:
                    # Call the main chat function (ensure it's accessible)
                    response_text = chat(messages_for_chat_func)
                except Exception as e:
                    import traceback
                    error_msg = f"Error in chat function: {str(e)}"
                    print(error_msg)
                    print(traceback.format_exc())
                    response_text = "Lo siento, ocurrió un error interno." if new_language == "es" else "Sorry, an internal error occurred."
                    log_interaction(query=user_message, response=response_text, query_type="chat_func_error", error=e)


            # Append assistant response dictionary
            chat_history_list.append({"role": "assistant", "content": response_text})

            # Log interaction (console only here, file logging is in chat func)
            print(f"User: {user_message}")
            print(f"Assistant ({new_language}): {response_text}")

            # Return empty string for textbox, updated history, and potentially updated language
            return "", chat_history_list, new_language

        def clear_history_func():
            """Clears the chat and resets language state."""
            return "", [{"role": "assistant", "content": initial_greeting}], "asking"

        # Event handlers
        submit_button.click(
            fn=handle_submit,
            inputs=[msg, chatbot, language_state],
            outputs=[msg, chatbot, language_state]
        )
        msg.submit(
            fn=handle_submit,
            inputs=[msg, chatbot, language_state],
            outputs=[msg, chatbot, language_state]
        )
        clear_button.click(
            fn=clear_history_func,
            inputs=[],
            outputs=[msg, chatbot, language_state]
        )

    return demo

# ==============================================================================
# Main Execution Block
# ==============================================================================
if __name__ == "__main__":
    # Ensure critical components are available before launching UI
    if engine and db and llm and vectorstore and retriever and sql_agent:
        print("All components initialized successfully. Launching Gradio interface...")
        interface = create_consolidated_interface()
        interface.launch(debug=True, share=False) # share=True can expose it publicly
    else:
        print("ERROR: Could not initialize all required components. Aborting launch.")
        log_interaction(query="Launch Check", response="Aborted", query_type="launch_error", error="Missing components")


