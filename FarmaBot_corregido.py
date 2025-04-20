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
    """Enhanced logging with more metrics"""
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "query_type": query_type,
        "response_length": len(str(response)),
        "duration_ms": duration_ms,
        "error": str(error) if error else None,
        "detected_language": detect_language(query),
        "query_tokens": len(query.split()),
        "response_tokens": len(str(response).split()),
        "session_id": get_current_session_id()
    }
    
    log_message = f"INTERACTION: {json.dumps(log_data)}"
    print(f"LOG: {log_message}")
    logging.info(log_message)

def detect_language(text):
    """Simple language detection based on common words"""
    es_indicators = ["que", "como", "para", "por", "con", "los", "las", "el", "la"]
    en_indicators = ["the", "for", "with", "what", "how", "are", "is", "to", "where"]
    
    text_lower = text.lower()
    es_count = sum(1 for word in es_indicators if f" {word} " in f" {text_lower} ")
    en_count = sum(1 for word in en_indicators if f" {word} " in f" {text_lower} ")
    
    return "es" if es_count >= en_count else "en"

def get_current_session_id():
    """Get current session ID from context or return default"""
    # In a real implementation, this would get the session from your context
    # For now, we'll just return a placeholder
    return "default_session"


# In[5]:


def handle_error(error, context="general", language="es"):
    """Centralized error handling with consistent logging and user-friendly messages"""
    error_str = str(error)
    traceback.print_exc()
    
    # Log with consistent format
    log_interaction(f"Error in {context}", error_str, f"error_{context}", error=error)
    
    # Return appropriate user message based on context and language
    error_messages = {
        "db_connection": {
            "es": "No pude conectarme a la base de datos. Por favor, intenta más tarde.",
            "en": "I couldn't connect to the database. Please try again later."
        },
        "rag_query": {
            "es": "Tuve problemas buscando información sobre ese medicamento.",
            "en": "I had trouble finding information about that medication."
        },
        # Add more context-specific messages
    }
    
    default_msg = {
        "es": "Ocurrió un error inesperado. Por favor, intenta con otra pregunta.",
        "en": "An unexpected error occurred. Please try with another question."
    }
    
    return error_messages.get(context, default_msg).get(language, default_msg["en"])


# In[6]:


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


# In[7]:


# Use one consistent LLM for the main chat and agent, unless specific temps are needed
llm = ChatOpenAI(model=MODEL, temperature=0.5) # Adjusted temperature slightly
embeddings = OpenAIEmbeddings()
print("LLM and Embeddings models initialized.")
# --- Updated SQL Agent Creation ---



# In[8]:


# ==============================================================================
# Vector Store Setup con enriquecimiento de síntomas comunes
# ==============================================================================
vectorstore = None
if not df_medicines.empty:
    docs = []

    sintomas_comunes = [
        "fiebre", "dolor de garganta", "dolor muscular", "congestión",
        "dolor de cabeza", "dolor de espalda", "malestar general",
        "tos", "resfriado", "gripe", "dolor corporal"
    ]

    sintoma_traducciones = {
        "fiebre": "fever",
        "dolor de garganta": "sore throat",
        "dolor muscular": "muscle pain",
        "congestión": "congestion",
        "dolor de cabeza": "headache",
        "dolor de espalda": "back pain",
        "malestar general": "general discomfort",
        "tos": "cough",
        "resfriado": "cold",
        "gripe": "flu",
        "dolor corporal": "body ache"
    }

    print("Starting document conversion with symptom enrichment...")
    for i, chunk_df in enumerate([df_medicines.iloc[x:x+5] for x in range(0, len(df_medicines), 5)]):
        for index, row in chunk_df.iterrows():
            try:
                uses_text = str(row.get("Uses", "")).lower()
                found_symptoms = [s for s in sintomas_comunes if s in uses_text]

                enriched_en = ""
                enriched_es = ""
                if found_symptoms:
                    enriched_en = "This medicine may help relieve symptoms such as " + ", ".join([sintoma_traducciones[s] for s in found_symptoms]) + "."
                    enriched_es = "Este medicamento puede ayudar a aliviar síntomas como " + ", ".join(found_symptoms) + "."

                status_flag = int(row.get('Prescription', -1))
                status_text = "Requires Prescription" if status_flag == 1 else "Over-the-Counter" if status_flag == 0 else "Unknown"

                page_content = f"Medicine: {row['Generic Name']}\nUses: {row['Uses']}\nPrescription Status: {status_text}"
                if enriched_en and enriched_es:
                    page_content += f"\n\n{enriched_en}\n{enriched_es}"

                metadata = {
                    "source_db_table": f"{SCHEMA_NAME}.{TABLE_NAME}",
                    "chunk_index": i,
                    "prescription_required_flag": status_flag,
                    "uses": row.get('Uses', ""),
                    "side_effects_common": row.get('Side Effects (Common)', ""),
                    "side_effects_rare": row.get('Side Effects (Rare)', ""),
                    "similar_drugs": row.get('Similar Drugs', ""),
                    "brand_name_1": row.get('Brand Name 1', "")
                }

                docs.append(Document(page_content=page_content, metadata=metadata))
            except Exception as e:
                print(f"Error enriching document at row {index}: {e}")

    print(f"Created {len(docs)} Document objects.")

    if os.path.exists(VECTOR_DB_PATH):
        try:
            print(f"Attempting to delete existing vector store in '{VECTOR_DB_PATH}'...")
            Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embeddings).delete_collection()
            print(f"Deleted existing collection in '{VECTOR_DB_PATH}'.")
        except Exception as e:
            print(f"Could not delete collection, will attempt overwrite: {e}")

    try:
        print("Creating new vector store...")
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=VECTOR_DB_PATH
        )
        print(f"Vector store created with {vectorstore._collection.count()} documents.")
    except Exception as e:
        print(f"FATAL ERROR: Could not create vector store: {e}")
        vectorstore = None
else:
    print("WARNING: df_medicines is empty. Skipping vectorstore creation.")

# Setup retriever
retriever = None
if vectorstore:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    print("Retriever created.")
else:
    print("WARNING: Vectorstore not available.")

# ======================
# Fallback para RAG
# ======================
# ======================
# Fallback para RAG
# ======================
def rag_with_fallback(query, session_id="default", language="es"):
    try:
        history = get_session_history(session_id)
        result = rag_chain.invoke({"question": query, "chat_history": history.messages})

        if result and isinstance(result, str) and len(result.strip()) > 20:
            history.add_user_message(query)
            history.add_ai_message(result)
            return result
        else:
            if language == "es":
                return "Lo siento, no encontré un medicamento asociado a esos síntomas. ¿Podrías indicarme el nombre del medicamento que te interesa?"
            else:
                return "Sorry, I couldn't find a medicine associated with those symptoms. Could you please specify which medicine you're interested in?"

    except Exception as e:
        print(f"[RAG fallback error]: {str(e)}")
        if language == "es":
            return "Ocurrió un error al buscar información sobre el medicamento."
        else:
            return "There was an error retrieving the medicine information."


# In[9]:


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
        
        IMPORTANT: The inventory tables use 'Inventory' as the quantity column and 'Medicine_ID' to connect to the Medicines table's MedicineID.
        
        When asked about stores, store counts, or general locations, query the 'Stores' table.
        When asked "how many stores", run 'SELECT COUNT(*) FROM {SCHEMA_NAME}.Stores'.
        
        IMPORTANT: When asked about inventory, stock, quantity, or availability for a specific medicine:
        1. Identify the medicine name precisely.
        2. Identify the specific store location if mentioned (e.g., 'Chorrera', 'Costa del Este').
        3. Query the appropriate inventory table(s) using the 'Inventory' column to find the current quantity.
        4. Return the quantity clearly as a number if possible. If reporting for a specific store, mention it.
        
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


# In[10]:


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
# Intentar extraer el mensaje del usuario
        try:
            user_message = ""
            for msg in reversed(message_list):
                if isinstance(msg, dict) and msg.get('role') == 'user':
                    content = msg.get('content', '')
                    if isinstance(content, str):
                        user_message = content
                    elif isinstance(content, list):
                        user_message = " ".join(str(x) for x in content)
                    else:
                        user_message = str(content)
                    break
        
            user_message_lower = user_message.lower() if isinstance(user_message, str) else ""


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


# In[11]:


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


# In[12]:


# Optionally, view the first chunk
print(len(chunks))


# In[13]:


# Let's investigate the vectors

collection = vectorstore._collection
count = collection.count()

sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
dimensions = len(sample_embedding)
print(f"There are {count:,} vectors with {dimensions:,} dimensions in the vector store")


# In[14]:


# Add this function to help with consistent medicine name display
def get_normalized_medicine_name(found_name, query_name):
    """Returns the appropriate medicine name to display in responses,
    maintaining consistency with what the user asked for"""
    if not found_name:
        return query_name
        
    # If user asked for a brand name, return that instead of generic
    common_brand_generic_pairs = {
        "advil": "ibuprofeno", 
        "tylenol": "paracetamol",
        "motrin": "ibuprofeno",
        "aleve": "naproxeno",
        "bayer": "aspirina",
        "excedrin": "excedrin migraine"
    }
    
    query_lower = query_name.lower()
    found_lower = found_name.lower()
    
    # If user asked for a brand name that matches what we found, use their term
    for brand, generic in common_brand_generic_pairs.items():
        if brand in query_lower and generic in found_lower:
            return brand.capitalize()
    
    # Default to what we found in the database
    return found_name


# In[15]:


def inspect_all_inventory_tables():
    """Checks all inventory tables and prints their schema"""
    print("=== Inspecting Inventory Tables Schema ===")
    for sucursal in STORE_NAMES_ES:
        tabla = f"inventory_{sucursal.lower().replace(' ', '_')}"
        try:
            columns = inspect_inventory_table_schema(tabla)
            print(f"Table {tabla} columns: {columns}")
        except Exception as e:
            print(f"Error inspecting {tabla}: {str(e)}")
    print("=== End of Inspection ===")

    
def consultar_inventario_sql(medicamento, sucursal=None):
    """
    Consulta el inventario de un medicamento específico en la base de datos SQL.
    Adaptado para manejar la estructura actual de las tablas.
    """
    try:
        # Primero identificamos el medicamento en la tabla principal
        query_med = f"""
        SELECT MedicineID, [Generic Name], [Brand Name 1], [Brand Name 2] 
        FROM {SCHEMA_NAME}.Medicines 
        WHERE LOWER([Generic Name]) LIKE '%{medicamento.lower()}%' 
        OR LOWER([Brand Name 1]) LIKE '%{medicamento.lower()}%'
        OR LOWER([Brand Name 2]) LIKE '%{medicamento.lower()}%'
        OR '{medicamento.lower()}' LIKE '%' + LOWER([Brand Name 1]) + '%'
        OR '{medicamento.lower()}' LIKE '%' + LOWER([Brand Name 2]) + '%'
        """
        
        with engine.connect() as connection:
            med_result = pd.read_sql(query_med, connection)
        
        if med_result.empty:
            return {"stock": 0, "tiendas": [], "nombre_real": None, "detalles": None}
        
        # Obtener MedicineID y nombre real
        medicine_id = med_result.iloc[0]['MedicineID']
        nombre_real = med_result.iloc[0]['Generic Name']
        
        # Si hay una sucursal específica
        if sucursal:
            # Asumimos que el nombre de la tabla sigue el patrón "inventory_nombredesucursal"
            tabla = f"inventory_{sucursal.lower().replace(' ', '_')}"
            
            # Inspect table schema to determine the correct column names
            columns = inspect_inventory_table_schema(tabla)
            print(f"Columns in {tabla}: {columns}")
            
            # Adapt query based on actual column names
            # Assuming inventory tables have a medicine ID column and a quantity column
            # with potentially different names
            
            # Try common column name patterns (adjust these based on what you find)
            id_col = next((col for col in columns if 'id' in col.lower() or 'medicine' in col.lower()), None)
            qty_col = next((col for col in columns if 'qty' in col.lower() or 'Inventory' in col.lower() or 'stock' in col.lower()), None)
            
            if id_col and qty_col:
                query = f"""
                SELECT m.[Generic Name], m.[Brand Name 1], m.[Brand Name 2], i.[{qty_col}] as Inventory
                FROM {SCHEMA_NAME}.{tabla} i 
                JOIN {SCHEMA_NAME}.Medicines m ON i.[{id_col}] = m.MedicineID 
                WHERE m.MedicineID = {medicine_id}
                """
                
                with engine.connect() as connection:
                    result = pd.read_sql(query, connection)
                
                if result.empty:
                    return {"stock": 0, "tiendas": [], "nombre_real": nombre_real, "detalles": None}
                
                stock = int(result.iloc[0]['Inventory'])
                return {
                    "stock": stock, 
                    "tiendas": [sucursal] if stock > 0 else [], 
                    "nombre_real": nombre_real,
                    "detalles": result.to_dict('records')
                }
            else:
                # If we couldn't determine the column names, report the error
                return {
                    "stock": 0, 
                    "tiendas": [], 
                    "nombre_real": nombre_real, 
                    "detalles": None,
                    "error": f"Could not determine column names for {tabla}. Found: {columns}"
                }
            
        # Si no se especifica sucursal, consultar en todas
        else:
            # For checking all stores, we need to examine each inventory table
            sucursales_disponibles = []
            stock_total = 0
            detalles = []
            
            for sucursal in STORE_NAMES_ES:
                tabla = f"inventory_{sucursal.lower().replace(' ', '_')}"
                
                try:
                    # Inspect schema for this table
                    columns = inspect_inventory_table_schema(tabla)
                    
                    # Try to determine column names
                    id_col = next((col for col in columns if 'id' in col.lower() or 'medicine' in col.lower()), None)
                    qty_col = next((col for col in columns if 'qty' in col.lower() or 'Inventory' in col.lower() or 'stock' in col.lower()), None)
                    
                    if id_col and qty_col:
                        query_inv = f"""
                        SELECT [{qty_col}] as Inventory
                        FROM {SCHEMA_NAME}.{tabla} 
                        WHERE [{id_col}] = {medicine_id}
                        """
                        
                        with engine.connect() as connection:
                            inv_result = pd.read_sql(query_inv, connection)
                        
                        if not inv_result.empty and inv_result.iloc[0]['Inventory'] > 0:
                            cantidad = int(inv_result.iloc[0]['Inventory'])
                            stock_total += cantidad
                            sucursales_disponibles.append(sucursal)
                            detalles.append({
                                "sucursal": sucursal,
                                "cantidad": cantidad
                            })
                    else:
                        print(f"Could not determine column names for {tabla}. Found: {columns}")
                except Exception as e:
                    print(f"Error consultando {tabla}: {str(e)}")
                    continue
            
            return {
                "stock": stock_total, 
                "tiendas": sucursales_disponibles, 
                "nombre_real": nombre_real,
                "detalles": detalles
            }
            
    except Exception as e:
        print(f"Error en consulta SQL de inventario: {str(e)}")
        traceback.print_exc()  # Print full traceback for debugging
        # En caso de error, devolver información predeterminada
        return {"stock": 0, "tiendas": [], "nombre_real": medicamento, "detalles": None, "error": str(e)}


# In[16]:


def es_consulta_sintomas(mensaje, es_espanol=True):
    """Enhanced symptom detection with scoring"""
    # Convert parameter for compatibility with existing code
    language = "es" if es_espanol else "en"
    message_lower = mensaje.lower()
    
    # Define symptoms with synonyms and patterns
    symptoms_data = {
        # Spanish symptoms
        "fiebre": {
            "synonyms": ["temperatura", "calentura", "febril"],
            "patterns": ["tengo fiebre", "con fiebre", "fiebre alta", "para la fiebre", "contra la fiebre"],
            "weight": 1.0,
            "language": "es"
        },
        "dolor de cabeza": {  # Use exact phrase as it appears in queries
            "synonyms": ["migraña", "cefalea", "jaqueca", "dolor cabeza"],
            "patterns": ["me duele la cabeza", "con dolor de cabeza", "para el dolor de cabeza", "contra el dolor de cabeza"],
            "weight": 1.0,
            "language": "es"
        },
        "dolor de garganta": {
            "synonyms": ["garganta irritada", "garganta inflamada"],
            "patterns": ["me duele la garganta", "tengo dolor de garganta", "para el dolor de garganta"],
            "weight": 1.0,
            "language": "es"
        },
        "tos": {
            "synonyms": ["tosiendo", "toses"],
            "patterns": ["tengo tos", "para la tos", "medicina para tos"],
            "weight": 1.0,
            "language": "es"
        },
        "gripe": {
            "synonyms": ["resfriado", "resfrío", "resfrio", "influenza"],
            "patterns": ["tengo gripe", "para la gripe", "remedio para gripe"],
            "weight": 1.0,
            "language": "es"
        },
        
        # English symptoms
        "fever": {
            "synonyms": ["temperature", "high temp"],
            "patterns": ["have a fever", "with fever", "high fever", "for fever", "against fever"],
            "weight": 1.0,
            "language": "en"
        },
        "headache": {
            "synonyms": ["migraine", "head pain", "cephalalgia"],
            "patterns": ["my head hurts", "with headache", "for headache", "against headache"],
            "weight": 1.0,
            "language": "en"
        },
        "sore throat": {
            "synonyms": ["throat pain", "pharyngitis"],
            "patterns": ["my throat hurts", "have a sore throat", "for sore throat"],
            "weight": 1.0,
            "language": "en"
        }
        # Add more symptoms as needed
    }
    
    # Filter symptoms by language
    lang_symptoms = {k: v for k, v in symptoms_data.items() if v["language"] == language}
    
    # Common medicine query patterns
    medicine_patterns_es = [
        "medicina para", "medicamento para", "remedio para", "pastilla para", "píldora para",
        "que sirve para", "que ayude con", "que alivie", "para aliviar",
        "para el", "para la", "contra el", "contra la", "que cure", "tratamiento para"
    ]
    
    medicine_patterns_en = [
        "medicine for", "medication for", "remedy for", "pill for", 
        "that helps with", "to help with", "that relieves", "to relieve",
        "for", "against", "to cure", "treatment for"
    ]
    
    medicine_patterns = medicine_patterns_es if language == "es" else medicine_patterns_en
    
    # Check if this is a medicine query
    is_medicine_query = any(pattern in message_lower for pattern in medicine_patterns)
    
    # If not a medicine query, return early
    if not is_medicine_query:
        return False, None
    
    # Score each symptom
    scores = {}
    for symptom_key, data in lang_symptoms.items():
        score = 0.0
        
        # Check for exact match
        if symptom_key in message_lower:
            score += data["weight"] * 1.0
            
        # Check for synonyms
        for synonym in data["synonyms"]:
            if synonym in message_lower:
                score += data["weight"] * 0.8
                
        # Check for patterns
        for pattern in data["patterns"]:
            if pattern in message_lower:
                score += data["weight"] * 0.9
                
        if score > 0:
            scores[symptom_key] = score
    
    # Get top symptom if any scored above threshold
    if scores:
        top_symptom = max(scores.items(), key=lambda x: x[1])
        if top_symptom[1] >= 0.5:  # Lower threshold to catch more cases
            return True, top_symptom[0]
    
    # Check direct keywords (simpler fallback)
    # This is a backup in case the scoring approach fails
    direct_symptoms_es = [
        "fiebre", "dolor de cabeza", "dolor de garganta", "tos", "gripe", 
        "resfriado", "alergia", "dolor muscular", "insomnio", "ansiedad"
    ]
    
    direct_symptoms_en = [
        "fever", "headache", "sore throat", "cough", "flu", 
        "cold", "allergy", "muscle pain", "insomnia", "anxiety"
    ]
    
    direct_symptoms = direct_symptoms_es if language == "es" else direct_symptoms_en
    
    for symptom in direct_symptoms:
        if symptom in message_lower:
            return True, symptom
    
    return False, None

# Then define the interface function that includes all Gradio components and event handlers
def create_consolidated_interface():
    """Creates a simplified Gradio interface with language selection buttons"""
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Farma AI Panama")
        gr.Markdown("Asistente virtual para consultas sobre farmacias, horarios, medicamentos e inventario.")

        # Estado del idioma: "asking" (preguntando), "es" (español), o "en" (english)
        language_state = gr.State("asking")
        initial_greeting = "¡Hola! Soy el asistente virtual de Farma AI Panama. Por favor, selecciona tu idioma preferido / Hello! I'm Farma AI Panama's virtual assistant. Please select your preferred language:"

        # Chatbot principal
        chatbot = gr.Chatbot(
            height=500,
            type="messages",
            value=[{"role": "assistant", "content": initial_greeting}]
        )

        # Contenedor para los controles (para poder mostrar/ocultar grupos)
        with gr.Column() as controls:
            # Botones de idioma - visibles inicialmente
            with gr.Row(visible=True) as language_row:
                espanol_btn = gr.Button("Español", variant="primary")
                english_btn = gr.Button("English", variant="primary")
            
            # Controles de chat - ocultos inicialmente
            with gr.Row(visible=False) as chat_row:
                msg = gr.Textbox(placeholder="Escribe aquí...", show_label=False, scale=4)
                submit_button = gr.Button("Enviar", variant="primary", scale=1)
            
            # Botón para borrar conversación - oculto inicialmente
            clear_button = gr.Button("Borrar Conversación", visible=False)

        # Función para seleccionar español
        def select_spanish():
            response_text = "Has seleccionado español. ¿En qué puedo ayudarte hoy?"
            new_history = chatbot.value.copy()
            new_history.append({"role": "assistant", "content": response_text})
            return (
                new_history,                     # chatbot
                "es",                           # language_state
                gr.update(visible=False),       # language_row
                gr.update(visible=True),        # chat_row
                gr.update(visible=True),        # clear_button
                gr.update(placeholder="Escribe aquí..."),
                gr.update(value="Enviar"),
                gr.update(value="Borrar Conversación")
            )

        # Función para seleccionar inglés
        def select_english():
            response_text = "You've selected English. How can I help you today?"
            new_history = chatbot.value.copy()
            new_history.append({"role": "assistant", "content": response_text})
            return (
                new_history,                     # chatbot
                "en",                           # language_state
                gr.update(visible=False),       # language_row
                gr.update(visible=True),        # chat_row
                gr.update(visible=True),        # clear_button
                gr.update(placeholder="Type here..."),
                gr.update(value="Send"),
                gr.update(value="Clear Conversation")
            )

        # Función para manejar los mensajes del usuario (implementación completa)
        def handle_submit(user_message, chat_history, current_language):
            """
            Función completa para manejar las consultas del usuario
            con detección mejorada de síntomas vs. consultas de inventario
            """
            if not user_message:
                return "", chat_history
            
            # Agregar mensaje del usuario al historial
            chat_history.append({"role": "user", "content": user_message})
            
            try:
                # Determinar respuesta basada en palabras clave
                user_message_lower = user_message.lower()
                es_espanol = current_language == "es"
                
                # 1. PRIMERO: Verificar si es una consulta de síntomas
                es_sintoma, sintoma_detectado = es_consulta_sintomas(user_message, es_espanol)
                
                if es_sintoma and sintoma_detectado:
                    # Make sure the detected symptom is recognized
                    print(f"DEBUG: Symptom detected: {sintoma_detectado}")
                    
                    # Normalize symptom names
                    if es_espanol:
                        if "dolor" in sintoma_detectado and "cabeza" in sintoma_detectado:
                            sintoma_detectado = "dolor de cabeza"
                        elif "garganta" in sintoma_detectado:
                            sintoma_detectado = "dolor de garganta"
                    else:
                        if "head" in sintoma_detectado:
                            sintoma_detectado = "headache"
                        elif "throat" in sintoma_detectado:
                            sintoma_detectado = "sore throat"
                    
                    # Now continue with your existing symptom response code...
                    if "fiebre" in sintoma_detectado or "fever" in sintoma_detectado:
                        # Fiebre responses...
                        if es_espanol:
                            response = """
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
                        else:
                            response = """
                            For fever, we have the following medications:
                            
                            1. Paracetamol (Acetaminophen) - Tylenol, Tempra
                               - Reduces fever and relieves pain
                               - Available in tablets, syrup, and drops
                            
                            2. Ibuprofen - Advil, Motrin
                               - Anti-inflammatory that reduces fever and pain
                               - Available in tablets and suspension
                            
                            3. Naproxen - Aleve
                               - Long-acting anti-inflammatory
                               - For adults
                            
                            4. Acetylsalicylic Acid - Aspirin
                               - Reduces fever and inflammation
                               - Only for adults, not recommended for children
                            
                            Would you like more specific information about any of these medications?
                            """
                    elif "dolor de cabeza" in sintoma_detectado or "headache" in sintoma_detectado or "migraña" in sintoma_detectado or "migraine" in sintoma_detectado:
                        if es_espanol:
                            response = """
                            Para el dolor de cabeza o migraña, ofrecemos:
                            
                            1. Paracetamol (Acetaminofén) - Tylenol, Tempra
                               - Analgésico eficaz para dolores leves a moderados
                            
                            2. Ibuprofeno - Advil, Motrin
                               - Antiinflamatorio especialmente útil para dolor con inflamación
                            
                            3. Naproxeno - Aleve
                               - Alivia dolores de cabeza por más tiempo (hasta 12 horas)
                            
                            4. Aspirina - Bayer, Ecotrin
                               - Eficaz para dolores de cabeza y migrañas en adultos
                            
                            5. Medicamentos específicos para migraña:
                               - Sumatriptán (con receta)
                               - Excedrin Migraine (combinación de acetaminofén, aspirina y cafeína)
                            
                            ¿Necesitas información más específica sobre alguno de estos?
                            """
                        else:
                            response = """
                            For headaches or migraines, we offer:
                            
                            1. Paracetamol (Acetaminophen) - Tylenol, Tempra
                               - Effective analgesic for mild to moderate pain
                            
                            2. Ibuprofen - Advil, Motrin
                               - Anti-inflammatory especially useful for pain with inflammation
                            
                            3. Naproxen - Aleve
                               - Relieves headaches for longer (up to 12 hours)
                            
                            4. Aspirin - Bayer, Ecotrin
                               - Effective for headaches and migraines in adults
                            
                            5. Migraine-specific medications:
                               - Sumatriptan (prescription required)
                               - Excedrin Migraine (combination of acetaminophen, aspirin, and caffeine)
                            
                            Do you need more specific information about any of these?
                            """
                    else:
                        # Para otros síntomas, intentar usar RAG o respuesta genérica
                        try:
                            # Intentar usar RAG para responder sobre el síntoma
                            if es_espanol:
                                # Formular una consulta más específica para el sistema RAG
                                consulta_rag = f"Qué medicamentos ayudan con {sintoma_detectado} y están disponibles en la farmacia?"
                                response = rag_with_fallback(consulta_rag, language="es")
                            else:
                                consulta_rag = f"What medications help with {sintoma_detectado} and are available in the pharmacy?"
                                response = rag_with_fallback(consulta_rag, language="en")
                            
                            # Si la respuesta es demasiado genérica o fallback, usar respuesta predefinida
                            if "no encontré un medicamento asociado" in response or "couldn't find a medicine" in response:
                                if es_espanol:
                                    response = f"""
                                    Para {sintoma_detectado}, recomendamos lo siguiente:
                                    
                                    - Consulta con uno de nuestros farmacéuticos en la sucursal más cercana para recomendaciones personalizadas.
                                    - Tenemos varias opciones que pueden ayudar, pero necesitaríamos más información sobre tu situación específica.
                                    
                                    ¿Hay algún medicamento específico que estés buscando para este síntoma?
                                    """
                                else:
                                    response = f"""
                                    For {sintoma_detectado}, we recommend the following:
                                    
                                    - Consult with one of our pharmacists at the nearest branch for personalized recommendations.
                                    - We have several options that may help, but we would need more information about your specific situation.
                                    
                                    Is there a specific medication you're looking for to treat this symptom?
                                    """
                        except Exception as e:
                            print(f"Error en RAG para síntomas: {e}")
                            # Respuesta genérica en caso de error
                            if es_espanol:
                                response = f"""
                                Tenemos varios medicamentos para tratar {sintoma_detectado}. 
                                Para recomendaciones más precisas, te sugiero visitar cualquiera de nuestras sucursales 
                                donde nuestros farmacéuticos pueden ayudarte según tu caso específico.
                                """
                            else:
                                response = f"""
                                We have several medications to treat {sintoma_detectado}.
                                For more precise recommendations, I suggest visiting any of our branches
                                where our pharmacists can help you according to your specific case.
                                """
                # 2. SEGUNDO: Si no es consulta de síntomas, verificar si es consulta de inventario
                else:
                    # Lista de palabras clave para detectar consultas de inventario
                    palabras_disponibilidad_es = ["tienes", "tienen", "hay", "disponible", "stock", "existencia", 
                                                "venden", "vende", "tendrás", "tendrán", "consigo", "encontrar"]
                    palabras_disponibilidad_en = ["have", "stock", "available", "sell", "find", "get", "inventory", "carry"]
                    
                    # Lista ampliada de palabras a excluir
                    palabras_excluidas_es = [
                        # Verbos de consulta
                        "tienes", "tienen", "hay", "tendrás", "tendrán", "venden", "vende", 
                        # Preposiciones y artículos
                        "para", "como", "este", "esta", "sobre", "cual", "algún", "algun", "un", "una", "unos", "unas",
                        # Adverbios y conjunciones
                        "también", "tambien", "pero", "aunque", "cuando", "donde", "dónde", "cómo", "como",
                        # Otras palabras comunes
                        "ustedes", "farmacia", "disponible", "disponibles", "stock", "existencia", "inventario",
                        # Palabras genéricas sobre medicamentos
                        "medicina", "medicinas", "medicamento", "medicamentos", "remedio", "remedios", "pastilla", "pastillas",
                        # Signos de puntuación (a eliminar, no a excluir como palabra)
                        "?", "¿", ".", ",", "!", "¡"
                    ]
                    
                    palabras_excluidas_en = [
                        # Query verbs
                        "have", "has", "had", "sell", "sells", "selling", "get", "getting", "carry", "carrying",
                        # Prepositions and articles
                        "for", "like", "this", "that", "about", "which", "some", "a", "an", "the",
                        # Adverbs and conjunctions
                        "also", "but", "although", "when", "where", "how",
                        # Other common words
                        "you", "your", "pharmacy", "available", "stock", "inventory",
                        # Generic medicine words
                        "medicine", "medicines", "medication", "medications", "remedy", "remedies", "pill", "pills", "drug", "drugs",
                        # Punctuation (to be removed, not excluded as words)
                        "?", ".", ",", "!"
                    ]
                    
                    # Verificar si hay palabras de disponibilidad en el mensaje
                    palabras_disp = palabras_disponibilidad_es if es_espanol else palabras_disponibilidad_en
                    palabras_excl = palabras_excluidas_es if es_espanol else palabras_excluidas_en
                    
                    es_pregunta_disponibilidad = any(palabra in user_message_lower.split() for palabra in palabras_disp)
                    
                    if es_pregunta_disponibilidad:
                        # Limpiar el mensaje de signos de puntuación
                        mensaje_limpio = user_message_lower.replace("?", " ").replace("¿", " ").replace(".", " ").replace(",", " ")
                        mensaje_limpio = mensaje_limpio.replace("!", " ").replace("¡", " ").replace(":", " ").replace(";", " ")
                        
                        # Eliminar espacios múltiples
                        while "  " in mensaje_limpio:
                            mensaje_limpio = mensaje_limpio.replace("  ", " ")
                        
                        mensaje_limpio = mensaje_limpio.strip()
                        palabras = mensaje_limpio.split()
                        
                        # Identificar candidatos a medicamentos (palabras de 4+ caracteres que no están en la lista de exclusión)
                        candidatos = []
                        for palabra in palabras:
                            if len(palabra) >= 4 and palabra not in palabras_excl:
                                candidatos.append(palabra)
                        
                        # Si hay candidatos, consultar cada uno
                        if candidatos:
                            for medicamento in candidatos:
                                try:
                                    # Consultar en la base de datos
                                    resultado = consultar_inventario_sql(medicamento)
                                    
                                    # Si encontramos coincidencia en la base de datos
                                    if resultado["nombre_real"]:
                                        nombre_medicamento = resultado["nombre_real"]
                                        info = resultado
                                        
                                        # Generar respuesta según disponibilidad e idioma
                                        if es_espanol:
                                            if info["stock"] > 0:
                                                if len(info["tiendas"]) == len(STORE_NAMES_ES):  # Si está en todas las tiendas
                                                    response = f"""
                                                    Sí, tenemos {nombre_medicamento} disponible en todas nuestras sucursales.
                                                    Actualmente contamos con un total de {info["stock"]} unidades en inventario.
                                                    
                                                    ¿Necesitas información adicional sobre este medicamento o te gustaría saber sobre alguna sucursal específica?
                                                    """
                                                else:
                                                    sucursales = ", ".join(info["tiendas"])
                                                    response = f"""
                                                    Sí, tenemos {nombre_medicamento} disponible en las siguientes sucursales: {sucursales}.
                                                    Actualmente contamos con un total de {info["stock"]} unidades distribuidas en estas sucursales.
                                                    
                                                    ¿Necesitas información adicional sobre este medicamento?
                                                    """
                                            else:
                                                response = f"""
                                                Lo siento, actualmente no tenemos {nombre_medicamento} en inventario en ninguna de nuestras sucursales.
                                                ¿Te gustaría que te recomendara alguna alternativa similar?
                                                """
                                        else:  # English
                                            if info["stock"] > 0:
                                                if len(info["tiendas"]) == len(STORE_NAMES_EN):
                                                    response = f"""
                                                    Yes, we have {nombre_medicamento} available in all our branches.
                                                    We currently have a total of {info["stock"]} units in inventory.
                                                    
                                                    Do you need additional information about this medication or would you like to know about a specific branch?
                                                    """
                                                else:
                                                    branches = ", ".join(info["tiendas"])
                                                    response = f"""
                                                    Yes, we have {nombre_medicamento} available at the following branches: {branches}.
                                                    We currently have a total of {info["stock"]} units distributed across these branches.
                                                    
                                                    Do you need additional information about this medication?
                                                    """
                                            else:
                                                response = f"""
                                                I'm sorry, we currently don't have {nombre_medicamento} in stock in any of our branches.
                                                Would you like me to recommend a similar alternative?
                                                """
                                        break  # Encontramos un medicamento, terminamos la búsqueda
                                except Exception as e:
                                    print(f"Error al consultar medicamento {medicamento}: {e}")
                                    continue
                            
                            # Si llegamos aquí sin respuesta, no encontramos medicamentos
                            if 'response' not in locals():
                                if es_espanol:
                                    response = """
                                    No pude identificar claramente qué medicamento estás buscando. 
                                    ¿Podrías ser más específico con el nombre del medicamento?
                                    """
                                else:
                                    response = """
                                    I couldn't clearly identify which medication you're looking for.
                                    Could you be more specific with the name of the medication?
                                    """
                        else:
                            # No hay candidatos a medicamentos
                            if es_espanol:
                                response = """
                                No pude identificar qué medicamento estás buscando.
                                ¿Podrías mencionar el nombre específico del medicamento?
                                """
                            else:
                                response = """
                                I couldn't identify which medication you're looking for.
                                Could you mention the specific name of the medication?
                                """
                    
                    # 3. TERCERO: Si no es consulta de inventario, procesar otras consultas
                    else:
                        # Para consultas sobre tiendas o ubicaciones usando SQL
                        if any(keyword in user_message_lower for keyword in [
                            "tienda", "ubicación", "dirección", "sucursal", "farmacia",
                            "store", "location", "address", "branch", "pharmacy"]):
                            try:
                                # Usar el SQL agent
                                response = sql_query.invoke(user_message)
                            except Exception as e:
                                print(f"Error en SQL query: {e}")
                                # Respuesta de error según idioma
                                if es_espanol:
                                    response = """
                                    Lo siento, tuve un problema consultando información sobre nuestras tiendas. 
                                    Por favor, intenta de nuevo con una pregunta más específica.
                                    """
                                else:
                                    response = """
                                    I'm sorry, I had an issue querying information about our stores. 
                                    Please try again with a more specific question.
                                    """
                        
                        # Para consultas sobre medicamentos específicos usando RAG
                        elif any(keyword in user_message_lower for keyword in [
                            "medicamento", "medicina", "droga", "pastilla", "jarabe", 
                            "medication", "medicine", "drug", "pill", "syrup"]):
                            try:
                                # Intentar buscar con el retriever
                                if es_espanol:
                                    response = rag_with_fallback(user_message, language="es")
                                else:
                                    response = rag_with_fallback(user_message, language="en")
                            except Exception as e:
                                print(f"Error en RAG: {e}")
                                # Respuesta de error según idioma
                                if es_espanol:
                                    response = """
                                    Lo siento, tuve un problema buscando información sobre ese medicamento. 
                                    ¿Podrías intentar reformular tu pregunta?
                                    """
                                else:
                                    response = """
                                    I'm sorry, I had an issue finding information about that medication. 
                                    Could you try rephrasing your question?
                                    """
                        
                        # Para otros casos, respuesta genérica según idioma
                        else:
                            if es_espanol:
                                response = """
                                Puedo ayudarte con información sobre:
                                
                                - Medicamentos y sus usos
                                - Síntomas y qué medicamentos pueden ayudar
                                - Ubicaciones de nuestras farmacias
                                - Disponibilidad de medicamentos en las sucursales
                                
                                ¿Qué información específica necesitas?
                                """
                            else:  # English
                                response = """
                                I can help you with information about:
                                
                                - Medications and their uses
                                - Symptoms and which medications may help
                                - Locations of our pharmacies
                                - Availability of medications in our branches
                                
                                What specific information do you need?
                                """
                
                # Agregar respuesta al historial
                chat_history.append({"role": "assistant", "content": response})
                
                # Logging de la interacción
                es_pregunta_disponibilidad = 'es_pregunta_disponibilidad' in locals() and es_pregunta_disponibilidad
                query_type = "symptom" if es_sintoma else "inventory" if es_pregunta_disponibilidad else "general"
                log_interaction(query=user_message, response=response, query_type=query_type)
                
                # Retornar mensaje vacío (para limpiar input) y el historial actualizado
                return "", chat_history
                
            except Exception as e:
                import traceback
                print(f"ERROR EN HANDLE_SUBMIT: {str(e)}")
                print(traceback.format_exc())
                
                # Respuesta genérica de error según idioma
                if current_language == "es":
                    error_msg = "Lo siento, ocurrió un error. Por favor, intenta con otra pregunta."
                else:
                    error_msg = "I'm sorry, an error occurred. Please try with another question."
                
                chat_history.append({"role": "assistant", "content": error_msg})
                return "", chat_history
        
        # Función para borrar el historial
        def clear_history():
            initial_msg = "¿En qué puedo ayudarte hoy?"
            return [{"role": "assistant", "content": initial_msg}]

        # Asociar eventos y botones
        espanol_btn.click(
            fn=select_spanish,
            inputs=[],
            outputs=[chatbot, language_state, language_row, chat_row, clear_button, msg, submit_button, clear_button]
        )
        
        english_btn.click(
            fn=select_english,
            inputs=[],
            outputs=[chatbot, language_state, language_row, chat_row, clear_button, msg, submit_button, clear_button]
        )
        
        msg.submit(
            fn=handle_submit,
            inputs=[msg, chatbot, language_state],
            outputs=[msg, chatbot]
        )
        
        submit_button.click(
            fn=handle_submit,
            inputs=[msg, chatbot, language_state],
            outputs=[msg, chatbot]
        )
        
        clear_button.click(
            fn=clear_history,
            inputs=[],
            outputs=[chatbot]
        )
        
        # Importante: retornar el objeto demo para que pueda ser lanzado
        return demo



# In[17]:


def inspect_all_inventory_tables():
    """Checks all inventory tables and prints their schema with a focus on the Inventory column"""
    print("=== Inspecting Inventory Tables Schema ===")
    for sucursal in STORE_NAMES_ES:
        tabla = f"inventory_{sucursal.lower().replace(' ', '_')}"
        try:
            # Execute a query to check table structure
            query = f"SELECT TOP 1 * FROM {SCHEMA_NAME}.{tabla}"
            with engine.connect() as connection:
                result = connection.execute(query)
                columns = list(result.keys())
                
            # Check specifically for Inventory column
            has_inventory = "Inventory" in columns
            medicine_id_col = next((col for col in columns if "id" in col.lower() and "medicine" in col.lower()), None)
            
            print(f"Table {tabla}:")
            print(f"  - Columns: {columns}")
            print(f"  - Has 'Inventory' column: {has_inventory}")
            print(f"  - Medicine ID column: {medicine_id_col}")
            
            # If inventory column not found, warn
            if not has_inventory:
                print(f"  ! WARNING: 'Inventory' column not found in {tabla}")
            
            # If medicine ID column not found, warn
            if not medicine_id_col:
                print(f"  ! WARNING: No medicine ID column found in {tabla}")
                
        except Exception as e:
            print(f"Error inspecting {tabla}: {str(e)}")
    print("=== End of Inspection ===")


# In[ ]:


# Main execution block
if __name__ == "__main__":
    # Ensure critical components are available before launching UI
    if engine and db and llm and vectorstore and retriever and sql_agent:
        print("All components initialized successfully. Launching Gradio interface...")
        interface = create_consolidated_interface()
        interface.launch(debug=True, share=True) # share=True can expose it publicly
    else:
        print("ERROR: Could not initialize all required components. Aborting launch.")
        log_interaction(query="Launch Check", response="Aborted", query_type="launch_error", error="Missing components")

