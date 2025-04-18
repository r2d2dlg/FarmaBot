#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install langgraph


# In[ ]:


pip install langchain_experimental


# In[ ]:


import langchain_experimental.agent_toolkits


# In[22]:


# imports

import os
import glob
from dotenv import load_dotenv
import gradio as gr


# In[24]:


# LangChain: Document loading, splitting, and core schema
from langchain.document_loaders import DirectoryLoader, TextLoader # Used if loading from files
from langchain.text_splitter import CharacterTextSplitter # Used if splitting generic text
from langchain.docstore.document import Document
from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.llms.openai import OpenAI
from langchain.agents.agent_types import AgentType

# Embeddings: OpenAI or HuggingFace
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from langchain.embeddings import HuggingFaceEmbeddings # Keep if using HF

# Vector store: Chroma (Recommended Import)
from langchain_chroma import Chroma
# from langchain.vectorstores import Chroma # Deprecated, can be removed if using the line above

# Conversational memory and chain
from langchain.sql_database import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
# from langchain.retrievers.self_query.base import SelfQueryRetriever # Needed for Option 2
from langchain.tools import Tool
# from langchain.chains.query_constructor.base import AttributeInfo # Needed for Option 2

# Visualization and analysis
import matplotlib.pyplot as plt # Not used in recent steps, but maybe later
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import numpy as np

# Data handling
import pandas as pd
from sqlalchemy import create_engine # Only need one of these imports
import urllib # Needed for quoting SQL connection parameters

# Callbacks (Optional, for debugging)
# from langchain_core.callbacks import StdOutCallbackHandler

# OS utilities (Used for checking db path)
import os



# In[26]:


# price is a factor for our company, so we're going to use a low cost model

MODEL = "gpt-4o-mini"
db_name = "Medicines"


# In[28]:


# Load environment variables in a file called .env

load_dotenv(override=True)
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')


# In[30]:


# --- Database Connection Details ---
SERVER_NAME = "localhost"
DATABASE_NAME = "ChatbotFarmacia" # <-- Good, you've set your DB name
TABLE_NAME = "Medicines" # Note: TABLE_NAME here isn't used for the engine itself
SCHEMA_NAME = "dbo"      # Note: SCHEMA_NAME here isn't used for the engine itself
driver = "ODBC Driver 17 for SQL Server"
connection_string = f"mssql+pyodbc://{SERVER_NAME}/{DATABASE_NAME}?driver={driver}&trusted_connection=yes"


# --- Create Engine & Load Data ---
df = pd.DataFrame()
chunks = []
try:
    print(f"Connecting to DB: {DATABASE_NAME} on {SERVER_NAME}...")
    engine = create_engine(connection_string)
    sql_query = f"SELECT * FROM [dbo].[Medicines]" # Or your relevant query
    print(f"Loading data...")
    df = pd.read_sql(sql_query, engine)
    print(f"Successfully loaded {len(df)} rows.")

    # --- Split DataFrame into Chunks ---
    # This line creates the 'chunks' variable needed below
    chunks = [df.iloc[i:i+5] for i in range(0, len(df), 5)]
    print(f"Data split into {len(chunks)} chunks.")

except Exception as e:
    print(f"Error loading data or creating chunks: {e}")
# --- Create Database Engine ---
try:
    print(f"Attempting to connect to {DATABASE_NAME} on {SERVER_NAME}...")
    engine = create_engine(connection_string)
    # Optional connection test
    connection = engine.connect()
    print(f"Successfully connected to database '{DATABASE_NAME}' on '{SERVER_NAME}'.")
    connection.close()
    print("SQLAlchemy engine created.")

except Exception as e:
    print(f"Error connecting to database: {e}")
    # Handle error
    exit()

# --- The 'engine' variable created above is what you need for the SQL Agent ---

include_tables = ["Medicines", "inventory", "inventory_chorrera", "inventory_costa_del_este", "inventory_david", "inventory_el_dorado", "inventory_san_francisco",  "Stores"] # List all tables
db = SQLDatabase(engine=engine, schema="dbo", include_tables=include_tables)
# Optional: print(db.get_table_info())


# In[32]:


include_tables = ["Medicines", "inventory", "inventory_chorrera", "inventory_costa_del_este", "inventory_david", "inventory_el_dorado", "inventory_san_francisco",  "Stores" ] # List all tables
db = SQLDatabase(engine=engine, schema="dbo", include_tables=include_tables)
# Optional: print(db.get_table_info())


# In[34]:


from langchain.agents import create_sql_agent


# In[36]:


llm = ChatOpenAI(model=MODEL, temperature=2) # Low temp recommended for agent logic

# Add this import line, typically near the top with your other imports


# --- Your existing code ---
# llm = ChatOpenAI(...)
# db = SQLDatabase(...)
# --- End of existing code ---

# Now you can create the agent (this line should work after the import)
sql_agent = create_sql_agent(llm=llm, db=db, agent_type="openai-tools", verbose=True) # Added verbose=True

print("SQL Agent created successfully.")


# In[40]:


print(list(AgentType))


# In[74]:


# Define the LLM
llm = OpenAI(temperature=0)

# Define the actual vector search function
def vector_search(query):
    # Replace with your actual vector search implementation
    result = conversation_chain.invoke({"question": query})
    return result["answer"]

# Define the actual SQL query function
def sql_query(query):
    # Replace with your actual SQL agent implementation
    return sql_agent.invoke(query)

# Define the tools with the actual functions
vector_search_tool = Tool(
    name="VectorSearchAgent",
    func=vector_search,
    description="Useful for answering questions about medicines, drugs, their properties, side effects, and uses."
)

sql_search_tool = Tool(
    name="SQLAgent",
    func=sql_query,
    description="Useful for answering questions about stock, inventory levels, and store availability."
)

# Initialize the agent
agent = initialize_agent(
    tools=[vector_search_tool, sql_search_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Update the Gradio interface
def chat(question, history):
    response = agent.invoke({"input": question})
    return response["output"]
# Test the agent
query = "What are the side effects of aspirin?"
response = agent.invoke({"input": query})
print(response["output"])

query = "How much stock do we have of ibuprofen?"
response = agent.invoke({"input": query})
print(response["output"])


# In[44]:


# --- Corrected Document Creation Loop (Option 1) ---
# Assuming 'chunks' is your list of DataFrames from the SQL query
# Requires: from langchain.docstore.document import Document

docs = []
print("Starting document conversion (translating 1/0 status to text in page_content)...")
for i, chunk_df in enumerate(chunks):
    for index, row in chunk_df.iterrows():
        try:
            # --- Get the numeric status (assuming column name is 'Prescription') ---
            try:
                 # Use the actual column name from your SQL table if different from 'Prescription'
                 status_flag = int(row.get('Prescription', -1)) # Get 1, 0, or -1
            except (ValueError, TypeError):
                 status_flag = -1 # Handle non-numeric or missing data

            # --- Translate numeric status to text ---
            if status_flag == 1:
                status_text = "Requires Prescription"
            elif status_flag == 0:
                status_text = "Over-the-Counter"
            else:
                status_text = "Unknown"

            # --- MODIFIED page_content to include the status TEXT ---
            # Use correct column names from your SQL table (e.g., 'Generic Name', 'Uses')
            page_content = f"Medicine: {row['Generic Name']}\nUses: {row['Uses']}\nPrescription Status: {status_text}"

            # --- Metadata: Store the numeric flag and other relevant fields ---
            metadata = {
                "source_db_table": f"{SCHEMA_NAME}.{TABLE_NAME}", # Identify source
                # Use primary key from DB if available and useful, otherwise use index
                # "db_primary_key": row.get('YourPrimaryKeyColumn'),
                "chunk_index": i,
                # Store the numeric flag using a clear key name
                "prescription_required_flag": status_flag,
                # Add other relevant fields from your DB table, ensuring column names match
                "uses": row.get('Uses', ""),
                "side_effects_common": row.get('Side Effects (Common)', ""),
                "side_effects_rare": row.get('Side Effects (Rare)', ""),
                "similar_drugs": row.get('Similar Drugs', ""),
                "brand_name_1": row.get('Brand Name 1', ""),
                # ... etc
            }
            docs.append(Document(page_content=page_content, metadata=metadata))
        except KeyError as e:
            print(f"KeyError processing row {index}: {e} - Check column names from DB query!")
        except Exception as e:
             print(f"Error processing row {index}: {e}")

print(f"Created {len(docs)} Document objects with updated page_content.")
# --- End of Corrected Document Creation Loop ---


# In[46]:


# Optionally, view the first chunk
print(len(chunks))


# In[48]:


# Put the chunks of data into a Vector Store that associates a Vector Embedding with each chunk
# Chroma is a popular open source Vector Database based on SQLLite

embeddings = OpenAIEmbeddings() # Assumes 'from langchain_openai import OpenAIEmbeddings' was used
                               # and the OpenAI API key is configured (e.g., environment variable)

# If you would rather use the free Vector Embeddings from HuggingFace sentence-transformers
# Then replace embeddings = OpenAIEmbeddings()
# with:
# from langchain.embeddings import HuggingFaceEmbeddings
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Delete if already exists
# Assumes 'db_name' variable (string path) is defined earlier
# Assumes 'import os' and 'from langchain_chroma import Chroma' were used

if os.path.exists(db_name):
    try:
        # Attempt to connect and delete the collection within the directory
        Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()
        print(f"Deleted existing collection in '{db_name}'.")
    except Exception as e:
        # Handle cases where deletion might fail (e.g., directory exists but isn't a valid Chroma DB)
        print(f"Could not delete collection in '{db_name}': {e}")

# Create vectorstore
# CRITICAL: Assumes 'docs' is a list of LangChain Document objects
# It seems like 'docs' might still be undefined based on your previous error.
# You need to convert your DataFrame chunks into Document objects first.

vectorstore = Chroma.from_documents(documents=docs, # 'docs' needs to be List[Document]
                                     embedding=embeddings,
                                     persist_directory=db_name)
print(f"Vectorstore created with {vectorstore._collection.count()} documents in '{db_name}'.")


# In[49]:


vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=db_name)
print(f"Vectorstore created with {vectorstore._collection.count()} documents")


# In[52]:


# Let's investigate the vectors

collection = vectorstore._collection
count = collection.count()

sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
dimensions = len(sample_embedding)
print(f"There are {count:,} vectors with {dimensions:,} dimensions in the vector store")


# In[54]:


result = collection.get(include=['embeddings', 'documents', 'metadatas'])
vectors = np.array(result['embeddings']) # Requires: import numpy as np
documents = result['documents']
metadatas = result['metadatas']
# doc_types = [metadata['doc_type'] for metadata in metadatas if metadata is not None] # REMOVED/COMMENTED
# colors = [['blue', 'green', 'red', 'orange'][['products', 'employees', 'contracts', 'company'].index(t)] for t in doc_types] # REMOVED/COMMENTED

# You can now work with vectors, documents, metadatas
print(f"Retrieved {len(vectors)} items.")
if metadatas:
     print("First item metadata:", metadatas[0])


# In[56]:


# Assume 'vectors' (numpy array) and 'documents' (list of strings) exist from collection.get()

print("Running t-SNE...")
tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(vectors)-1)) # Added perplexity adjustment
reduced_vectors = tsne.fit_transform(vectors)
print("t-SNE complete.")

# Create the 2D scatter plot (Simplified)
fig = go.Figure(data=[go.Scatter(
    x=reduced_vectors[:, 0],
    y=reduced_vectors[:, 1],
    mode='markers',
    marker=dict(size=5, opacity=0.8), # Removed 'color=colors'
    # Simplified hover text using index and document snippet
    text=[f"Index: {i}<br>Text: {d[:100]}..." for i, d in enumerate(documents)],
    hoverinfo='text'
)])

fig.update_layout(
    title='2D Chroma Vector Store Visualization (t-SNE)',
    xaxis_title='t-SNE Component 1', # More specific axis titles
    yaxis_title='t-SNE Component 2',
    width=800,
    height=600,
    margin=dict(r=20, b=10, l=10, t=40)
)

fig.show()


# In[57]:


# Assume 'vectors', 'documents', 'metadatas' exist

# --- Block to define colors (MUST RUN BEFORE PLOTTING) ---
print("Defining colors based on metadata...")
status_list = [metadata.get('prescription_required', 'Unknown') for metadata in metadatas if metadata is not None]
color_map = {'Prescription': 'red', 'Non-Prescription': 'blue', 'Unknown': 'grey'} # Adjust as needed
colors = [color_map.get(status, 'grey') for status in status_list]
print("Colors defined.")
# --- End of color definition block ---

tsne = TSNE(n_components=3, random_state=42, perplexity=min(30, len(vectors)-1))
print("Running 3D t-SNE...")
reduced_vectors = tsne.fit_transform(vectors)
print("3D t-SNE complete.")

# Create the 3D scatter plot (Using defined colors and status_list)
fig = go.Figure(data=[go.Scatter3d(
    x=reduced_vectors[:, 0],
    y=reduced_vectors[:, 1],
    z=reduced_vectors[:, 2],
    mode='markers',
    marker=dict(size=3, color=colors, opacity=0.7), # Use defined 'colors'
    # Update hover text
    text=[f"Prescription: {s}<br>Text: {d[:100]}..." for s, d in zip(status_list, documents)],
    hoverinfo='text'
)])

fig.update_layout(
    title='3D Chroma Vector Store Visualization (t-SNE by Prescription Status)',
    scene=dict(xaxis_title='t-SNE Comp 1', yaxis_title='t-SNE Comp 2', zaxis_title='t-SNE Comp 3'),
    width=900,
    height=700,
    margin=dict(r=20, b=10, l=10, t=40)
)

fig.show()


# In[59]:


# Define the model name as a string
MODEL = "gpt-4"

# create a new Chat with OpenAI
llm = ChatOpenAI(temperature=0.7, model_name=MODEL)

# Alternative - if you'd like to use Ollama locally, uncomment this line instead
# llm = ChatOpenAI(temperature=0.7, model_name='llama3.2', base_url='http://localhost:11434/v1', api_key='ollama')

# set up the conversation memory for the chat
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

# the retriever is an abstraction over the VectorStore that will be used during RAG
retriever = vectorstore.as_retriever()

# putting it together: set up the conversation chain with the GPT 3.5 LLM, the vector store and memory
conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)


# In[62]:


# Wrapping that in a function

def chat(question, history):
    result = conversation_chain.invoke({"question": question})
    return result["answer"]


# In[64]:


# And in Gradio:

view = gr.ChatInterface(chat, type="messages").launch(inbrowser=True)


# In[65]:


# Let's investigate what gets sent behind the scenes

from langchain_core.callbacks import StdOutCallbackHandler

llm = ChatOpenAI(temperature=0.7, model_name=MODEL)

memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

retriever = vectorstore.as_retriever()

conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory, callbacks=[StdOutCallbackHandler()])

query = "What medications can you recommend for headaches?"
result = conversation_chain.invoke({"question": query})
answer = result["answer"]
print("\nAnswer:", answer)


# In[68]:


# create a new Chat with OpenAI
llm = ChatOpenAI(temperature=0.7, model_name=MODEL)

# set up the conversation memory for the chat
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

# the retriever is an abstraction over the VectorStore that will be used during RAG; k is how many chunks to use
retriever = vectorstore.as_retriever(search_kwargs={"k": 25})

# putting it together: set up the conversation chain with the GPT 3.5 LLM, the vector store and memory
conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)


# In[70]:


def chat(question, history):
    result = conversation_chain.invoke({"question": question})
    return result["answer"]


# In[72]:


view = gr.ChatInterface(chat, type="messages").launch(inbrowser=True)


# In[ ]:




