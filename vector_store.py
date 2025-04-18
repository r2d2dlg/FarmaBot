from langchain.docstore.document import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import os

def setup_vector_store(df_medicines, vector_db_path):
    docs = []
    for _, row in df_medicines.iterrows():
        page_content = f"Medicine: {row['Generic Name']}\nUses: {row['Uses']}"
        metadata = {"source_db_table": "dbo.Medicines"}
        docs.append(Document(page_content=page_content, metadata=metadata))

    if os.path.exists(vector_db_path):
        Chroma(persist_directory=vector_db_path).delete_collection()

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=OpenAIEmbeddings(),
        persist_directory=vector_db_path
    )
    return vectorstore