# config.py
import os
from dotenv import load_dotenv

load_dotenv(override=True)

# Database configuration
SERVER_NAME = "localhost"
DATABASE_NAME = "ChatbotFarmacia"
TABLE_NAME = "Medicines"
SCHEMA_NAME = "dbo"
DRIVER = "ODBC Driver 17 for SQL Server"
CONNECTION_STRING = f"mssql+pyodbc://{SERVER_NAME}/{DATABASE_NAME}?driver={DRIVER}&trusted_connection=yes"

# Model configuration
MODEL = "gpt-4o-mini"

# OpenAI API Key
load_dotenv(override=True)
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')

