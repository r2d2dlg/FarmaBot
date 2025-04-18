import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Database connection details
SERVER_NAME = "localhost"
DATABASE_NAME = "ChatbotFarmacia"
DRIVER = "ODBC Driver 17 for SQL Server"
SCHEMA_NAME = "dbo"
TABLE_NAME = "Medicines"

connection_string = f"mssql+pyodbc://{SERVER_NAME}/{DATABASE_NAME}?driver={DRIVER}"
engine = create_engine(connection_string)

def connect_to_database():
    try:
        with engine.connect() as connection:
            print(f"Successfully connected to database '{DATABASE_NAME}' on '{SERVER_NAME}'.")
        return engine
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

def load_medicines_data():
    try:
        sql_query = f"SELECT * FROM [{SCHEMA_NAME}].[{TABLE_NAME}]"
        df = pd.read_sql(sql_query, engine)
        print(f"Loaded {len(df)} rows from the Medicines table.")
        return df
    except Exception as e:
        print(f"Error loading medicines data: {e}")
        return pd.DataFrame()