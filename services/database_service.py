"""
Database service for FarmaBot - Handles database connections and operations.
"""

import logging
from typing import Optional, List, Dict, Any
from sqlalchemy import create_engine, text
from langchain_community.utilities.sql_database import SQLDatabase
from langchain.agents import AgentType, create_sql_agent
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain_openai import ChatOpenAI

class DatabaseService:
    def __init__(self, connection_string: str, model: str = "gpt-4-turbo"):
        """Initialize the database service."""
        if not connection_string:
            raise ValueError("Database connection string is required.")
        self.connection_string = connection_string
        self.model = model
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.setup_database()
        self.setup_sql_agent()
        
    def setup_database(self):
        """Set up the database connection."""
        try:
            self.engine = create_engine(self.connection_string)
            self.db = SQLDatabase.from_uri(self.connection_string)
            # Extract database name for logging if possible, otherwise log generic message
            try:
                db_name = self.engine.url.database
                self.logger.info(f"Connected to database: {db_name}")
            except Exception:
                self.logger.info("Connected to database using provided connection string.")
        except Exception as e:
            self.logger.error(f"Error connecting to database: {e}")
            raise
            
    def setup_sql_agent(self):
        """Set up the SQL agent for natural language queries."""
        try:
            llm = ChatOpenAI(model=self.model)
            tools = [QuerySQLDatabaseTool(db=self.db)]
            self.sql_agent = create_sql_agent(
                llm=llm,
                db=self.db,
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True
            )
            self.logger.info("SQL agent initialized successfully")
        except Exception as e:
            self.logger.error(f"Error setting up SQL agent: {e}")
            raise
            
    def execute_query(self, query: str, params: Optional[List[Any]] = None) -> List[Dict[str, Any]]:
        """Execute a SQL query and return the results."""
        try:
            self.logger.debug(f"Executing SQL query: {query}")
            with self.engine.connect() as connection:
                result = connection.execute(text(query), params if params else {})
                rows = [row._asdict() for row in result]
                self.logger.debug(f"Query returned {len(rows)} rows.")
                return rows
        except Exception as e:
            self.logger.error(f"Error executing query: {e}")
            raise
            
    def get_medicine_info(self, medicine_id: Optional[int] = None, 
                         medicine_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get medicine information from the database."""
        try:
            query = "SELECT * FROM [dbo].[Medicines]"
            if medicine_id:
                query += f" WHERE MedicineID = {medicine_id}"
            elif medicine_name:
                query += f" WHERE Name LIKE '%{medicine_name}%'"
                
            return self.execute_query(query)
        except Exception as e:
            self.logger.error(f"Error getting medicine info: {e}")
            raise
            
    def get_store_info(self, store_id: Optional[int] = None, 
                      location: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get store information from the database."""
        try:
            query = """
                SELECT StoreID, StoreName, InventoryTableName, Location, Address
                FROM dbo.Stores
                WHERE 1=1
            """
            params = []
            
            if store_id:
                query += " AND StoreID = ?"
                params.append(store_id)
            if location:
                query += " AND Location LIKE ?"
                params.append(f"%{location}%")
                
            return self.execute_query(query, params)
        except Exception as e:
            self.logger.error(f"Error getting store info: {e}")
            return []
            
    def process_natural_language_query(self, query: str) -> str:
        """Process a natural language query using the SQL agent."""
        try:
            result = self.sql_agent.invoke({"input": query})
            return result.get("output", "No results found.")
        except Exception as e:
            self.logger.error(f"Error processing natural language query: {e}")
            raise 