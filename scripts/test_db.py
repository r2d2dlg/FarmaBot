import os
import sys
# Ensure project root is in sys.path for relative imports
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)
from dotenv import load_dotenv
from services.database_service import DatabaseService

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path)

# Retrieve connection string from environment
conn_str = os.getenv("DB_CONNECTION_STRING")
if not conn_str:
    print("ERROR: DB_CONNECTION_STRING is not set in .env")
    exit(1)

# Initialize database service
db_path = conn_str.replace("sqlite:///", "")
db = DatabaseService(db_path=db_path)

# Fetch and display store info
stores = db.get_store_info()
print("Stores:")
for store in stores:
    print(f"- {store['Location']} (Table: {store['InventoryTableName']})")

# Fetch sample rows from each inventory table
print("\nSample inventory rows:")
for store in stores:
    table = store['InventoryTableName']
    try:
        rows = db.execute_query(f"SELECT TOP 3 * FROM dbo.[{table}]")
        print(f"\n[{table}]")
        for row in rows:
            print(row)
    except Exception as e:
        print(f"Error querying {table}: {e}") 