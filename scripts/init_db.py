import os
import sqlite3
from pathlib import Path

def init_database():
    # Get the root directory (where farmabot.db should be)
    root_dir = Path(__file__).parent.parent
    db_path = root_dir / "farmabot.db"
    sql_path = root_dir / "setup_database.sql"
    
    # Create database connection
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Read and execute the SQL script
        with open(sql_path, 'r') as sql_file:
            sql_script = sql_file.read()
            # Split the script into individual statements
            statements = sql_script.split(';')
            for statement in statements:
                if statement.strip():
                    cursor.execute(statement)
        
        # Commit the changes
        conn.commit()
        print(f"Database initialized successfully at {db_path}")
        
        # Verify the data
        cursor.execute("SELECT COUNT(*) FROM Medicines")
        medicine_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM Stores")
        store_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM Inventory")
        inventory_count = cursor.fetchone()[0]
        
        print(f"\nVerification:")
        print(f"- Medicines: {medicine_count}")
        print(f"- Stores: {store_count}")
        print(f"- Inventory items: {inventory_count}")
        
    except Exception as e:
        print(f"Error initializing database: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    init_database() 