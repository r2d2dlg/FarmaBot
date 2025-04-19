import sqlite3

def cleanup_duplicates():
    conn = sqlite3.connect('farmabot.db')
    cursor = conn.cursor()
    
    try:
        # Remove duplicate medicines
        cursor.execute("""
        DELETE FROM Medicines
        WHERE MedicineID NOT IN (
            SELECT MIN(MedicineID)
            FROM Medicines
            GROUP BY [Generic Name], [Brand Name 1]
        );
        """)
        
        # Remove duplicate inventory entries
        cursor.execute("""
        DELETE FROM Inventory
        WHERE rowid NOT IN (
            SELECT MIN(rowid)
            FROM Inventory
            GROUP BY StoreID, MedicineID
        );
        """)
        
        conn.commit()
        print("Duplicates removed successfully!")
        
        # Verify Valium inventory
        cursor.execute("""
        SELECT s.Name, i.Quantity
        FROM Inventory i
        JOIN Medicines m ON i.MedicineID = m.MedicineID
        JOIN Stores s ON i.StoreID = s.StoreID
        WHERE m.[Generic Name] = 'Diazepam' OR m.[Brand Name 1] = 'Valium'
        ORDER BY s.Name;
        """)
        
        results = cursor.fetchall()
        print("\nValium Inventory (After Cleanup):")
        print("==============================")
        for store, quantity in results:
            print(f"Store: {store}, Quantity: {quantity}")
            
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    cleanup_duplicates() 