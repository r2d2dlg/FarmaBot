import sqlite3
import os

def reset_database():
    # Remove existing database
    if os.path.exists('farmabot.db'):
        os.remove('farmabot.db')
    
    conn = sqlite3.connect('farmabot.db')
    cursor = conn.cursor()
    
    try:
        # Create tables
        cursor.execute("""
        CREATE TABLE Medicines (
            MedicineID INTEGER PRIMARY KEY,
            [Generic Name] TEXT NOT NULL,
            [Brand Name 1] TEXT,
            [Brand Name 2] TEXT,
            Description TEXT,
            [Side Effects] TEXT,
            [Requires Prescription] INTEGER DEFAULT 0
        );
        """)
        
        cursor.execute("""
        CREATE TABLE Stores (
            StoreID INTEGER PRIMARY KEY,
            Name TEXT NOT NULL,
            Address TEXT,
            OpeningHours TEXT,
            PhoneNumber TEXT
        );
        """)
        
        cursor.execute("""
        CREATE TABLE Inventory (
            StoreID INTEGER,
            MedicineID INTEGER,
            Quantity INTEGER DEFAULT 0,
            PRIMARY KEY (StoreID, MedicineID),
            FOREIGN KEY (StoreID) REFERENCES Stores(StoreID),
            FOREIGN KEY (MedicineID) REFERENCES Medicines(MedicineID)
        );
        """)
        
        # Insert stores
        cursor.execute("""
        INSERT INTO Stores (Name, Address, OpeningHours, PhoneNumber) VALUES
            ('Chorrera', 'Calle Principal, La Chorrera', '8:00 AM - 10:00 PM', '+507-123-4567'),
            ('Costa del Este', 'Ave. Principal Costa del Este', '24 hours', '+507-234-5678'),
            ('David', 'Calle Central, David, Chiriquí', '7:00 AM - 11:00 PM', '+507-345-6789'),
            ('El Dorado', 'Centro Comercial El Dorado', '8:00 AM - 9:00 PM', '+507-456-7890'),
            ('San Francisco', 'Calle 50, San Francisco', '24 hours', '+507-567-8901');
        """)
        
        # Insert Valium
        cursor.execute("""
        INSERT INTO Medicines ([Generic Name], [Brand Name 1], Description, [Side Effects], [Requires Prescription])
        VALUES (
            'Diazepam',
            'Valium',
            'Anti-anxiety medication that belongs to a class of drugs called benzodiazepines',
            'May cause drowsiness, dizziness, memory problems, and coordination issues',
            1
        );
        """)
        
        # Add inventory (10 units per store)
        cursor.execute("""
        INSERT INTO Inventory (StoreID, MedicineID, Quantity)
        SELECT StoreID, 1, 10
        FROM Stores;
        """)
        
        conn.commit()
        print("Database reset successfully!")
        
        # Verify Valium inventory
        cursor.execute("""
        SELECT s.Name, i.Quantity
        FROM Inventory i
        JOIN Medicines m ON i.MedicineID = m.MedicineID
        JOIN Stores s ON i.StoreID = s.StoreID
        WHERE m.[Generic Name] = 'Diazepam'
        ORDER BY s.Name;
        """)
        
        results = cursor.fetchall()
        print("\nValium Inventory:")
        print("================")
        for store, quantity in results:
            print(f"Store: {store}, Quantity: {quantity}")
            
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    reset_database() 