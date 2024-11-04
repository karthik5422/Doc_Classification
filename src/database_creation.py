import sqlite3
import os

DATABASE = "meta_data.db"

def init_db():
    """Initialize the SQLite database and create the metadata table."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            class TEXT NOT NULL,
            metadata TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()
    print(f"Database initialized at {DATABASE}")

def insert_metadata(class_label, metadata):
    """Insert metadata into the database."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    cursor.execute("INSERT INTO metadata (class, metadata) VALUES (?, ?)", (class_label, str(metadata)))
    
    conn.commit()
    conn.close()
    print(f"Inserted metadata for class '{class_label}'")

if __name__ == "__main__":
    init_db()  
