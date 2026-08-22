import sqlite3
import pandas as pd
import os

DB_FILENAME = 'legal_docs.db'

def init_db():
    conn = sqlite3.connect(DB_FILENAME)
    cursor = conn.cursor()
    
    # Create tables if not exists
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_name TEXT,
            word_count INTEGER,
            sentence_count INTEGER,
            chunk_count INTEGER,
            mean_chunk_len REAL,
            median_chunk_len REAL,
            std_chunk_len REAL,
            mean_conf REAL,
            median_conf REAL,
            std_conf REAL,
            mean_uncertainty REAL,
            median_uncertainty REAL,
            std_uncertainty REAL,
            summary TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS clauses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id INTEGER,
            clause_text TEXT,
            category TEXT,
            confidence REAL,
            uncertainty REAL,
            FOREIGN KEY (doc_id) REFERENCES documents(id)
        )
    ''')
    
    # Schema Migration Check: Ensure new columns exist
    cursor.execute("PRAGMA table_info(documents)")
    columns = [info[1] for info in cursor.fetchall()]
    new_columns = {
        "mean_uncertainty": "REAL",
        "median_uncertainty": "REAL",
        "std_uncertainty": "REAL"
    }
    
    for col, col_type in new_columns.items():
        if col not in columns:
            cursor.execute(f"ALTER TABLE documents ADD COLUMN {col} {col_type}")
            
    conn.commit()
    conn.close()

def save_document(doc_name, stats, obj_stats, summary):
    conn = sqlite3.connect(DB_FILENAME)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO documents (
            doc_name, word_count, sentence_count, chunk_count,
            mean_chunk_len, median_chunk_len, std_chunk_len,
            mean_conf, median_conf, std_conf,
            mean_uncertainty, median_uncertainty, std_uncertainty,
            summary
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        doc_name, stats['word_count'], stats['sentence_count'], stats['chunk_count'],
        stats['mean_chunk_len'], stats['median_chunk_len'], stats['std_chunk_len'],
        obj_stats['mean_conf'], obj_stats['median_conf'], obj_stats['std_conf'],
        obj_stats['mean_uncertainty'], obj_stats['median_uncertainty'], obj_stats['std_uncertainty'],
        summary
    ))
    doc_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return doc_id

def save_clauses(doc_id, clauses):
    conn = sqlite3.connect(DB_FILENAME)
    cursor = conn.cursor()
    for c in clauses:
        cursor.execute('''
            INSERT INTO clauses (doc_id, clause_text, category, confidence, uncertainty)
            VALUES (?, ?, ?, ?, ?)
        ''', (doc_id, c['clause'], c['category'], c['confidence'], c['uncertainty']))
    conn.commit()
    conn.close()

def get_documents_df():
    conn = sqlite3.connect(DB_FILENAME)
    df = pd.read_sql_query("SELECT * FROM documents", conn)
    conn.close()
    return df

def get_clauses_df(doc_id):
    conn = sqlite3.connect(DB_FILENAME)
    df = pd.read_sql_query("SELECT * FROM clauses WHERE doc_id = ?", conn, params=(doc_id,))
    conn.close()
    return df

def delete_document_by_id(doc_id):
    """
    Deletes a document and its associated clauses from the SQLite database.
    """
    conn = sqlite3.connect(DB_FILENAME)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM clauses WHERE doc_id = ?", (doc_id,))
    cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
    conn.commit()
    conn.close()
