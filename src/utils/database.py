import sqlite3
import json
import os
from datetime import datetime

DATABASE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "library.db")

def init_db():
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS papers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            source TEXT,
            paper_id TEXT,
            summary_text TEXT,
            bullets TEXT,
            insights TEXT,
            stats TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def save_paper(title, source, paper_id, summary_text, bullets, insights, stats):
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO papers (title, source, paper_id, summary_text, bullets, insights, stats)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (title, source, paper_id, summary_text, json.dumps(bullets), json.dumps(insights), json.dumps(stats)))
    conn.commit()
    conn.close()

def get_all_papers():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM papers ORDER BY timestamp DESC')
    rows = cursor.fetchall()
    conn.close()
    
    papers = []
    for row in rows:
        paper = dict(row)
        paper['bullets'] = json.loads(paper['bullets'])
        paper['insights'] = json.loads(paper['insights'])
        paper['stats'] = json.loads(paper['stats'])
        papers.append(paper)
    return papers

def delete_paper(paper_id):
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM papers WHERE id = ?', (paper_id,))
    conn.commit()
    conn.close()
