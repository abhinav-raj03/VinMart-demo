import sqlite3
import os
# Ensure logs/ directory exists
os.makedirs("logs", exist_ok=True)
# Initialize database if it doesn't exist
def init_db():
    conn = sqlite3.connect("logs/detection_log.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS detections (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 bottle_id INTEGER,
                 class_id INTEGER,
                 timestamp TEXT,
                 center_x INTEGER,
                 center_y INTEGER
                 )''')
    conn.commit()
    conn.close()

# Log each detection
def log_detection(bottle_id, class_id, timestamp, center_x, center_y):
    conn = sqlite3.connect("logs/detection_log.db")
    c = conn.cursor()
    c.execute("INSERT INTO detections (bottle_id, class_id, timestamp, center_x, center_y) VALUES (?, ?, ?, ?, ?)",
              (bottle_id, class_id, timestamp, center_x, center_y))
    conn.commit()
    conn.close()

# Initialize database on import
init_db()
