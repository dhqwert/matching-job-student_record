"""
db.py — psycopg2 connection helper for ai_service.
Each worker creates its own connection (no shared state across threads).
"""
import psycopg2
from config import DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_DATABASE


def get_connection():
    """Return a new psycopg2 connection. Caller is responsible for closing it."""
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        dbname=DB_DATABASE,
    )
