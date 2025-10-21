import psycopg2
import logging
from typing import List, Dict, Optional
from contextlib import contextmanager
import config

# Initialize logger
logger = logging.getLogger(__name__)


@contextmanager
def get_db_connection():
    """Context manager for database connections."""
    logger.debug(f"Connecting to database {config.DB_NAME} at {config.DB_HOST}:{config.DB_PORT}")
    conn = psycopg2.connect(
        host=config.DB_HOST,
        port=config.DB_PORT,
        dbname=config.DB_NAME,
        user=config.DB_USER,
        password=config.DB_PASSWORD
    )
    logger.debug("Database connection established")
    try:
        yield conn
    finally:
        logger.debug("Closing database connection")
        conn.close()


def fetch_all_sentences() -> List[Dict[str, any]]:
    """
    Fetch all sentences from the database.
    
    Returns:
        List of dictionaries with 'id' and 'sentence' keys
    """
    logger.debug("Fetching all sentences from database")
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, sentence FROM sentences ORDER BY id")
        rows = cursor.fetchall()
        cursor.close()
        
        result = [{"id": row[0], "sentence": row[1]} for row in rows]
        logger.debug(f"Fetched {len(result)} sentences from database")
        return result


def fetch_sentences_by_ids(sentence_ids: List[int]) -> List[Dict[str, any]]:
    """
    Fetch sentences by their IDs.
    
    Args:
        sentence_ids: List of sentence IDs to fetch
        
    Returns:
        List of dictionaries with 'id' and 'sentence' keys
    """
    if not sentence_ids:
        logger.debug("No sentence IDs provided, returning empty list")
        return []
    
    logger.debug(f"Fetching {len(sentence_ids)} sentences by IDs: {sentence_ids}")
    with get_db_connection() as conn:
        cursor = conn.cursor()
        # Use parameterized query to prevent SQL injection
        query = "SELECT id, sentence FROM sentences WHERE id = ANY(%s) ORDER BY id"
        cursor.execute(query, (sentence_ids,))
        rows = cursor.fetchall()
        cursor.close()
        
        result = [{"id": row[0], "sentence": row[1]} for row in rows]
        logger.debug(f"Retrieved {len(result)} sentences from database")
        return result


def test_connection() -> bool:
    """
    Test database connection.
    
    Returns:
        True if connection successful, False otherwise
    """
    logger.debug("Testing database connection")
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            logger.info("Database connection test successful")
            return True
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        return False



