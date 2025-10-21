#!/usr/bin/env python3
"""
Database initialization script for RAG Training API.

This script creates the PostgreSQL database and populates it with Polish sentences.
It reads database credentials from the .env file.
"""

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import logging
import config
import sys

# Initialize logger
logger = logging.getLogger(__name__)


POLISH_SENTENCES = [
    'Dzień dobry, jak się masz?',
    'Pogoda dzisiaj jest piękna i słoneczna.',
    'Kocham czytać książki w wolnym czasie.',
    'Polska to kraj w Europie Środkowej.',
    'Warszawa jest stolicą Polski.',
    'Lubię pić kawę rano.',
    'Gdzie jest najbliższa apteka?',
    'Uczę się programowania w Pythonie.',
    'Kolacja jest gotowa o godzinie osiemnastej.',
    'Mój ulubiony kolor to niebieski.',
    'W weekendy lubię jeździć na rowerze.',
    'Jak dojść do dworca kolejowego?',
    'Jutro mam ważne spotkanie w biurze.',
    'Moja rodzina pochodzi z Krakowa.',
    'Chciałbym nauczyć się grać na gitarze.',
    'Sklep jest otwarty od poniedziałku do soboty.',
    'Lubię oglądać filmy w kinie.',
    'Czy możesz mi pomóc z tym problemem?',
    'Zima w Polsce może być bardzo mroźna.',
    'Studiuję informatykę na uniwersytecie.',
    'Mój pies nazywa się Max.',
    'Wolę herbatę od kawy.',
    'Kraków to jedno z najpiękniejszych miast w Polsce.',
    'Czy możesz podać mi sól?',
    'Moja siostra pracuje jako lekarz.',
    'Lubię gotować włoskie dania.',
    'W czerwcu jadę na wakacje nad morze.',
    'Potrzebuję kupić nowy telefon.',
    'Uczę się języka angielskiego.',
    'Moja córka idzie do szkoły podstawowej.',
    'Pracuję jako programista w firmie IT.',
    'Czy ten autobus jedzie do centrum?',
    'Wieczorem lubię czytać książki.',
    'Mój brat studiuje medycynę.',
    'Czy mógłbyś zamknąć okno?',
    'Lubię jeść pierogi z kapustą i grzybami.',
    'W Polsce mamy cztery pory roku.',
    'Jutro będzie padać deszcz.',
    'Muszę iść do lekarza na badania.',
    'Czy ten sklep przyjmuje karty kredytowe?',
    'Mój samochód jest w warsztacie.',
    'Lubię słuchać muzyki klasycznej.',
    'Wieczorem idę na trening fitness.',
    'Moja mama gotuje najlepsze zupy.',
    'Czy możesz mi powiedzieć, która godzina?',
    'W sobotę spotykam się z przyjaciółmi.',
    'Muszę zapłacić rachunki za prąd.',
    'Lubię spacerować po parku.',
    'Mój ulubiony pisarz to Henryk Sienkiewicz.',
    'Czy możesz mi polecić dobrą restaurację?',
    'Pracuję zdalnie z domu.',
    'Moja babcia mieszka na wsi.',
    'Lubię jeździć na nartach zimą.',
    'Czy ten hotel ma Wi-Fi?',
    'Muszę kupić bilety na koncert.',
    'Studiuję sztuczną inteligencję.',
    'Mój kot lubi spać na kanapie.',
    'Czy możesz mówić wolniej?',
    'Lubię zwiedzać muzea i galerie sztuki.',
    'W Polsce mamy wiele pięknych zamków.',
    'Muszę oddać książki do biblioteki.',
    'Czy mógłbyś wyłączyć światło?',
    'Mój dziadek był inżynierem.',
    'Lubię pływać w basenie.',
    'Czy możesz mi pokazać drogę?'
]


def create_database():
    """Create the rag_db_training database if it doesn't exist."""
    print(f"Connecting to PostgreSQL at {config.DB_HOST}:{config.DB_PORT}...")
    logger.info(f"Attempting to connect to PostgreSQL at {config.DB_HOST}:{config.DB_PORT}")
    
    try:
        # Connect to default postgres database to create our database
        logger.debug("Connecting to 'postgres' database...")
        conn = psycopg2.connect(
            host=config.DB_HOST,
            port=config.DB_PORT,
            dbname='postgres',
            user=config.DB_USER,
            password=config.DB_PASSWORD
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        logger.debug("Connected successfully")
        
        # Check if database exists
        logger.debug(f"Checking if database '{config.DB_NAME}' exists...")
        cursor.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s",
            (config.DB_NAME,)
        )
        exists = cursor.fetchone()
        
        if exists:
            print(f"✓ Database '{config.DB_NAME}' already exists")
            logger.info(f"Database '{config.DB_NAME}' already exists")
        else:
            # Create database
            logger.info(f"Creating database '{config.DB_NAME}'...")
            cursor.execute(f'CREATE DATABASE {config.DB_NAME}')
            print(f"✓ Created database '{config.DB_NAME}'")
            logger.info(f"Database '{config.DB_NAME}' created successfully")
        
        cursor.close()
        conn.close()
        logger.debug("Connection closed")
        return True
        
    except psycopg2.Error as e:
        print(f"✗ Error creating database: {e}")
        logger.error(f"Error creating database: {e}", exc_info=True)
        return False


def create_table_and_insert_data():
    """Create the sentences table and insert Polish sentences."""
    try:
        # Connect to our newly created database
        print(f"\nConnecting to database '{config.DB_NAME}'...")
        logger.info(f"Connecting to database '{config.DB_NAME}'...")
        conn = psycopg2.connect(
            host=config.DB_HOST,
            port=config.DB_PORT,
            dbname=config.DB_NAME,
            user=config.DB_USER,
            password=config.DB_PASSWORD
        )
        cursor = conn.cursor()
        logger.debug("Connected to database successfully")
        
        # Check if table exists
        logger.debug("Checking if 'sentences' table exists...")
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'sentences'
            )
        """)
        table_exists = cursor.fetchone()[0]
        
        if table_exists:
            # Check if table has data
            cursor.execute("SELECT COUNT(*) FROM sentences")
            count = cursor.fetchone()[0]
            
            if count > 0:
                print(f"✓ Table 'sentences' already exists with {count} records")
                logger.info(f"Table 'sentences' exists with {count} records")
                response = input("  Do you want to drop and recreate it? (y/N): ")
                if response.lower() != 'y':
                    print("  Skipping table creation and data insertion")
                    logger.info("User chose to skip table recreation")
                    cursor.close()
                    conn.close()
                    return True
                
                # Drop existing table
                logger.warning("Dropping existing 'sentences' table")
                cursor.execute("DROP TABLE sentences")
                print("  Dropped existing table")
        
        # Create sentences table
        print("Creating 'sentences' table...")
        logger.info("Creating 'sentences' table...")
        cursor.execute("""
            CREATE TABLE sentences (
                id SERIAL PRIMARY KEY,
                sentence TEXT NOT NULL
            )
        """)
        print("✓ Created table 'sentences'")
        logger.info("Table 'sentences' created successfully")
        
        # Insert Polish sentences
        print(f"\nInserting {len(POLISH_SENTENCES)} Polish sentences...")
        logger.info(f"Inserting {len(POLISH_SENTENCES)} sentences...")
        for i, sentence in enumerate(POLISH_SENTENCES, 1):
            cursor.execute(
                "INSERT INTO sentences (sentence) VALUES (%s)",
                (sentence,)
            )
            if i % 20 == 0:
                logger.debug(f"Inserted {i}/{len(POLISH_SENTENCES)} sentences")
        
        conn.commit()
        print(f"✓ Inserted {len(POLISH_SENTENCES)} sentences")
        logger.info(f"Successfully inserted {len(POLISH_SENTENCES)} sentences")
        
        # Verify
        logger.debug("Verifying inserted data...")
        cursor.execute("SELECT COUNT(*) FROM sentences")
        count = cursor.fetchone()[0]
        print(f"✓ Verified: {count} sentences in database")
        logger.info(f"Verification successful: {count} sentences in database")
        
        cursor.close()
        conn.close()
        logger.debug("Connection closed")
        return True
        
    except psycopg2.Error as e:
        print(f"✗ Error creating table or inserting data: {e}")
        logger.error(f"Error creating table or inserting data: {e}", exc_info=True)
        return False


def main():
    """Main function to initialize the database."""
    print("=" * 60)
    print("RAG Training Database Initialization")
    print("=" * 60)
    print()
    print(f"Database: {config.DB_NAME}")
    print(f"Host: {config.DB_HOST}:{config.DB_PORT}")
    print(f"User: {config.DB_USER}")
    print()
    
    logger.info("=" * 60)
    logger.info("Starting database initialization")
    logger.info(f"Target database: {config.DB_NAME} at {config.DB_HOST}:{config.DB_PORT}")
    
    # Step 1: Create database
    logger.info("Step 1: Creating database")
    if not create_database():
        print("\n✗ Failed to create database")
        logger.error("Database creation failed")
        sys.exit(1)
    
    # Step 2: Create table and insert data
    logger.info("Step 2: Creating table and inserting data")
    if not create_table_and_insert_data():
        print("\n✗ Failed to create table or insert data")
        logger.error("Table creation or data insertion failed")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("✓ Database initialization completed successfully!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Start the API server: uvicorn main:app --reload")
    print("2. Initialize FAISS index: curl -X POST http://localhost:8000/init")
    print("3. Test chat endpoint: curl -X POST http://localhost:8000/chat \\")
    print("     -H 'Content-Type: application/json' \\")
    print("     -d '{\"question\": \"Gdzie jest Warszawa?\"}'")
    print()
    
    logger.info("=" * 60)
    logger.info("Database initialization completed successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

