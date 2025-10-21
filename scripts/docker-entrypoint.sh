#!/bin/bash
set -e

echo "=========================================="
echo "RAG Training API - Docker Entrypoint"
echo "=========================================="
echo ""

# Configuration
MAX_RETRIES=30
RETRY_INTERVAL=2

# Function to wait for PostgreSQL
wait_for_postgres() {
    echo "Waiting for PostgreSQL to be ready..."
    
    retries=0
    until pg_isready -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" > /dev/null 2>&1; do
        retries=$((retries + 1))
        if [ $retries -ge $MAX_RETRIES ]; then
            echo "ERROR: PostgreSQL is not available after $MAX_RETRIES attempts"
            exit 1
        fi
        echo "  Attempt $retries/$MAX_RETRIES: PostgreSQL is unavailable - waiting ${RETRY_INTERVAL}s..."
        sleep $RETRY_INTERVAL
    done
    
    echo "✓ PostgreSQL is ready!"
}

# Function to check if database is initialized
check_database_initialized() {
    echo "Checking if database is initialized..."
    
    # Try to connect and check if sentences table exists
    if PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" \
        -c "SELECT 1 FROM sentences LIMIT 1;" > /dev/null 2>&1; then
        echo "✓ Database is already initialized"
        return 0
    else
        echo "  Database needs initialization"
        return 1
    fi
}

# Function to initialize database
initialize_database() {
    echo "Initializing database..."
    
    # Run the Python initialization script
    python init_db.py
    
    if [ $? -eq 0 ]; then
        echo "✓ Database initialized successfully"
    else
        echo "ERROR: Database initialization failed"
        exit 1
    fi
}

# Main execution
echo "Configuration:"
echo "  DB Host: $DB_HOST"
echo "  DB Port: $DB_PORT"
echo "  DB Name: $DB_NAME"
echo "  DB User: $DB_USER"
echo "  LLM Model: $LLM_MODEL"
echo "  Embedding Model: $EMBEDDING_MODEL"
echo "  Log Level: $LOG_LEVEL"
echo "  Workers: ${WORKERS:-1}"
echo ""

# Wait for PostgreSQL
wait_for_postgres

# Check and initialize database if needed
if ! check_database_initialized; then
    initialize_database
fi

echo ""
echo "=========================================="
echo "Starting FastAPI Application"
echo "=========================================="
echo ""

# Execute the main command (passed as arguments to the script)
exec "$@"

