#!/bin/bash

# RAG Training Setup Script

echo "Setting up RAG Training environment..."

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Copy environment file
if [ ! -f .env ]; then
    echo "Creating .env file from env.example..."
    cp env.example .env
    echo "Please edit .env file with your configuration"
else
    echo ".env file already exists"
fi

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Edit .env file with your API keys and PostgreSQL credentials"
echo "3. Initialize the database: python init_db.py"
echo "4. Start the server: uvicorn main:app --reload"
echo "5. Initialize FAISS index: curl -X POST http://localhost:8000/init"



