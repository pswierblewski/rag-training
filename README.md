# RAG Training API

A RAG (Retrieval-Augmented Generation) solution using FastAPI, PostgreSQL, LlamaIndex, and FAISS for Polish sentence retrieval with VoyageAI embeddings and OpenAI LLM integration.

## Features

- **FAISS Vector Store**: Efficient similarity search for Polish sentences
- **LlamaIndex Framework**: Powerful RAG orchestration
- **VoyageAI Embeddings**: High-quality embeddings with voyage-3-large model
- **OpenAI LLM**: GPT-4o for intelligent query answering
- **PostgreSQL Database**: Persistent storage for Polish sentences
- **FastAPI**: Modern, fast web framework with automatic API documentation

## Prerequisites

- Python 3.8+
- PostgreSQL database (running locally)
- OpenAI API key (for LLM)
- VoyageAI API key (for embeddings)

## Installation

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

Copy `.env.example` to `.env` and fill in your configuration:

```bash
cp .env.example .env
```

Edit `.env` with your settings:
```
OPENAI_API_KEY=your_openai_api_key_here
VOYAGE_API_KEY=your_voyage_api_key_here
DB_HOST=localhost
DB_PORT=5432
DB_NAME=rag_db_training
DB_USER=postgres
DB_PASSWORD=your_password
```

### 4. Initialize Database

Run the Python initialization script to create the database and populate it with Polish sentences:

```bash
python init_db.py
```

The script will:
- Create the `rag_db_training` database (if it doesn't exist)
- Create the `sentences` table
- Insert 65 Polish sample sentences
- Provide clear feedback on success/errors

## Usage

### Start the Server

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

### API Documentation

Interactive API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## API Endpoints

### 1. Initialize Vector Store

**POST** `/init`

Creates and persists the FAISS vector store from Polish sentences in the database.

**Response:**
```json
{
  "message": "FAISS vector store initialized successfully",
  "sentences_indexed": 65
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/init
```

### 2. Chat - Query the RAG System

**POST** `/chat`

Queries the RAG system with a Polish question and returns relevant sentences.

**Request Body:**
```json
{
  "question": "Jak dojść do dworca kolejowego?"
}
```

**Response:**
```json
{
  "sentence_ids": [12],
  "sentences": ["Jak dojść do dworca kolejowego?"]
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Gdzie mogę kupić kawę?"}'
```

### 3. Health Check

**GET** `/health`

Checks the health status of the API and its dependencies.

**Response:**
```json
{
  "status": "healthy",
  "database": "connected",
  "faiss_index": "initialized"
}
```

## How It Works

### Init Endpoint Flow

1. Fetches all Polish sentences from PostgreSQL database
2. Creates LlamaIndex documents with sentence IDs as metadata
3. Generates embeddings using VoyageAI's `voyage-3-large` model
4. Builds FAISS vector store for efficient similarity search
5. Persists the index to local disk (`./faiss_index/`)

### Chat Endpoint Flow

1. Loads FAISS vector store from disk
2. Creates a node retriever from the FAISS index
3. Retrieves top-k most relevant sentences (default: 5)
4. Formats results as a markdown table with sentence IDs and sentences
5. Sends structured prompt to OpenAI LLM (GPT-3.5-turbo)
6. LLM returns sentence IDs that answer the question (using Pydantic model)
7. Fetches original sentences from PostgreSQL by IDs
8. Returns matching sentences to the user

## Project Structure

```
rag-training/
├── main.py              # FastAPI application with endpoints
├── models.py            # Pydantic models
├── database.py          # PostgreSQL connection and queries
├── config.py            # Configuration management
├── requirements.txt     # Python dependencies
├── init_db.sql         # Database initialization script
├── .env.example        # Environment variables template
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

## Example Usage Scenarios

### Scenario 1: Initialize the System

```bash
# 1. Start the server
uvicorn main:app --reload

# 2. Initialize FAISS index
curl -X POST http://localhost:8000/init
```

### Scenario 2: Ask Questions in Polish

```bash
# Question about coffee
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Gdzie mogę kupić kawę?"}'

# Question about location
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Gdzie jest Warszawa?"}'

# Question about hobbies
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Co lubisz robić w weekendy?"}'
```

## Configuration Options

All configuration options can be set in the `.env` file:

- `OPENAI_API_KEY`: Your OpenAI API key for LLM (required)
- `VOYAGE_API_KEY`: Your VoyageAI API key for embeddings (required)
- `DB_HOST`: PostgreSQL host (default: localhost)
- `DB_PORT`: PostgreSQL port (default: 5432)
- `DB_NAME`: Database name (default: rag_db_training)
- `DB_USER`: Database user (default: postgres)
- `DB_PASSWORD`: Database password (default: postgres)
- `FAISS_INDEX_PATH`: Path to store FAISS index (default: ./faiss_index)
- `LLM_MODEL`: OpenAI model to use (default: gpt-4o)
- `EMBEDDING_MODEL`: VoyageAI embedding model (default: voyage-3-large)
- `TOP_K_RESULTS`: Number of results to retrieve (default: 5)

## Troubleshooting

### Database Connection Issues

If you get database connection errors:
1. Ensure PostgreSQL is running
2. Verify credentials in `.env`
3. Run `python init_db.py` to initialize the database
4. Test connection with the health endpoint: `curl http://localhost:8000/health`

### FAISS Index Not Found

If you get "FAISS index not found" error:
1. Call the `/init` endpoint first to create the index
2. Verify `./faiss_index/` directory was created
3. Check file permissions

### OpenAI API Errors

If you get OpenAI-related errors:
1. Verify your API key is correct in `.env`
2. Check your OpenAI account has available credits
3. Ensure you have access to the specified models

## Development

### Running in Development Mode

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Debugging in VS Code

The project includes VS Code debug configurations in `.vscode/launch.json`:

1. **FastAPI: Debug Server** - Debug with auto-reload (recommended for development)
   - Press `F5` or use Run & Debug panel
   - Set breakpoints in your code
   - Server restarts automatically on file changes

2. **FastAPI: Debug Server (No Reload)** - Debug without auto-reload
   - Better for debugging startup issues
   - Breakpoints won't be missed during reload

3. **Python: Current File** - Debug any Python file
   - Open the file you want to debug
   - Press `F5` to run it

4. **Python: Init Database** - Debug the database initialization script
   - Useful for debugging database setup issues

**To start debugging:**
1. Open the Run and Debug panel (`Ctrl+Shift+D`)
2. Select "FastAPI: Debug Server" from the dropdown
3. Press `F5` or click the green play button
4. Set breakpoints by clicking left of line numbers
5. Make requests to your endpoints to trigger breakpoints

### Running Tests

```bash
# Health check
curl http://localhost:8000/health

# Test database connection
python -c "import database; print(database.test_connection())"
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
