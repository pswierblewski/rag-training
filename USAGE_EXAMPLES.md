# Usage Examples

This document contains practical examples of using the RAG Training API.

## Prerequisites

1. Virtual environment is activated
2. PostgreSQL is running with the `rag_db_training` database
3. `.env` file is configured with your OpenAI and VoyageAI API keys
4. Server is running: `uvicorn main:app --reload`

## Setup and Initialization

### 1. Initialize the Database

```bash
# Run the Python initialization script
python init_db.py
```

The script will:
- Create the database and table
- Insert 65 Polish sentences
- Show clear progress messages

Expected output: `✓ Database initialization completed successfully!`

### 2. Start the API Server

```bash
# Activate virtual environment
source venv/bin/activate

# Start the server
uvicorn main:app --reload
```

The server will be available at `http://localhost:8000`

### 3. Initialize FAISS Vector Store

```bash
curl -X POST http://localhost:8000/init
```

Expected response:
```json
{
  "message": "FAISS vector store initialized successfully",
  "sentences_indexed": 65
}
```

## Query Examples

### Example 1: Finding Information About Coffee

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Gdzie mogę kupić kawę?"}'
```

Expected response:
```json
{
  "sentence_ids": [6],
  "sentences": ["Lubię pić kawę rano."]
}
```

### Example 2: Asking About Location

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Gdzie jest Warszawa?"}'
```

Expected response:
```json
{
  "sentence_ids": [5],
  "sentences": ["Warszawa jest stolicą Polski."]
}
```

### Example 3: Finding Programming-Related Content

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Czy ktoś uczy się programowania?"}'
```

Expected response:
```json
{
  "sentence_ids": [8, 20, 31, 56],
  "sentences": [
    "Uczę się programowania w Pythonie.",
    "Studiuję informatykę na uniwersytecie.",
    "Pracuję jako programista w firmie IT.",
    "Studiuję sztuczną inteligencję."
  ]
}
```

### Example 4: Finding Hobby-Related Content

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Jakie są hobby ludzi?"}'
```

Expected response:
```json
{
  "sentence_ids": [3, 11, 17, 33, 42, 48, 53, 59, 65],
  "sentences": [
    "Kocham czytać książki w wolnym czasie.",
    "W weekendy lubię jeździć na rowerze.",
    "Lubię oglądać filmy w kinie.",
    "Wieczorem lubię czytać książki.",
    "Lubię słuchać muzyki klasycznej.",
    "Lubię spacerować po parku.",
    "Lubię jeździć na nartach zimą.",
    "Lubię zwiedzać muzea i galerie sztuki.",
    "Lubię pływać w basenie."
  ]
}
```

### Example 5: Transportation Question

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Jak dostać się do dworca?"}'
```

Expected response:
```json
{
  "sentence_ids": [12],
  "sentences": ["Jak dojść do dworca kolejowego?"]
}
```

### Example 6: Family-Related Questions

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Kto pracuje jako lekarz?"}'
```

Expected response:
```json
{
  "sentence_ids": [25, 34],
  "sentences": [
    "Moja siostra pracuje jako lekarz.",
    "Mój brat studiuje medycynę."
  ]
}
```

### Example 7: Food and Cooking

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Co lubisz jeść?"}'
```

Expected response:
```json
{
  "sentence_ids": [26, 36, 44],
  "sentences": [
    "Lubię gotować włoskie dania.",
    "Lubię jeść pierogi z kapustą i grzybami.",
    "Moja mama gotuje najlepsze zupy."
  ]
}
```

## Using Python Requests

### Example Script

```python
import requests
import json

BASE_URL = "http://localhost:8000"

# Initialize FAISS index
def initialize():
    response = requests.post(f"{BASE_URL}/init")
    print("Init Response:", json.dumps(response.json(), indent=2, ensure_ascii=False))

# Query the system
def query(question):
    response = requests.post(
        f"{BASE_URL}/chat",
        json={"question": question}
    )
    result = response.json()
    print(f"\nQuestion: {question}")
    print(f"Found {len(result['sentence_ids'])} matching sentences:")
    for i, (sid, sentence) in enumerate(zip(result['sentence_ids'], result['sentences']), 1):
        print(f"  {i}. [ID: {sid}] {sentence}")

# Health check
def health_check():
    response = requests.get(f"{BASE_URL}/health")
    print("Health Status:", json.dumps(response.json(), indent=2))

# Run examples
if __name__ == "__main__":
    # Check health
    health_check()
    
    # Query examples
    query("Gdzie jest Warszawa?")
    query("Co ludzie lubią robić w wolnym czasie?")
    query("Kto uczy się programowania?")
```

## Using JavaScript/Node.js

### Example Script

```javascript
const axios = require('axios');

const BASE_URL = 'http://localhost:8000';

// Initialize FAISS index
async function initialize() {
    const response = await axios.post(`${BASE_URL}/init`);
    console.log('Init Response:', JSON.stringify(response.data, null, 2));
}

// Query the system
async function query(question) {
    const response = await axios.post(`${BASE_URL}/chat`, {
        question: question
    });
    const result = response.data;
    console.log(`\nQuestion: ${question}`);
    console.log(`Found ${result.sentence_ids.length} matching sentences:`);
    result.sentence_ids.forEach((id, index) => {
        console.log(`  ${index + 1}. [ID: ${id}] ${result.sentences[index]}`);
    });
}

// Health check
async function healthCheck() {
    const response = await axios.get(`${BASE_URL}/health`);
    console.log('Health Status:', JSON.stringify(response.data, null, 2));
}

// Run examples
(async () => {
    await healthCheck();
    await query('Gdzie jest Warszawa?');
    await query('Co ludzie lubią robić w wolnym czasie?');
})();
```

## Interactive API Documentation

Visit `http://localhost:8000/docs` for Swagger UI where you can:
- See all available endpoints
- Test endpoints interactively
- View request/response schemas
- Download OpenAPI specification

## Troubleshooting

### FAISS Index Not Found

If you get "FAISS index not found" error:
```bash
# Reinitialize the index
curl -X POST http://localhost:8000/init
```

### Database Connection Failed

If you get database errors:
```bash
# Verify PostgreSQL is running
sudo systemctl status postgresql

# Reinitialize the database
python init_db.py

# Test via API health endpoint
curl http://localhost:8000/health
```

### OpenAI API Errors

If you get OpenAI errors:
1. Verify API key in `.env`
2. Check API key has credits: https://platform.openai.com/account/usage
3. Verify model access (gpt-3.5-turbo should be available to all)

### Empty Results

If queries return empty results:
1. Ensure FAISS index was initialized successfully
2. Try rephrasing the question in Polish
3. Check the TOP_K_RESULTS setting in `.env` (default: 5)
4. Review retrieved sentences in the server logs

## Advanced Configuration

### Changing the Number of Retrieved Results

Edit `.env`:
```bash
TOP_K_RESULTS=10  # Retrieve top 10 instead of 5
```

### Using a Different LLM Model

Edit `.env`:
```bash
LLM_MODEL=gpt-4o-mini  # Use GPT-4o-mini for faster/cheaper responses
```

### Custom FAISS Index Location

Edit `.env`:
```bash
FAISS_INDEX_PATH=/custom/path/to/faiss_index
```

Restart the server after making changes to `.env`.



