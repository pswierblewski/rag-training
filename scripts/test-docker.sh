#!/bin/bash
# Test script for Docker deployment

set -e

echo "=========================================="
echo "RAG Training API - Docker Test Script"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
API_URL="http://localhost:8000"
MAX_WAIT=60

# Function to print success
success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Function to print error
error() {
    echo -e "${RED}✗ $1${NC}"
    exit 1
}

# Function to print info
info() {
    echo -e "${YELLOW}→ $1${NC}"
}

# Check if Docker is running
info "Checking Docker..."
if ! docker info > /dev/null 2>&1; then
    error "Docker is not running. Please start Docker first."
fi
success "Docker is running"

# Check if docker-compose is available
info "Checking Docker Compose..."
if ! command -v docker-compose &> /dev/null; then
    error "docker-compose is not installed"
fi
success "Docker Compose is available"

# Check if .env file exists
info "Checking environment file..."
if [ ! -f .env ]; then
    error ".env file not found. Please create it from .env.example"
fi
success ".env file exists"

# Check if API keys are set
info "Checking API keys..."
source .env
if [ -z "$OPENAI_API_KEY" ] || [ "$OPENAI_API_KEY" = "your_openai_api_key_here" ]; then
    error "OPENAI_API_KEY not set in .env file"
fi
if [ -z "$VOYAGE_API_KEY" ] || [ "$VOYAGE_API_KEY" = "your_voyage_api_key_here" ]; then
    error "VOYAGE_API_KEY not set in .env file"
fi
success "API keys are configured"

# Build Docker images
info "Building Docker images..."
docker-compose build --quiet
success "Docker images built"

# Start services
info "Starting services..."
docker-compose up -d
success "Services started"

# Wait for API to be ready
info "Waiting for API to be ready (max ${MAX_WAIT}s)..."
waited=0
while [ $waited -lt $MAX_WAIT ]; do
    if curl -sf $API_URL/health > /dev/null 2>&1; then
        break
    fi
    sleep 2
    waited=$((waited + 2))
    echo -n "."
done
echo ""

if [ $waited -ge $MAX_WAIT ]; then
    error "API failed to start within ${MAX_WAIT} seconds"
fi
success "API is ready"

# Check health endpoint
info "Testing health endpoint..."
HEALTH=$(curl -sf $API_URL/health)
if echo $HEALTH | grep -q "healthy\|connected"; then
    success "Health check passed"
else
    error "Health check failed: $HEALTH"
fi

# Initialize FAISS index
info "Initializing FAISS index..."
INIT_RESPONSE=$(curl -sf -X POST $API_URL/init)
if echo $INIT_RESPONSE | grep -q "successfully"; then
    SENTENCES=$(echo $INIT_RESPONSE | grep -o '"sentences_indexed":[0-9]*' | grep -o '[0-9]*')
    success "FAISS index initialized ($SENTENCES sentences indexed)"
else
    error "Failed to initialize FAISS index: $INIT_RESPONSE"
fi

# Test chat endpoint
info "Testing chat endpoint..."
CHAT_RESPONSE=$(curl -sf -X POST $API_URL/chat \
    -H 'Content-Type: application/json' \
    -d '{"question": "Gdzie jest Warszawa?"}')

if echo $CHAT_RESPONSE | grep -q "sentence"; then
    success "Chat endpoint working"
    echo "   Response: $CHAT_RESPONSE"
else
    error "Chat endpoint failed: $CHAT_RESPONSE"
fi

# Check container status
info "Checking container status..."
if docker-compose ps | grep -q "Up"; then
    success "Containers are running"
else
    error "Some containers are not running"
fi

# Check volumes
info "Checking volumes..."
if docker volume ls | grep -q "rag_postgres_data"; then
    success "PostgreSQL volume exists"
else
    error "PostgreSQL volume not found"
fi

if docker volume ls | grep -q "rag_faiss_index"; then
    success "FAISS index volume exists"
else
    error "FAISS index volume not found"
fi

# Show logs summary
echo ""
echo "=========================================="
echo "Recent API Logs:"
echo "=========================================="
docker-compose logs --tail=10 api

echo ""
echo "=========================================="
info "All tests passed! ✨"
echo "=========================================="
echo ""
echo "Services are running at:"
echo "  - API: $API_URL"
echo "  - API Docs: $API_URL/docs"
echo "  - PostgreSQL: localhost:5432"
echo ""
echo "Useful commands:"
echo "  - View logs: docker-compose logs -f"
echo "  - Stop services: docker-compose down"
echo "  - Rebuild: docker-compose up --build"
echo ""

