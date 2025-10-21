# Docker Deployment Guide

This guide covers deploying the RAG Training API using Docker and Docker Compose.

## Architecture

The Docker setup includes two services:

1. **postgres**: PostgreSQL 15 database for storing Polish sentences
2. **api**: FastAPI application with LlamaIndex and FAISS

Both services are connected via a bridge network and use persistent volumes for data storage.

## Files Overview

### Core Files

- **`Dockerfile`**: Multi-stage build for optimized production image
- **`docker-compose.yml`**: Development/production orchestration
- **`docker-compose.prod.yml`**: Production-specific overrides
- **`.dockerignore`**: Excludes unnecessary files from build context
- **`scripts/docker-entrypoint.sh`**: Container initialization script

### Environment Configuration

Create a `.env` file in the project root with your configuration:

```bash
# Copy from example
cp .env.example .env

# Edit with your values
nano .env
```

Required environment variables:
```env
OPENAI_API_KEY=sk-...
VOYAGE_API_KEY=pa-...
```

Optional environment variables (with defaults):
```env
DB_NAME=rag_db_training
DB_USER=postgres
DB_PASSWORD=postgres
LLM_MODEL=gpt-4o
EMBEDDING_MODEL=voyage-3-large
TOP_K_RESULTS=5
LOG_LEVEL=INFO
WORKERS=4
```

## Quick Start

```bash
# 1. Set up environment
cp .env.example .env
# Edit .env with your API keys

# 2. Start services
docker-compose up -d

# 3. Wait for services to be ready (check logs)
docker-compose logs -f

# 4. Initialize FAISS index
curl -X POST http://localhost:8000/init

# 5. Test the API
curl -X POST http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"question": "Gdzie jest Warszawa?"}'
```

## Service Details

### PostgreSQL Service

- **Image**: postgres:15-alpine
- **Port**: 5432
- **Volume**: `rag_postgres_data` (persistent)
- **Health Check**: `pg_isready` every 10s
- **Restart Policy**: unless-stopped

### API Service

- **Build**: Multi-stage Dockerfile
- **Port**: 8000
- **Volume**: `rag_faiss_index` (persistent)
- **Health Check**: `/health` endpoint every 30s
- **Restart Policy**: unless-stopped
- **User**: non-root (appuser, UID 1000)

## Development Workflow

### Building and Running

```bash
# Build images
docker-compose build

# Build with no cache (fresh build)
docker-compose build --no-cache

# Start services
docker-compose up

# Start in background
docker-compose up -d

# Follow logs
docker-compose logs -f

# Follow specific service logs
docker-compose logs -f api
docker-compose logs -f postgres
```

### Making Code Changes

```bash
# After changing Python code
docker-compose up --build

# Or rebuild specific service
docker-compose build api
docker-compose up -d api
```

### Accessing Services

```bash
# Execute command in API container
docker-compose exec api bash
docker-compose exec api python init_db.py

# Access PostgreSQL
docker-compose exec postgres psql -U postgres -d rag_db_training

# Check running containers
docker-compose ps

# View resource usage
docker stats
```

## Production Deployment

### Using Production Configuration

```bash
# Start with production settings
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# View logs
docker-compose -f docker-compose.yml -f docker-compose.prod.yml logs -f
```

### Production Features

The `docker-compose.prod.yml` adds:

1. **Resource Limits**:
   - API: 2 CPU cores, 2GB RAM (limit)
   - PostgreSQL: 2 CPU cores, 2GB RAM (limit)

2. **Security Hardening**:
   - Read-only root filesystem
   - Dropped capabilities (cap_drop: ALL)
   - No new privileges
   - PostgreSQL port not exposed externally

3. **Optimized Performance**:
   - 4 Uvicorn workers
   - PostgreSQL tuning (shared_buffers, max_connections)

4. **Logging**:
   - JSON log driver
   - Log rotation (10MB max, 3-5 files)

### Reverse Proxy Setup (Nginx)

For production, use Nginx as a reverse proxy:

```nginx
server {
    listen 80;
    server_name api.example.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## Data Management

### Volumes

```bash
# List volumes
docker volume ls | grep rag

# Inspect volume details
docker volume inspect rag_postgres_data
docker volume inspect rag_faiss_index

# Remove all volumes (WARNING: deletes data)
docker-compose down -v
```

### Backup and Restore

#### FAISS Index

```bash
# Backup
docker run --rm \
  -v rag_faiss_index:/data \
  -v $(pwd):/backup \
  ubuntu tar czf /backup/faiss_backup.tar.gz -C /data .

# Restore
docker run --rm \
  -v rag_faiss_index:/data \
  -v $(pwd):/backup \
  ubuntu tar xzf /backup/faiss_backup.tar.gz -C /data
```

#### PostgreSQL Database

```bash
# Backup
docker-compose exec postgres pg_dump -U postgres rag_db_training > backup.sql

# Restore
docker-compose exec -T postgres psql -U postgres rag_db_training < backup.sql
```

## Monitoring

### Health Checks

```bash
# Check service health
docker-compose ps

# Manual health check
curl http://localhost:8000/health
```

### Logs

```bash
# All logs
docker-compose logs

# Last 100 lines
docker-compose logs --tail=100

# Follow logs
docker-compose logs -f

# Specific service
docker-compose logs -f api

# Timestamps
docker-compose logs -t
```

### Metrics

```bash
# Resource usage
docker stats

# Specific container
docker stats rag-api
```

## Troubleshooting

### Common Issues

#### 1. Container Exits Immediately

```bash
# Check exit reason
docker-compose logs api

# Common causes:
# - Database not ready (wait for postgres health check)
# - Missing environment variables
# - Python dependencies error
```

#### 2. Database Connection Failed

```bash
# Verify PostgreSQL is running
docker-compose ps postgres

# Check PostgreSQL logs
docker-compose logs postgres

# Test connection
docker-compose exec postgres pg_isready -U postgres

# Manual connection test
docker-compose exec api python -c "import database; print(database.test_connection())"
```

#### 3. FAISS Index Errors

```bash
# Remove corrupted index
docker volume rm rag_faiss_index

# Restart and reinitialize
docker-compose up -d
curl -X POST http://localhost:8000/init
```

#### 4. Port Already in Use

```bash
# Find process using port
lsof -i :8000
# or
netstat -tulpn | grep 8000

# Change port in docker-compose.yml
ports:
  - "8001:8000"
```

#### 5. Out of Disk Space

```bash
# Clean Docker system
docker system prune -a --volumes

# Remove stopped containers
docker container prune

# Remove unused images
docker image prune -a

# Remove unused volumes
docker volume prune
```

### Debug Mode

```bash
# Set debug logging
echo "LOG_LEVEL=DEBUG" >> .env

# Restart with debug logs
docker-compose up --build

# Or set temporarily
docker-compose run -e LOG_LEVEL=DEBUG api uvicorn main:app --host 0.0.0.0
```

## Performance Tuning

### Scaling Workers

```bash
# Set workers in .env
WORKERS=8

# Or override in compose
docker-compose up -d --scale api=2
```

### PostgreSQL Tuning

Edit `docker-compose.prod.yml`:

```yaml
postgres:
  command: >
    postgres 
    -c shared_buffers=512MB
    -c effective_cache_size=2GB
    -c maintenance_work_mem=128MB
    -c max_connections=200
```

### Resource Limits

Adjust in `docker-compose.prod.yml`:

```yaml
deploy:
  resources:
    limits:
      cpus: '4.0'
      memory: 4G
    reservations:
      cpus: '2.0'
      memory: 2G
```

## Security Best Practices

1. **Never commit `.env` file**: Use `.env.example` as template
2. **Use secrets management**: For production, use Docker secrets or external secret managers
3. **Update base images regularly**: `docker-compose pull`
4. **Scan images for vulnerabilities**: `docker scan rag-api`
5. **Use specific image tags**: Avoid `latest` in production
6. **Enable Docker Content Trust**: `export DOCKER_CONTENT_TRUST=1`
7. **Limit container capabilities**: Already configured in prod compose
8. **Use read-only filesystem**: Already configured in prod compose

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Docker Build and Deploy

on:
  push:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Build Docker image
        run: docker-compose build
      
      - name: Run tests
        run: docker-compose run api pytest
      
      - name: Deploy to production
        run: |
          docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

## Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)
- [PostgreSQL Docker Hub](https://hub.docker.com/_/postgres)

