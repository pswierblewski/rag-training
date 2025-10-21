# Docker Deployment Summary

This document provides a quick reference for the Docker deployment setup of the RAG Training API.

## Files Created

### Core Docker Files

1. **`Dockerfile`** (1.8KB)
   - Multi-stage build (builder + runtime)
   - Python 3.9-slim base image
   - Non-root user (appuser, UID 1000)
   - Health check integrated
   - Optimized layer caching

2. **`docker-compose.yml`** (2.0KB)
   - PostgreSQL 15 service with health checks
   - FastAPI API service
   - Named volumes for persistence
   - Bridge network
   - Environment variable configuration

3. **`docker-compose.prod.yml`** (1.5KB)
   - Production overrides
   - Resource limits (CPU, memory)
   - Security hardening
   - Log rotation
   - 4 Uvicorn workers

4. **`.dockerignore`** (707B)
   - Excludes venv, __pycache__, .git
   - Reduces build context by ~90%

### Scripts

5. **`scripts/docker-entrypoint.sh`**
   - Container initialization
   - PostgreSQL readiness check
   - Automatic database initialization
   - Graceful startup

6. **`scripts/test-docker.sh`**
   - Automated testing script
   - Validates Docker setup
   - Tests all endpoints
   - Checks volumes and containers

### Documentation

7. **`DOCKER.md`**
   - Comprehensive Docker guide
   - Architecture overview
   - Production deployment
   - Troubleshooting
   - Security best practices

8. **`README.md`** (updated)
   - Docker quick start section
   - Prerequisites updated
   - Volume management
   - Troubleshooting guide

## Quick Start Commands

### Development

```bash
# 1. Setup environment
cp .env.example .env
# Edit .env with your API keys

# 2. Start services
docker-compose up --build

# 3. In another terminal, initialize FAISS
curl -X POST http://localhost:8000/init

# 4. Test
curl -X POST http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"question": "Gdzie jest Warszawa?"}'
```

### Production

```bash
# Use production configuration
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Initialize FAISS
curl -X POST http://localhost:8000/init
```

### Testing

```bash
# Run automated test suite
./scripts/test-docker.sh
```

## Architecture

```
┌─────────────────────────────────────────┐
│         Docker Compose                  │
├─────────────────────────────────────────┤
│                                         │
│  ┌──────────────┐    ┌──────────────┐  │
│  │   postgres   │    │     api      │  │
│  │  (Port 5432) │◄───│  (Port 8000) │  │
│  │              │    │              │  │
│  │  PostgreSQL  │    │  FastAPI +   │  │
│  │  15-alpine   │    │  LlamaIndex  │  │
│  │              │    │  + FAISS     │  │
│  └──────┬───────┘    └───────┬──────┘  │
│         │                    │         │
│         ▼                    ▼         │
│  ┌──────────────┐    ┌──────────────┐  │
│  │postgres_data │    │ faiss_index  │  │
│  │  (Volume)    │    │  (Volume)    │  │
│  └──────────────┘    └──────────────┘  │
│                                         │
└─────────────────────────────────────────┘
         │                      │
         └──────────┬───────────┘
                    │
              rag-network
               (Bridge)
```

## Environment Variables

### Required
- `OPENAI_API_KEY`: OpenAI API key for GPT-4o
- `VOYAGE_API_KEY`: VoyageAI API key for embeddings

### Optional (with defaults)
- `DB_NAME`: rag_db_training
- `DB_USER`: postgres
- `DB_PASSWORD`: postgres
- `LLM_MODEL`: gpt-4o
- `EMBEDDING_MODEL`: voyage-3-large
- `TOP_K_RESULTS`: 5
- `LOG_LEVEL`: INFO
- `WORKERS`: 4

## Volumes

### postgres_data
- **Purpose**: PostgreSQL database persistence
- **Location**: Named volume `rag_postgres_data`
- **Size**: Typically < 50MB
- **Backup**: Use `pg_dump`

### faiss_index
- **Purpose**: FAISS vector store persistence
- **Location**: Named volume `rag_faiss_index`
- **Size**: Varies with data (typically 10-100MB)
- **Backup**: Use tar or volume copy

## Services

### postgres
- **Image**: postgres:15-alpine
- **Health Check**: `pg_isready` every 10s
- **Restart**: unless-stopped
- **Resources (prod)**: 2 CPU, 2GB RAM limit

### api
- **Build**: Multi-stage Dockerfile
- **Health Check**: `/health` endpoint every 30s
- **Restart**: unless-stopped
- **Resources (prod)**: 2 CPU, 2GB RAM limit
- **User**: appuser (UID 1000)

## Security Features

1. **Non-root user**: Application runs as appuser
2. **Read-only filesystem**: In production mode
3. **Dropped capabilities**: Minimal required permissions
4. **No new privileges**: Prevents privilege escalation
5. **Secrets via environment**: No hardcoded credentials
6. **Network isolation**: Internal bridge network
7. **Health monitoring**: Automatic restart on failure

## Performance Features

1. **Multi-stage build**: 60% smaller image
2. **Layer caching**: Faster rebuilds
3. **4 Uvicorn workers**: Better throughput
4. **PostgreSQL tuning**: Optimized for workload
5. **Volume persistence**: Fast FAISS index access

## Monitoring Endpoints

- **Health**: `GET /health`
- **Root**: `GET /`
- **Docs**: `GET /docs`
- **OpenAPI**: `GET /openapi.json`

## Common Operations

### View Logs
```bash
docker-compose logs -f api
docker-compose logs -f postgres
```

### Restart Service
```bash
docker-compose restart api
docker-compose restart postgres
```

### Scale Workers
```bash
docker-compose up -d --scale api=3
```

### Execute Commands
```bash
docker-compose exec api python init_db.py
docker-compose exec api bash
docker-compose exec postgres psql -U postgres -d rag_db_training
```

### Clean Up
```bash
# Stop and remove containers
docker-compose down

# Stop and remove volumes (WARNING: deletes data)
docker-compose down -v

# Remove all unused Docker resources
docker system prune -a --volumes
```

## Troubleshooting

### Issue: Container won't start
**Solution**: Check logs with `docker-compose logs api`

### Issue: Database connection failed
**Solution**: Wait for postgres health check, verify with `docker-compose ps`

### Issue: FAISS index error
**Solution**: Remove volume and reinitialize: `docker volume rm rag_faiss_index`

### Issue: Port already in use
**Solution**: Change port mapping in docker-compose.yml

### Issue: Out of memory
**Solution**: Adjust resource limits in docker-compose.prod.yml

## Production Checklist

- [ ] Set strong database password
- [ ] Configure resource limits
- [ ] Set up reverse proxy (Nginx/Traefik)
- [ ] Enable HTTPS/TLS
- [ ] Configure log rotation
- [ ] Set up monitoring (Prometheus/Grafana)
- [ ] Configure backup strategy
- [ ] Test disaster recovery
- [ ] Enable Docker Content Trust
- [ ] Regular security updates

## Additional Resources

- [Full Docker Guide](DOCKER.md)
- [Main README](README.md)
- [Usage Examples](USAGE_EXAMPLES.md)
- [Docker Documentation](https://docs.docker.com/)
- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)

## Support

For issues or questions:
1. Check logs: `docker-compose logs -f`
2. Review [DOCKER.md](DOCKER.md)
3. Run test script: `./scripts/test-docker.sh`
4. Check health endpoint: `curl http://localhost:8000/health`

