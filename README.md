# Natural Language to SQL API

A FastAPI application that converts natural language queries to SQL using Retrieval Augmented Generation (RAG) with Qdrant vector database.

## Features

- **Database Connectivity**: Supports MySQL and PostgreSQL databases
- **Schema Storage**: Automatically extracts and stores database schemas in Qdrant vector database
- **Natural Language Processing**: Converts natural language queries to SQL using Ollama LLM
- **RAG Implementation**: Uses vector similarity search to retrieve relevant schema context
- **RESTful API**: Clean FastAPI endpoints for easy integration

## Prerequisites

1. **Python 3.8+**
2. **Ollama** - Install and run locally on port 11434
   - Install models: `bge-m3` (for embeddings) and `llama3.2` (for LLM)
3. **Qdrant** - Vector database running on port 6333
4. **Database** - MySQL or PostgreSQL instance

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd natural-language-to-sql
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start required services:

### Start Qdrant (using Docker):
```bash
docker run -p 6333:6333 qdrant/qdrant
```

### Start Ollama and download models:
```bash
# Install Ollama first: https://ollama.ai/download
ollama run bge-m3
ollama run llama3.2
```

4. Start the FastAPI application:
```bash
python main.py
```

The API will be available at `http://localhost:8000`

## API Documentation

Once the application is running, visit:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## API Endpoints

### 1. Health Check
```http
GET /health
```

### 2. Get Supported Databases
```http
GET /supported-databases
```

### 3. Connect to Database and Store Schema
```http
POST /connect-and-store-schema
```

**Request Body:**
```json
{
  "database_type": "postgresql",
  "credentials": {
    "host": "localhost",
    "port": 5432,
    "username": "your_username",
    "password": "your_password",
    "database": "your_database"
  },
  "collection_name": "database_schema"
}
```

### 4. Generate SQL from Natural Language
```http
POST /generate-sql
```

**Request Body:**
```json
{
  "query": "Show me all customers from the USA",
  "database_name": "your_database",
  "collection_name": "database_schema",
  "top_k": 3
}
```

**Response:**
```json
{
  "sql_query": "SELECT * FROM customers WHERE country = 'USA';",
  "confidence": 0.8,
  "retrieved_schemas": [
    {
      "table_name": "customers",
      "raw_schema": "...",
      "score": 0.95
    }
  ],
  "metadata": {
    "original_query": "Show me all customers from the USA",
    "database_name": "your_database",
    "schemas_retrieved": 1
  }
}
```

## Usage Example

1. **First, connect to your database and store the schema:**
```python
import requests

# Connect to database and store schema
response = requests.post("http://localhost:8000/connect-and-store-schema", json={
    "database_type": "postgresql",
    "credentials": {
        "host": "localhost",
        "port": 5432,
        "username": "postgres",
        "password": "password",
        "database": "sakila"
    }
})
print(response.json())
```

2. **Then, generate SQL from natural language:**
```python
# Generate SQL query
response = requests.post("http://localhost:8000/generate-sql", json={
    "query": "Find all actors who appeared in more than 5 films",
    "database_name": "sakila"
})
print(response.json()["sql_query"])
```

## Configuration

### Environment Variables
You can customize the following via environment variables:

- `QDRANT_HOST`: Qdrant host (default: localhost)
- `QDRANT_PORT`: Qdrant port (default: 6333)
- `OLLAMA_BASE_URL`: Ollama base URL (default: http://localhost:11434)
- `EMBEDDING_MODEL`: Embedding model name (default: bge-m3)
- `LLM_MODEL`: LLM model name (default: llama3.2)

### Ollama Models
Ensure you have the required Ollama models:
```bash
ollama pull bge-m3      # For embeddings
ollama pull llama3.2    # For SQL generation
```

## Docker Compose Setup

For a complete setup with all services:

```yaml
version: '3.8'
services:
  qdrant:
    image: qdrant/qdrant
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

  api:
    build: .
    ports:
      - "8000:8000"
    depends_on:
      - qdrant
    environment:
      - QDRANT_HOST=qdrant
      - OLLAMA_BASE_URL=http://host.docker.internal:11434

volumes:
  qdrant_data:
```

## Troubleshooting

### Common Issues

1. **Qdrant Connection Error**
   - Ensure Qdrant is running on port 6333
   - Check firewall settings

2. **Ollama Model Not Found**
   - Pull the required models: `ollama pull bge-m3` and `ollama pull llama3.2`
   - Ensure Ollama is running on port 11434

3. **Database Connection Failed**
   - Verify database credentials
   - Check network connectivity
   - Ensure the database driver is installed (pymysql for MySQL, psycopg2 for PostgreSQL)

4. **No Relevant Schemas Found**
   - First run `/connect-and-store-schema` to populate the vector database
   - Check that the database name matches in both requests

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI App   │    │     Qdrant      │    │     Ollama      │
│                 │    │  Vector Store   │    │   LLM + Embed   │
├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│ • Schema Store  │◄──►│ • Schema Vectors│    │ • bge-m3 (embed)│
│ • Query Process │    │ • Metadata      │◄──►│ • llama3.2 (LLM)│
│ • SQL Generate  │    │ • Similarity    │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                                              │
         ▼                                              ▼
┌─────────────────┐                          ┌─────────────────┐
│    Database     │                          │   Generated     │
│  (MySQL/PG)     │                          │   SQL Query     │
└─────────────────┘                          └─────────────────┘
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details 