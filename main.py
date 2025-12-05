import logging
import getpass
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any
from enum import Enum
from dataclasses import dataclass
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError, validator, Field
from qdrant_client import models, QdrantClient
from uuid import uuid4
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from abc import ABC, abstractmethod
from sqlalchemy import create_engine, text, select, schema 
from sqlalchemy.engine import Engine
from urllib.parse import quote_plus
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from langchain_community.utilities.sql_database import SQLDatabase
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Initialize FastAPI app
app = FastAPI(
    title="Natural Language to SQL API",
    description="Convert natural language queries to SQL using RAG with Qdrant",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Qdrant client and embedding model
try:
    client = QdrantClient("localhost", port=6333)
    ollama_embedding = OllamaEmbeddings(base_url='http://localhost:11434', model='bge-m3')
    ollama_llm = OllamaLLM(base_url='http://localhost:11434', model='llama3.2')
    logger.info("Successfully initialized Qdrant client and Ollama models")
except Exception as e:
    logger.error(f"Failed to initialize services: {e}")
    client = None
    ollama_embedding = None
    ollama_llm = None

# Pydantic Models
class DatabaseCredentials(BaseModel):
    host: str = Field(..., description="Database host address")
    port: int = Field(..., description="Database port")
    username: str = Field(..., description="Database username")
    password: str = Field(..., description="Database password")
    database: str = Field(..., description="Database name")
    
    class Config:
        schema_extra = {
            "example": {
                "host": "localhost",
                "port": 5432,
                "username": "user",
                "password": "password",
                "database": "sakila"
            }
        }

class DatabaseConnectionRequest(BaseModel):
    database_type: str = Field(..., description="Type of database (mysql or postgresql)")
    credentials: DatabaseCredentials
    collection_name: Optional[str] = Field(default="database_schema", description="Qdrant collection name")

class NaturalLanguageQuery(BaseModel):
    query: str = Field(..., description="Natural language query")
    database_name: str = Field(..., description="Database name for context")
    collection_name: Optional[str] = Field(default="database_schema", description="Qdrant collection name")
    top_k: Optional[int] = Field(default=3, description="Number of relevant schemas to retrieve")

class SQLResponse(BaseModel):
    sql_query: str
    confidence: float
    retrieved_schemas: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class ErrorResponse(BaseModel):
    error: str
    details: Optional[str] = None

class HybridSearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    database_name: Optional[str] = Field(default=None, description="Database name for filtering")
    collection_name: Optional[str] = Field(default="database_schema", description="Qdrant collection name")
    top_k: Optional[int] = Field(default=3, description="Number of results to return")
    use_text_filter: Optional[bool] = Field(default=True, description="Whether to use text filtering")

# Original Database Classes (Refactored for FastAPI)
class AvailableDataBase:
    supported_databases = ["MySQL", "PostgreSQL"]

    def show_available_database(self) -> str:
        databases = "\n".join(f"{i+1}. {database}" for i, database in enumerate(self.supported_databases))
        return f"The List of supported Database Types :\n {databases}"

class DatabaseAdapter(ABC):
    @abstractmethod 
    def build_connection_url(self, credentials: Dict[str, Any]) -> str:
        pass 
    
    @abstractmethod
    def get_default_port(self) -> int:
        pass 

    @abstractmethod
    def validate_credentials(self, credentials: Dict[str, Any]) -> bool:
        pass 

    def get_langchain_params(self) -> Dict[str, Any]:
        return {
            'sample_rows_in_table_info': 3,
            'include_tables': None,
            'ignore_tables': None
        }

class PostgreSQLAdapter(DatabaseAdapter):
    def build_connection_url(self, credentials: Dict[str, Any]) -> str:
        username = quote_plus(credentials['username'])
        password = quote_plus(credentials['password'])
        host = credentials.get('host', 'localhost')
        port = credentials.get('port', self.get_default_port())
        database = credentials['database']
        
        return f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}"
    
    def get_default_port(self) -> int:
        return 5432
    
    def validate_credentials(self, credentials: Dict[str, Any]) -> bool:
        required = {'username', 'password', 'database'}
        return all(key in credentials for key in required)

class MySQLAdapter(DatabaseAdapter):
    def build_connection_url(self, credentials: Dict[str, Any]) -> str:
        username = quote_plus(credentials['username'])
        password = quote_plus(credentials['password'])
        host = credentials.get('host', 'localhost')
        port = credentials.get('port', self.get_default_port())
        database = credentials['database']

        return f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}"
    
    def get_default_port(self) -> int:
        return 3306
    
    def validate_credentials(self, credentials: Dict[str, Any]) -> bool:
        required = {'username', 'password', 'database'}
        return all(key in credentials for key in required)

class DatabaseAdapterFactory:
    _adapters = {
        'mysql': MySQLAdapter,
        'postgresql': PostgreSQLAdapter
    }
    
    @classmethod
    def create_adapter(cls, database_type: str) -> DatabaseAdapter:
        database_type = database_type.lower()
        if database_type not in cls._adapters:
            raise ValueError(f"Unsupported Database Type: {database_type}")
        return cls._adapters[database_type]()
    
    @classmethod
    def get_supported_types(cls) -> List[str]:
        return list(cls._adapters.keys())

class DatabaseConnection:
    def __init__(self, database_type: str, database_credentials: Dict[str, Any]):
        self.database_type = database_type.lower()
        self.database_credentials = database_credentials
        self.adapter = DatabaseAdapterFactory.create_adapter(self.database_type)
        
        if not self.adapter.validate_credentials(database_credentials):
            raise ValueError(f"Invalid Credentials for: {self.database_type}")
        
    def connect(self) -> SQLDatabase:
        try:
            connection_url = self.adapter.build_connection_url(self.database_credentials)
            langchain_params = self.adapter.get_langchain_params()

            self.sql_database = SQLDatabase.from_uri(
                database_uri=connection_url,
                **langchain_params
            )
            
            logger.info(f"Successfully connected to {self.database_type} database")
            logger.info(f"Available tables: {self.sql_database.get_usable_table_names()}")
            
            return self.sql_database
    
        except Exception as e:
            logger.error(f"Failed to connect to {self.database_type}: {str(e)}")
            raise ConnectionError(f"Database connection failed: {str(e)}")

    def get_database_schema(self) -> str:
        try:
            sql_database = self.connect()
            database_schema = sql_database.get_table_info()
            return database_schema
        except Exception as e:
            raise ConnectionError("Error occurred during the connection")

# Qdrant and RAG Functions
def ensure_collection_exists(collection_name: str):
    """Ensure Qdrant collection exists"""
    try:
        collections = client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        if collection_name not in collection_names:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE),    
            )
            logger.info(f"Created collection: {collection_name}")
        else:
            logger.info(f"Collection {collection_name} already exists")
    except Exception as e:
        logger.error(f"Error ensuring collection exists: {e}")
        raise

def store_schema_in_qdrant(
    database_engine: SQLDatabase, 
    client: QdrantClient, 
    embedding_model, 
    database_name: str, 
    collection_name: str
):
    """Store database schema in Qdrant for RAG"""
    try:
        ensure_collection_exists(collection_name)
        
        table_names = database_engine.get_usable_table_names()
        logger.info(f"Found tables: {table_names}")
        
        points = []
        for table in table_names:
            try:
                # Get table schema
                raw_schema = database_engine.get_table_info([table])
                
                # Generate embedding
                vector = embedding_model.embed_query(raw_schema)
                
                # Create point
                point_id = str(uuid4())
                point = models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "database_name": database_name,
                        "table_name": table,
                        "raw_schema": raw_schema
                    }
                )
                points.append(point)
                
            except Exception as e:
                logger.error(f"Error processing table {table}: {e}")
                continue
        
        # Upsert all points
        if points:
            client.upsert(
                collection_name=collection_name,
                points=points
            )
            logger.info(f"Stored {len(points)} table schemas in Qdrant")
        else:
            logger.warning("No schemas were stored")
            
    except Exception as e:
        logger.error(f"Error storing schema in Qdrant: {e}")
        raise

def search_relevant_schemas(
    query: str, 
    database_name: str, 
    collection_name: str, 
    top_k: int = 3
) -> List[Dict[str, Any]]:
    """Search for relevant schemas using RAG"""
    try:
        # Vectorize the query
        query_vector = ollama_embedding.embed_query(query)
        
        # Search in Qdrant
        search_results = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="database_name",
                        match=models.MatchValue(value=database_name)
                    )
                ]
            ),
            limit=top_k
        )
        
        relevant_schemas = []
        for result in search_results:
            relevant_schemas.append({
                "table_name": result.payload["table_name"],
                "raw_schema": result.payload["raw_schema"],
                "score": result.score
            })
        
        logger.info(f"Retrieved {len(relevant_schemas)} relevant schemas")
        return relevant_schemas
        
    except Exception as e:
        logger.error(f"Error searching schemas: {e}")
        raise

def hybrid_search(
    client: QdrantClient,
    query: str,
    collection_name: str = "database_schema",
    database_name: str = None,
    top_k: int = 3,
    use_text_filter: bool = True
) -> List[Dict[str, Any]]:
    """
    Perform hybrid search combining vector similarity and optional text filtering
    
    Args:
        client: QdrantClient instance
        query: Search query string
        collection_name: Name of the Qdrant collection
        database_name: Optional database name for filtering
        top_k: Number of results to return
        use_text_filter: Whether to apply additional text filtering
    
    Returns:
        List of search results with scores and payloads
    """
    try:
        # Vectorize the query
        query_vector = ollama_embedding.embed_query(query)
        
        # Build query filter
        must_conditions = []
        
        # Add database name filter if provided
        if database_name:
            must_conditions.append(
                models.FieldCondition(
                    key="database_name",
                    match=models.MatchValue(value=database_name)
                )
            )
        
        # Add text filter if enabled (searches within the raw_schema text)
        if use_text_filter:
            must_conditions.append(
                models.FieldCondition(
                    key="raw_schema",
                    match=models.MatchText(text=query)
                )
            )
        
        # Create filter object
        query_filter = None
        if must_conditions:
            query_filter = models.Filter(must=must_conditions)
        
        # Perform hybrid search using vector similarity + optional filters
        search_results = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
            with_vectors=False
        )
        
        # Format results
        results = []
        for result in search_results:
            results.append({
                "id": result.id,
                "score": result.score,
                "payload": result.payload,
                "table_name": result.payload.get("table_name", ""),
                "raw_schema": result.payload.get("raw_schema", ""),
                "database_name": result.payload.get("database_name", "")
            })
        
        logger.info(f"Hybrid search returned {len(results)} results")
        return results
        
    except Exception as e:
        logger.error(f"Error in hybrid search: {e}")
        raise

def generate_sql_with_llm(query: str, schemas: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate SQL using LLM with retrieved schemas"""
    try:
        # Prepare context from retrieved schemas
        schema_context = "\n\n".join([
            f"Table: {schema['table_name']}\n{schema['raw_schema']}"
            for schema in schemas
        ])
        
        # Create prompt template
        prompt_template = PromptTemplate(
            input_variables=["schema_context", "query"],
            template="""
Given the following database schema information:

{schema_context}

Generate a SQL query for the following natural language request:
"{query}"

Please provide only the SQL query without any explanation or additional text.
SQL Query:
"""
        )
        
        # Create LLM chain
        llm_chain = LLMChain(llm=ollama_llm, prompt=prompt_template)
        
        # Generate SQL
        result = llm_chain.run(schema_context=schema_context, query=query)
        
        # Clean up the result
        sql_query = result.strip()
        if sql_query.startswith("SQL Query:"):
            sql_query = sql_query[10:].strip()
        
        return {
            "sql_query": sql_query,
            "confidence": 0.8,  # You could implement a confidence scoring mechanism
            "retrieved_schemas": schemas,
            "prompt_used": prompt_template.format(schema_context=schema_context, query=query)
        }
        
    except Exception as e:
        logger.error(f"Error generating SQL: {e}")
        raise

# FastAPI Endpoints
@app.get("/")
async def root():
    return {"message": "Natural Language to SQL API", "status": "running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "services": {
            "qdrant": client is not None,
            "ollama_embedding": ollama_embedding is not None,
            "ollama_llm": ollama_llm is not None
        }
    }

@app.get("/supported-databases")
async def get_supported_databases():
    """Get list of supported database types"""
    available_db = AvailableDataBase()
    return {
        "supported_databases": DatabaseAdapterFactory.get_supported_types(),
        "description": available_db.show_available_database()
    }

@app.post("/connect-and-store-schema")
async def connect_and_store_schema(request: DatabaseConnectionRequest):
    """Connect to database and store schema in Qdrant"""
    try:
        if not all([client, ollama_embedding]):
            raise HTTPException(status_code=503, detail="Required services not available")
        
        # Create database connection
        db_connection = DatabaseConnection(
            database_type=request.database_type,
            database_credentials=request.credentials.dict()
        )
        
        # Connect and get SQLDatabase instance
        sql_database = db_connection.connect()
        
        # Store schema in Qdrant
        store_schema_in_qdrant(
            database_engine=sql_database,
            client=client,
            embedding_model=ollama_embedding,
            database_name=request.credentials.database,
            collection_name=request.collection_name
        )
        
        return {
            "message": "Successfully connected to database and stored schema",
            "database_type": request.database_type,
            "database_name": request.credentials.database,
            "collection_name": request.collection_name,
            "tables": sql_database.get_usable_table_names()
        }
        
    except Exception as e:
        logger.error(f"Error in connect_and_store_schema: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-sql", response_model=SQLResponse)
async def generate_sql(request: NaturalLanguageQuery):
    """Generate SQL from natural language query using RAG"""
    try:
        if not all([client, ollama_embedding, ollama_llm]):
            raise HTTPException(status_code=503, detail="Required services not available")
        
        # Search for relevant schemas
        relevant_schemas = search_relevant_schemas(
            query=request.query,
            database_name=request.database_name,
            collection_name=request.collection_name,
            top_k=request.top_k
        )
        
        if not relevant_schemas:
            raise HTTPException(
                status_code=404, 
                detail=f"No relevant schemas found for database: {request.database_name}"
            )
        
        # Generate SQL using LLM
        result = generate_sql_with_llm(request.query, relevant_schemas)
        
        return SQLResponse(
            sql_query=result["sql_query"],
            confidence=result["confidence"],
            retrieved_schemas=result["retrieved_schemas"],
            metadata={
                "original_query": request.query,
                "database_name": request.database_name,
                "schemas_retrieved": len(relevant_schemas)
            }
        )
        
    except Exception as e:
        logger.error(f"Error in generate_sql: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/hybrid-search")
async def perform_hybrid_search(request: HybridSearchRequest):
    """Perform hybrid search combining vector similarity and text filtering"""
    try:
        if not all([client, ollama_embedding]):
            raise HTTPException(status_code=503, detail="Required services not available")
        
        results = hybrid_search(
            client=client,
            query=request.query,
            collection_name=request.collection_name,
            database_name=request.database_name,
            top_k=request.top_k,
            use_text_filter=request.use_text_filter
        )
        
        return {
            "query": request.query,
            "results": results,
            "total_results": len(results),
            "collection_name": request.collection_name,
            "database_name": request.database_name
        }
        
    except Exception as e:
        logger.error(f"Error in hybrid search endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/collections")
async def list_collections():
    """List all Qdrant collections"""
    try:
        if not client:
            raise HTTPException(status_code=503, detail="Qdrant client not available")
        
        collections = client.get_collections()
        return {
            "collections": [
                {
                    "name": col.name,
                    "vectors_count": client.count(col.name).count if hasattr(client.count(col.name), 'count') else 0
                }
                for col in collections.collections
            ]
        }
    except Exception as e:
        logger.error(f"Error listing collections: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/collections/{collection_name}")
async def delete_collection(collection_name: str):
    """Delete a Qdrant collection"""
    try:
        if not client:
            raise HTTPException(status_code=503, detail="Qdrant client not available")
        
        client.delete_collection(collection_name)
        return {"message": f"Collection {collection_name} deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting collection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 