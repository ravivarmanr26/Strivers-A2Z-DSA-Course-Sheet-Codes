#!/usr/bin/env python3
"""
Test script for the Natural Language to SQL API
Demonstrates how to use the API endpoints
"""

import requests
import json
import time
from typing import Dict, Any

# API base URL
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint"""
    print("🔍 Testing health check...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_supported_databases():
    """Test the supported databases endpoint"""
    print("\n📋 Testing supported databases...")
    response = requests.get(f"{BASE_URL}/supported-databases")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_connect_and_store_schema(database_config: Dict[str, Any]):
    """Test connecting to database and storing schema"""
    print("\n🔗 Testing database connection and schema storage...")
    
    response = requests.post(
        f"{BASE_URL}/connect-and-store-schema",
        json=database_config
    )
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        print(f"✅ Success: {json.dumps(response.json(), indent=2)}")
        return True
    else:
        print(f"❌ Error: {response.text}")
        return False

def test_generate_sql(query: str, database_name: str):
    """Test generating SQL from natural language"""
    print(f"\n🤖 Testing SQL generation for query: '{query}'...")
    
    payload = {
        "query": query,
        "database_name": database_name,
        "collection_name": "database_schema",
        "top_k": 3
    }
    
    response = requests.post(
        f"{BASE_URL}/generate-sql",
        json=payload
    )
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Generated SQL: {result['sql_query']}")
        print(f"📊 Confidence: {result['confidence']}")
        print(f"📋 Retrieved {len(result['retrieved_schemas'])} schemas")
        return True
    else:
        print(f"❌ Error: {response.text}")
        return False

def test_list_collections():
    """Test listing Qdrant collections"""
    print("\n📦 Testing collection listing...")
    response = requests.get(f"{BASE_URL}/collections")
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        print(f"Collections: {json.dumps(response.json(), indent=2)}")
        return True
    else:
        print(f"❌ Error: {response.text}")
        return False

def main():
    """Main test function"""
    print("🚀 Starting API tests...\n")
    
    # Test basic endpoints
    if not test_health_check():
        print("❌ Health check failed. Make sure the API is running.")
        return
    
    if not test_supported_databases():
        print("❌ Supported databases test failed.")
        return
    
    # Example database configurations
    # Uncomment and modify the one you want to test
    
    # PostgreSQL example
    postgres_config = {
        "database_type": "postgresql",
        "credentials": {
            "host": "localhost",
            "port": 5432,
            "username": "postgres",
            "password": "password",
            "database": "sakila"
        },
        "collection_name": "database_schema"
    }
    
    # MySQL example
    mysql_config = {
        "database_type": "mysql",
        "credentials": {
            "host": "localhost",
            "port": 3306,
            "username": "root",
            "password": "password",
            "database": "sakila"
        },
        "collection_name": "database_schema"
    }
    
    # Choose your database configuration
    # Modify the credentials according to your setup
    database_config = postgres_config  # Change this to mysql_config if using MySQL
    
    print("⚠️  Make sure you have:")
    print("   - Database running with correct credentials")
    print("   - Qdrant running on port 6333")
    print("   - Ollama running on port 11434 with bge-m3 and llama3.2 models")
    print("\n🔄 Proceeding with tests in 3 seconds...")
    time.sleep(3)
    
    # Test database connection and schema storage
    if not test_connect_and_store_schema(database_config):
        print("❌ Database connection failed. Please check your credentials.")
        return
    
    # Wait a bit for schema to be processed
    print("\n⏳ Waiting for schema to be processed...")
    time.sleep(2)
    
    # Test listing collections
    test_list_collections()
    
    # Test SQL generation with various queries
    test_queries = [
        "Show me all customers",
        "Find customers from USA",
        "Get all actors who appeared in more than 5 films",
        "List all films released in 2006",
        "Show customer information with their addresses"
    ]
    
    database_name = database_config["credentials"]["database"]
    
    for query in test_queries:
        test_generate_sql(query, database_name)
        time.sleep(1)  # Small delay between requests
    
    print("\n✅ All tests completed!")

if __name__ == "__main__":
    main() 