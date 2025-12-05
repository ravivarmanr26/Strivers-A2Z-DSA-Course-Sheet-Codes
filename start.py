#!/usr/bin/env python3
"""
Quick start script for Natural Language to SQL API
"""

import uvicorn
import sys
import requests
import time
from pathlib import Path

def check_prerequisites():
    """Check if all required services are running"""
    print("🔍 Checking prerequisites...")
    
    # Check if main.py exists
    if not Path("main.py").exists():
        print("❌ main.py not found in current directory")
        return False
    
    # Check Qdrant
    try:
        response = requests.get("http://localhost:6333/health", timeout=5)
        print("✅ Qdrant is running")
    except requests.exceptions.RequestException:
        print("❌ Qdrant is not running")
        print("   Start with: docker run -p 6333:6333 qdrant/qdrant")
        return False
    
    # Check Ollama
    try:
        response = requests.get("http://localhost:11434", timeout=5)
        print("✅ Ollama is running")
    except requests.exceptions.RequestException:
        print("❌ Ollama is not running")
        print("   Please start Ollama service")
        return False
    
    return True

def main():
    """Main function to start the API"""
    print("🚀 Starting Natural Language to SQL API...")
    print("=" * 50)
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Please resolve the issues above.")
        print("\n💡 Quick setup:")
        print("   1. Run: python setup.py")
        print("   2. Start Qdrant: docker run -p 6333:6333 qdrant/qdrant")
        print("   3. Start Ollama and pull models:")
        print("      ollama pull bge-m3")
        print("      ollama pull llama3.2")
        sys.exit(1)
    
    print("\n🚀 Starting FastAPI server...")
    print("📍 API will be available at: http://localhost:8000")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("📖 Alternative docs: http://localhost:8000/redoc")
    print("\n🛑 Press Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        # Start the server
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info",
            access_log=True
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server failed to start: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 