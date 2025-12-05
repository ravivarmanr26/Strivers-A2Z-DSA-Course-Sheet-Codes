#!/usr/bin/env python3
"""
Setup script for Natural Language to SQL API
Helps users set up the environment and download required models
"""

import subprocess
import sys
import time
import requests
from pathlib import Path

def run_command(command, description):
    """Run a shell command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False

def check_service(url, service_name):
    """Check if a service is running"""
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print(f"✅ {service_name} is running")
            return True
        else:
            print(f"❌ {service_name} returned status code {response.status_code}")
            return False
    except requests.exceptions.RequestException:
        print(f"❌ {service_name} is not accessible")
        return False

def install_dependencies():
    """Install Python dependencies"""
    print("📦 Installing Python dependencies...")
    if run_command("pip install -r requirements.txt", "Installing requirements"):
        return True
    else:
        print("⚠️  Try running: pip install --upgrade pip")
        return False

def start_qdrant():
    """Start Qdrant using Docker"""
    print("🐳 Starting Qdrant with Docker...")
    if run_command("docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant", "Starting Qdrant container"):
        time.sleep(5)  # Wait for container to start
        return check_service("http://localhost:6333/health", "Qdrant")
    return False

def start_qdrant_compose():
    """Start Qdrant using Docker Compose"""
    print("🐳 Starting Qdrant with Docker Compose...")
    if run_command("docker-compose up -d qdrant", "Starting Qdrant via Docker Compose"):
        time.sleep(5)  # Wait for container to start
        return check_service("http://localhost:6333/health", "Qdrant")
    return False

def check_ollama():
    """Check if Ollama is running and pull required models"""
    print("🦙 Checking Ollama...")
    
    if not check_service("http://localhost:11434", "Ollama"):
        print("⚠️  Please install and start Ollama:")
        print("   1. Download from: https://ollama.ai/download")
        print("   2. Start Ollama service")
        return False
    
    # Pull required models
    models = ["bge-m3", "llama3.2"]
    for model in models:
        print(f"📥 Pulling {model} model...")
        if not run_command(f"ollama pull {model}", f"Pulling {model}"):
            print(f"⚠️  Failed to pull {model}. Please run manually: ollama pull {model}")
            return False
    
    return True

def check_docker():
    """Check if Docker is available"""
    print("🐳 Checking Docker...")
    if run_command("docker --version", "Checking Docker version"):
        return True
    else:
        print("⚠️  Docker is not installed or not in PATH")
        print("   Please install Docker from: https://docker.com/get-started")
        return False

def create_test_config():
    """Create a test configuration file"""
    config_content = """# Test Configuration
# Copy this file and modify the credentials for your database

# PostgreSQL Configuration
POSTGRES_CONFIG = {
    "database_type": "postgresql",
    "credentials": {
        "host": "localhost",
        "port": 5432,
        "username": "postgres",
        "password": "your_password",
        "database": "your_database"
    }
}

# MySQL Configuration  
MYSQL_CONFIG = {
    "database_type": "mysql",
    "credentials": {
        "host": "localhost",
        "port": 3306,
        "username": "root",
        "password": "your_password",
        "database": "your_database"
    }
}
"""
    
    with open("config_example.py", "w") as f:
        f.write(config_content)
    print("📝 Created config_example.py")

def main():
    """Main setup function"""
    print("🚀 Natural Language to SQL API Setup")
    print("=" * 50)
    
    # Check Python version
    python_version = sys.version_info
    if python_version < (3, 8):
        print("❌ Python 3.8+ is required")
        return
    print(f"✅ Python {python_version.major}.{python_version.minor} detected")
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        return
    
    # Check Docker
    docker_available = check_docker()
    
    # Start Qdrant
    qdrant_started = False
    if docker_available:
        if Path("docker-compose.yml").exists():
            qdrant_started = start_qdrant_compose()
        if not qdrant_started:
            qdrant_started = start_qdrant()
    
    if not qdrant_started:
        print("⚠️  Could not start Qdrant automatically")
        print("   Please start Qdrant manually or install Docker")
    
    # Check Ollama and models
    ollama_ready = check_ollama()
    
    # Create example config
    create_test_config()
    
    print("\n" + "=" * 50)
    print("📋 Setup Summary:")
    print(f"✅ Dependencies: Installed")
    print(f"{'✅' if qdrant_started else '❌'} Qdrant: {'Running' if qdrant_started else 'Not running'}")
    print(f"{'✅' if ollama_ready else '❌'} Ollama: {'Ready' if ollama_ready else 'Not ready'}")
    
    if qdrant_started and ollama_ready:
        print("\n🎉 Setup completed successfully!")
        print("\n🚀 Next steps:")
        print("1. Configure your database credentials in config_example.py")
        print("2. Start the API: python main.py")
        print("3. Test the API: python test_api.py")
        print("4. Visit http://localhost:8000/docs for API documentation")
    else:
        print("\n⚠️  Setup incomplete. Please resolve the issues above.")
        print("\n📚 Manual setup instructions:")
        if not qdrant_started:
            print("   - Start Qdrant: docker run -p 6333:6333 qdrant/qdrant")
        if not ollama_ready:
            print("   - Install Ollama: https://ollama.ai/download")
            print("   - Pull models: ollama pull bge-m3 && ollama pull llama3.2")

if __name__ == "__main__":
    main() 