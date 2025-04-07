# Production-ready Justfile for Data Analysis Assistant
set shell := ["bash", "-ce"]
set dotenv-load

# --- Configuration ---
venv := ".venv"
host := "0.0.0.0"
port := "8000"
keepalive := "300"
ollama_model := "llama3:8b"

# --- Main Commands ---
setup:
    #!/usr/bin/env bash
    set -euo pipefail
    
    echo "🚀 Setting up project environment..."
    
    # 1. Create Python virtualenv
    python3 -m venv {{venv}}
    
    # 2. Install Python dependencies
    source {{venv}}/bin/activate && \
    pip install --upgrade "pip>=24.0" && \
    pip install \
        --no-cache-dir \
        -r requirements.txt
    
    # 3. Install Ollama if missing
    if ! command -v ollama &> /dev/null; then
        echo "📦 Installing Ollama..."
        curl -fsSL https://ollama.com/install.sh | sh
    fi
    
    # 4. Pull the LLM model
    echo "🔍 Pulling {{ollama_model}}..."
    ollama pull {{ollama_model}}
    
    echo "✅ Setup completed successfully"

run:
    #!/usr/bin/env bash
    set -euo pipefail
    
    # Verify setup
    if [ ! -f "{{venv}}/bin/activate" ]; then
        echo "❌ Error: Virtualenv missing. Run 'just setup' first"
        exit 1
    fi
    
    # More flexible model check
    if ! ollama list | grep -q "llama3"; then
        echo "⚠️  No llama3 model found. Pulling..."
        ollama pull {{ollama_model}}
    fi
    
    echo "🚀 Starting services..."
    
    source {{venv}}/bin/activate && \
    (python3 logging_server.py &) && \
    sleep 2 && \
    (uvicorn app:app \
        --host {{host}} \
        --port {{port}} \
        --timeout-keep-alive {{keepalive}} &) && \
    sleep 2 && \
    streamlit run st.py
    
clean:
    #!/usr/bin/env bash
    echo "🧹 Cleaning project..."
    rm -rf \
        {{venv}} \
        __pycache__ \
        .pytest_cache \
        *.log
    echo "✅ Clean complete"