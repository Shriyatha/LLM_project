# Justfile for reliable project setup and execution
set shell := ["bash", "-c"]
set dotenv-load

# System-agnostic setup that works on fresh Ubuntu/MacOS
setup:
    # Create isolated virtual environment
    if ! command -v python3.10 >/dev/null 2>&1; then \
        echo "Installing Python 3.10..."; \
        if command -v apt-get >/dev/null; then \
            sudo apt-get update && sudo apt-get install -y python3.10 python3.10-venv; \
        elif command -v brew >/dev/null; then \
            brew install python@3.10; \
        else \
            echo "Unsupported system - please install Python 3.10 manually"; \
            exit 1; \
        fi; \
    fi
    
    python3.10 -m venv .venv && \
    . .venv/bin/activate && \
    pip install --upgrade pip wheel && \
    pip install -r requirements.txt && \
    echo "Setup completed successfully"

run:
    # Verify virtual environment exists
    if [ ! -d ".venv" ]; then \
        echo "Virtual environment not found. Run 'just setup' first"; \
        exit 1; \
    fi
    
    # Create logs directory if it doesn't exist
    mkdir -p logs
    
    # Start services
    . .venv/bin/activate && \
    (python -m logging_server > logs/server.log 2>&1 &) && \
    echo "Started logging server" && \
    sleep 2 && \
    (uvicorn app:app --host 0.0.0.0 --port 8000 --timeout-keep-alive 300 > logs/api.log 2>&1 &) && \
    echo "Started API server" && \
    sleep 2 && \
    streamlit run st.py

# Clean environment with process safety
clean:
    pkill -f "logging_server" || true
    pkill -f "uvicorn main:app" || true
    pkill -f "streamlit run st.py" || true
    rm -rf .venv
    rm -rf logs
    mkdir -p logs

# Health check
check:
    . .venv/bin/activate && \
    python -c "import sys; from importlib.util import find_spec; sys.exit(0 if all(find_spec(pkg) for pkg in ['uvicorn', 'streamlit']) else 1)" && \
    echo "All dependencies are properly installed"