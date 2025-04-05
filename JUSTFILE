# Set shell mode to work with environment variables
set shell := ["bash", "-c"]

PYTHON := `command -v python3 || command -v python`

setup:
    uv venv .venv_test
    source .venv_test/bin/activate
    uv pip install -r requirements.txt
    uv pip install --upgrade pip

run:
    source .venv_test/bin/activate
    {{PYTHON}} logging_server.py & sleep 2
    uvicorn app:app --host 0.0.0.0 --port 8000 --timeout-keep-alive 300 & sleep 2
    streamlit run st.py
    wait

docs: 
    mkdocs serve