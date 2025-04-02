import streamlit as st
import requests
import pandas as pd
from io import StringIO, BytesIO
import time
import os
import base64
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
from logging_client import log_info, log_error, log_debug, log_warning, log_trace, log_success

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")
DATA_FOLDER = "data"
os.makedirs(DATA_FOLDER, exist_ok=True)

# Initialize session state with logging
if "messages" not in st.session_state:
    log_info("Initializing new chat session")
    st.session_state.messages = [
        {"role": "assistant", "content": "I can analyze your data files. Try:\n\n"
                                        "- 'Show data quality report'\n"
                                        "- 'Plot sales trends'\n"
                                        "- 'Create histogram of ages'"}
    ]

# Page setup
st.set_page_config(page_title="📊 Data Analysis Chatbot", layout="wide")
st.title("📊 Data Analysis Chatbot")

# --- File Management Sidebar ---
with st.sidebar:
    st.header("📂 File Management")
    
    # File uploader with logging
    uploaded_files = st.file_uploader(
        "Upload CSV/Excel files",
        type=["csv", "xlsx"],
        accept_multiple_files=True,
        key="file_uploader"
    )
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                file_path = os.path.join(DATA_FOLDER, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                log_success(f"File uploaded: {uploaded_file.name}")
                st.success(f"Saved: {uploaded_file.name}")
            except Exception as e:
                log_error(f"File upload failed: {str(e)}")
                st.error(f"Error saving {uploaded_file.name}: {str(e)}")
    
    # Display available files with delete option
    available_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(('.csv', '.xlsx'))]
    
    if available_files:
        st.subheader("Your Files")
        cols = st.columns([3, 1])
        files_to_delete = []
        
        for file in available_files:
            cols[0].markdown(f"📄 {file}")
            if cols[1].button("🗑️", key=f"del_{file}"):
                files_to_delete.append(file)
        
        # Handle file deletion with logging
        for file in files_to_delete:
            try:
                file_path = os.path.join(DATA_FOLDER, file)
                os.remove(file_path)
                log_info(f"File deleted: {file}")
                st.success(f"Deleted: {file}")
                st.rerun()
            except Exception as e:
                log_error(f"File deletion failed: {file} - {str(e)}")
                st.error(f"Error deleting {file}: {str(e)}")
    else:
        log_debug("No data files found in directory")
        st.warning("No data files found")
    
    # Test queries button
    if st.button("🧪 Run Test Queries", help="Execute predefined test queries"):
        log_info("Test queries initiated")
        st.session_state.run_tests = True
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        log_info("Chat history cleared")
        st.session_state.messages = []
        st.rerun()

# Custom message display function with logging
def display_message(message: Dict):
    """Display a chat message with support for plots and dataframes."""
    with st.chat_message(message["role"]):
        try:
            st.markdown(message["content"])
            
            # Display plot if exists
            if message.get("plot"):
                try:
                    plot_bytes = base64.b64decode(message["plot"])
                    img = Image.open(BytesIO(plot_bytes))
                    st.image(img, use_column_width=True)
                    
                    # Download button for plot
                    buf = BytesIO()
                    img.save(buf, format="PNG")
                    st.download_button(
                        label="📥 Download Plot",
                        data=buf,
                        file_name="analysis_plot.png",
                        mime="image/png"
                    )
                    log_debug("Plot displayed successfully")
                except Exception as e:
                    log_warning(f"Plot display failed: {str(e)}")
                    st.warning(f"Couldn't display plot: {str(e)}")
            
            # Display dataframe if exists
            if message.get("dataframe"):
                try:
                    df = pd.read_json(StringIO(message["dataframe"]))
                    st.dataframe(df)
                    
                    # Download button for data
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "💾 Download Data",
                        data=csv,
                        file_name="analysis_results.csv",
                        mime="text/csv"
                    )
                    log_debug("Dataframe displayed successfully")
                except Exception as e:
                    log_warning(f"Dataframe display failed: {str(e)}")
                    st.warning(f"Couldn't display data: {str(e)}")
            
            # Show analysis steps if available
            if message.get("steps"):
                with st.expander("🔍 Analysis Steps"):
                    for i, step in enumerate(message["steps"], 1):
                        st.markdown(f"**Step {i}**")
                        st.code(step if isinstance(step, str) else str(step))
                        log_trace(f"Analysis step {i}: {step}")
        
        except Exception as e:
            log_error(f"Message display failed: {str(e)}")
            st.error(f"Error displaying message: {str(e)}")

# --- Main Chat Interface ---
for message in st.session_state.messages:
    display_message(message)

# Handle test queries with logging
if getattr(st.session_state, "run_tests", False):
    st.session_state.run_tests = False
    with st.spinner("Running test queries..."):
        try:
            log_debug("Executing test queries via API")
            response = requests.get(f"{API_URL}/test-queries", timeout=30)
            
            if response.status_code == 200:
                results = response.json().get("results", [])
                log_info(f"Received {len(results)} test results")
                
                for result in results:
                    # Add to chat history
                    st.session_state.messages.append(
                        {"role": "user", "content": result["query"]}
                    )
                    with st.chat_message("user"):
                        st.markdown(result["query"])
                    
                    if result["success"]:
                        assistant_message = {
                            "role": "assistant",
                            "content": result["output"],
                            "plot": result.get("plot"),
                            "dataframe": result.get("dataframe"),
                            "steps": result.get("steps", [])
                        }
                        st.session_state.messages.append(assistant_message)
                        display_message(assistant_message)
                        log_debug(f"Test query succeeded: {result['query']}")
                    else:
                        error_msg = f"Error: {result['error']}"
                        st.session_state.messages.append(
                            {"role": "assistant", "content": error_msg}
                        )
                        with st.chat_message("assistant"):
                            st.error(error_msg)
                        log_warning(f"Test query failed: {result['query']} - {result['error']}")
            else:
                error_msg = f"API returned status {response.status_code}"
                log_error(error_msg)
                st.error(error_msg)
        
        except Exception as e:
            log_error("Test queries execution failed", exc_info=True)
            st.error(f"Failed to run tests: {str(e)}")

# Handle user input with comprehensive logging
if prompt := st.chat_input("Ask about your data..."):
    try:
        log_info(f"New user query: {prompt}")
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                # Include available files in the query context
                context_files = available_files if available_files else ["No files available"]
                prompt_with_context = f"Files: {', '.join(context_files)}\n\nQuery: {prompt}"
                log_debug(f"Query with context: {prompt_with_context}")
                
                # API call with timeout
                log_debug("Making API request")
                response = requests.post(
                    f"{API_URL}/query",
                    json={"query": prompt_with_context},
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    output = data.get("output", "")
                    
                    # Typewriter effect
                    log_debug("Displaying response with typewriter effect")
                    for chunk in output.split():
                        full_response += chunk + " "
                        time.sleep(0.05)
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
                    
                    # Prepare assistant message
                    assistant_message = {
                        "role": "assistant",
                        "content": full_response,
                        "plot": data.get("plot"),
                        "dataframe": data.get("dataframe"),
                        "steps": data.get("steps", [])
                    }
                    
                    # Display all components
                    display_message(assistant_message)
                    
                    # Add to chat history
                    st.session_state.messages.append(assistant_message)
                    log_info("Query processed successfully")
                else:
                    error_msg = f"API Error: {response.text[:200]}..."
                    log_error(f"API response error: {response.status_code} - {error_msg}")
                    st.error(error_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_msg}
                    )
            
            except requests.Timeout:
                error_msg = "Request timed out. Please try again."
                log_warning("API request timeout")
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )
            
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                log_error("Query processing failed", exc_info=True)
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )
    
    except Exception as e:
        log_error("Chat input handling failed", exc_info=True)
        st.error(f"System error: {str(e)}")

# Enhanced CSS with logging
try:
    st.markdown("""
        <style>
            .stChatInput {position: fixed; bottom: 20px;}
            .stChatMessage {padding: 12px; border-radius: 8px;}
            [data-testid="stExpander"] {margin-top: 20px;}
            .stButton button {border: 1px solid #ccc;}
            .st-emotion-cache-1qg05tj {padding: 0.5rem;}
            .plot-container {margin-top: 1rem; margin-bottom: 1rem;}
        </style>
    """, unsafe_allow_html=True)
    log_debug("CSS styles applied successfully")
except Exception as e:
    log_warning(f"CSS application failed: {str(e)}")