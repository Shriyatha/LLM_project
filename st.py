import streamlit as st
import requests
import pandas as pd
from io import StringIO
import time
import os
from typing import List, Dict

# Configuration
API_URL = "http://localhost:8000"
DATA_FOLDER = "data"
os.makedirs(DATA_FOLDER, exist_ok=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "I can analyze your data files. Try:\n\n"
                                        "- 'Show data quality report'\n"
                                        "- 'Check missing values'\n"
                                        "- 'Find outliers in salary column'"}
    ]

# Page setup
st.set_page_config(page_title="📊 Data Analysis Chatbot", layout="wide")
st.title("📊 Data Analysis Chatbot")

# --- File Management Sidebar ---
with st.sidebar:
    st.header("📂 File Management")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload CSV/Excel files",
        type=["csv", "xlsx"],
        accept_multiple_files=True,
        key="file_uploader"
    )
    
    # Process uploaded files
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_path = os.path.join(DATA_FOLDER, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"Saved: {uploaded_file.name}")
    
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
        
        # Handle file deletion
        for file in files_to_delete:
            try:
                os.remove(os.path.join(DATA_FOLDER, file))
                st.success(f"Deleted: {file}")
                st.rerun()
            except Exception as e:
                st.error(f"Error deleting {file}: {str(e)}")
    else:
        st.warning("No data files found")
    
    # Test queries button
    if st.button("🧪 Run Test Queries", help="Execute predefined test queries"):
        st.session_state.run_tests = True
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# --- Main Chat Interface ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle test queries
if getattr(st.session_state, "run_tests", False):
    st.session_state.run_tests = False
    with st.spinner("Running test queries..."):
        try:
            response = requests.get(f"{API_URL}/test-queries")
            if response.status_code == 200:
                results = response.json().get("results", [])
                for result in results:
                    # Add to chat history
                    st.session_state.messages.append(
                        {"role": "user", "content": result["query"]}
                    )
                    with st.chat_message("user"):
                        st.markdown(result["query"])
                    
                    if result["success"]:
                        st.session_state.messages.append(
                            {"role": "assistant", "content": result["output"]}
                        )
                        with st.chat_message("assistant"):
                            st.markdown(result["output"])
                    else:
                        error_msg = f"Error: {result['error']}"
                        st.session_state.messages.append(
                            {"role": "assistant", "content": error_msg}
                        )
                        with st.chat_message("assistant"):
                            st.error(error_msg)
        except Exception as e:
            st.error(f"Failed to run tests: {str(e)}")

# Handle user input
if prompt := st.chat_input("Ask about your data..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # Include available files in the query context
            if available_files:
                prompt_with_context = f"Available files: {', '.join(available_files)}\n\n{prompt}"
            else:
                prompt_with_context = prompt
            
            response = requests.post(
                f"{API_URL}/query",
                json={"query": prompt_with_context},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                output = data.get("output", "")
                
                # Typewriter effect
                for chunk in output.split():
                    full_response += chunk + " "
                    time.sleep(0.05)
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
                
                # Show analysis steps if available
                if steps := data.get("steps"):
                    with st.expander("🔍 Analysis Steps"):
                        for i, step in enumerate(steps, 1):
                            st.markdown(f"**Step {i}**")
                            st.code(step if isinstance(step, str) else str(step))
                
                # Show data if returned
                if "dataframe" in data:
                    try:
                        df = pd.read_json(StringIO(data["dataframe"]))
                        st.subheader("📊 Results")
                        st.dataframe(df)
                        
                        # Download button
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "💾 Download Results",
                            data=csv,
                            file_name="analysis_results.csv",
                            mime="text/csv"
                        )
                    except Exception as e:
                        st.warning(f"Couldn't display data: {str(e)}")
                
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response}
                )
            else:
                error_msg = f"API Error: {response.text[:200]}..."
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )
        
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append(
                {"role": "assistant", "content": error_msg}
            )

# CSS improvements
st.markdown("""
    <style>
        .stChatInput {position: fixed; bottom: 20px;}
        .stChatMessage {padding: 12px; border-radius: 8px;}
        [data-testid="stExpander"] {margin-top: 20px;}
        .stButton button {border: 1px solid #ccc;}
        .st-emotion-cache-1qg05tj {padding: 0.5rem;}
    </style>
""", unsafe_allow_html=True)