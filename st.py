"""Streamlit application for data analysis with file management capabilities."""
import time
from pathlib import Path

import requests
import streamlit as st

from logging_client import log_debug, log_error, log_info, log_success, log_warning

# Constants
API_URL = "http://localhost:8000"  # Update if your API is hosted elsewhere
DATA_FOLDER = Path("./data")
TIMEOUT = 30  # seconds
HTTP_OK = 200

# Initialize logging
log_info("Initializing Streamlit application")

# Create data directory if it doesn't exist
DATA_FOLDER.mkdir(parents=True, exist_ok=True)

# Page setup
try:
    st.set_page_config(page_title="Data Analysis Chat", layout="wide")
    st.title("📊 Data Analysis Chat")
    log_debug("Page configuration set")
except Exception as e:
    log_error(f"Page setup failed: {e!s}")
    raise

# Initialize chat history with logging
if "messages" not in st.session_state:
    log_info("Initializing new chat session")
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me anything about your data!"},
    ]
    log_debug(f"Initial messages: {st.session_state.messages}")

def handle_file_upload(uploaded_files: list) -> None:
    """Handle file uploads with proper error handling."""
    for uploaded_file in uploaded_files:

        file_path = DATA_FOLDER / uploaded_file.name
        with file_path.open("wb") as f:
            f.write(uploaded_file.getbuffer())
        log_success(f"File uploaded: {uploaded_file.name}")
        st.success(f"Saved: {uploaded_file.name}")


def handle_file_deletion(files_to_delete: list[str]) -> None:
    """Handle file deletion with proper error handling."""
    for file in files_to_delete:
        file_path = DATA_FOLDER / file
        file_path.unlink()
        log_info(f"File deleted: {file}")
        st.success(f"Deleted: {file}")

# File Management Sidebar
with st.sidebar:
    st.header("📂 File Management")

    # File uploader with logging
    uploaded_files = st.file_uploader(
        "Upload CSV/Excel files",
        type=["csv", "xlsx"],
        accept_multiple_files=True,
        key="file_uploader",
    )

    if uploaded_files:
        handle_file_upload(uploaded_files)

    # Display available files with delete option
    available_files = [
        f.name for f in DATA_FOLDER.iterdir()
        if f.suffix in (".csv", ".xlsx")
    ]

    if available_files:
        st.subheader("Your Files")
        cols = st.columns([3, 1])
        files_to_delete = []

        for file in available_files:
            cols[0].markdown(f"📄 {file}")
            if cols[1].button("🗑️", key=f"del_{file}"):
                files_to_delete.append(file)

        if files_to_delete:
            handle_file_deletion(files_to_delete)
    else:
        log_debug("No data files found in directory")
        st.warning("No data files found")

    st.header("API Controls")

    # Health check with logging
    if st.button("🩺 Check API Health"):
        log_info("Initiating API health check")
        try:
            with st.spinner("Checking API health..."):
                response = requests.get(
                    f"{API_URL}/health",
                    timeout=TIMEOUT,
                )

            if response.status_code == HTTP_OK:
                health = response.json()
                status = ("🟢 Healthy" if health["status"] == "healthy"
                          else "🔴 Unhealthy")
                st.success(f"API Status: {status}")
                st.json(health)
                log_success(f"API health check: {status}")
                log_debug(f"Health details: {health}")
            else:
                error_msg = f"API Error: {response.status_code}"
                st.error(error_msg)
                log_error(f"Health check failed: {error_msg}")
        except requests.ConnectionError as e:
            error_msg = "Could not connect to API"
            st.error(error_msg)
            log_error(f"API connection failed: {e!s}")
        except requests.RequestException as e:
            log_error(f"Health check error: {e!s}")
            st.error(f"Unexpected error: {e!s}")

    # Test suite with detailed logging
    if st.button("🧪 Run Test Suite"):
        log_info("Starting test suite execution")
        try:
            with st.spinner("Running test queries..."):
                start_time = time.time()
                response = requests.get(
                    f"{API_URL}/run-test-suite",
                    timeout=TIMEOUT,
                )
                processing_time = time.time() - start_time

                if response.status_code == HTTP_OK:
                    results = response.json()
                    success_count = len([r for r in results if r["success"]])
                    success_msg = (
                        f"Completed {success_count}/{len(results)} tests "
                        f"successfully in {processing_time:.2f}s"
                    )
                    st.success(success_msg)
                    log_success(
                        f"Test suite completed: {success_count}/{len(results)} "
                        "successful",
                    )

                    for result in results:
                        with st.expander(f"Test: {result['query'][:30]}..."):
                            st.markdown(f"**Result:** {result['output'][:200]}...")
                            st.code(f"Time: {result['processing_time']:.2f}s")
                            log_debug(
                                f"Test result - Query: {result['query'][:50]}..., "
                                f"Success: {result['success']}",
                            )
                else:
                    error_msg = f"API Error: {response.status_code}"
                    st.error(error_msg)
                    log_error(f"Test suite failed: {error_msg}")
        except requests.ConnectionError as e:
            error_msg = "Could not connect to API"
            st.error(error_msg)
            log_error(f"API connection failed during test suite: {e!s}")
        except requests.RequestException as e:
            log_error(f"Test suite execution error: {e!s}")
            st.error(f"Unexpected error: {e!s}")

def display_message(message: dict) -> None:
    """Display a chat message with logging.

    Args:
        message: Dictionary containing message content and metadata.

    """
    try:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            log_debug(f"Displayed {message['role']} message")

            # Show steps if available
            if message.get("steps"):
                with st.expander("🔍 See analysis steps"):
                    for step in message["steps"]:
                        st.code(step)
                    log_info("Displayed analysis steps")
    except (KeyError, TypeError) as e:
        log_error(f"Failed to display message: {e!s}")
        st.error(f"Error displaying message: {e!s}")

# Display all messages
for message in st.session_state.messages:
    display_message(message)

# Chat input with comprehensive logging
if prompt := st.chat_input("Ask about your data..."):
    try:
        log_info(f"New user input received: {prompt[:50]}...")

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        display_message({"role": "user", "content": prompt})

        # Prepare API request with file context
        available_files = [
            f.name for f in DATA_FOLDER.iterdir()
            if f.suffix in (".csv", ".xlsx")
        ]
        file_context = (
            f"Available files: {', '.join(available_files)}"
            if available_files
            else "No files available"
        )
        full_prompt = f"{file_context}\n\nQuestion: {prompt}"

        request_data = {
            "query": full_prompt,
            "session_id": f"streamlit_session_{time.time()}",
        }
        log_debug(f"Prepared API request with files: {available_files}")

        # Display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            try:
                # Call API with timing
                log_info("Making API request to /execute-query")
                start_time = time.time()

                response = requests.post(
                    f"{API_URL}/execute-query",
                    json=request_data,
                    timeout=TIMEOUT,
                )

                processing_time = time.time() - start_time
                log_info(f"API response received in {processing_time:.2f} seconds")

                if response.status_code == HTTP_OK:
                    result = response.json()
                    full_response = result.get("output", "No response")
                    steps = result.get("steps", [])

                    # Typewriter effect with logging
                    log_debug("Displaying response with typewriter effect")
                    for chunk in full_response.split():
                        full_response += chunk + " "
                        time.sleep(0.05)
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)

                    # Show steps if available
                    if steps:
                        with st.expander("🔍 See analysis steps"):
                            for i, step in enumerate(steps, 1):
                                st.code(f"Step {i}: {step}")
                                log_info(f"Analysis step {i}: {step[:50]}...")

                    success_msg = (
                        f"Query processed successfully in {processing_time:.2f}s"
                    )
                    log_success(success_msg)
                else:
                    error_msg = (
                        f"API Error {response.status_code}: "
                        f"{response.text[:200]}..."
                    )
                    st.error(error_msg)
                    full_response = error_msg
                    log_error(f"API request failed: {error_msg}")

            except requests.Timeout:
                error_msg = "Request timed out. Please try again."
                st.error(error_msg)
                full_response = error_msg
                log_warning("API request timeout")

            except requests.ConnectionError:
                error_msg = "Could not connect to API"
                st.error(error_msg)
                full_response = error_msg
                log_error("API connection failed")

            except requests.RequestException as e:
                error_msg = f"Unexpected error: {e!s}"
                st.error(error_msg)
                full_response = error_msg
                log_error(f"Query processing failed: {e!s}")

        # Add assistant response to chat history
        assistant_message = {
            "role": "assistant",
            "content": full_response,
            "steps": steps if "steps" in locals() else [],
        }
        st.session_state.messages.append(assistant_message)
        log_debug("Added assistant response to chat history")

    except (requests.RequestException, KeyError, TypeError) as e:
        log_error(f"Chat input processing failed: {e!s}")
        st.error(f"System error: {e!s}")
    except (ValueError, RuntimeError, ConnectionAbortedError) as e:
        log_error(f"Unexpected error in chat processing: {e!s}")
        st.error("An unexpected error occurred. Please try again.")

log_debug("Streamlit application running")
