"""Enhanced Streamlit application for data analysis with superior UX."""
import base64
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests
import streamlit as st

from logging_client import log_debug, log_error, log_info, log_success, log_warning

# Constants
API_URL = "http://localhost:8000"  # Update if your API is hosted elsewhere
DATA_FOLDER = Path("./data")
PLOTS_FOLDER = Path("./plots")
TIMEOUT = 30  # seconds
HTTP_OK = 200
SUPPORTED_PLOT_FORMATS = [".png", ".jpg", ".jpeg", ".svg"]
MAX_PLOTS_TO_KEEP = 20
MAX_PLOTS_TO_DISPLAY = 5
MAX_FILE_SIZE_MB = 10  # Maximum file size to accept in MB

# Initialize logging
log_info("Initializing Enhanced Streamlit application")

# Create necessary directories if they don't exist
DATA_FOLDER.mkdir(parents=True, exist_ok=True)
PLOTS_FOLDER.mkdir(parents=True, exist_ok=True)

def convert_bytes_to_mb(size_bytes: int) -> float:
    """Convert bytes to megabytes."""
    return size_bytes / (1024 * 1024)

def cleanup_plots(max_plots: int = MAX_PLOTS_TO_KEEP) -> None:
    """Keep only the most recent plots to avoid clutter."""
    try:
        plot_files = sorted(PLOTS_FOLDER.iterdir(), key=os.path.getmtime, reverse=True)
        for old_plot in plot_files[max_plots:]:
            try:
                old_plot.unlink()
                log_info(f"Cleaned up old plot: {old_plot.name}")
            except Exception as e:
                log_error(f"Failed to delete {old_plot.name}: {e}")
    except Exception as e:
        log_error(f"Plot cleanup failed: {e}")

cleanup_plots()

# Page setup
try:
    st.set_page_config(
        page_title="Data Analysis Chat",
        layout="wide",
        page_icon="📊",
        initial_sidebar_state="expanded",
    )
    st.title("📊 Data Analysis Chat")
    st.caption("Analyze your data with natural language queries")
    log_debug("Page configuration set")
except Exception as e:
    log_error(f"Page setup failed: {e!s}")
    raise

# Custom CSS for better UI
def inject_custom_css() -> None:
    """Inject custom CSS for better styling."""
    st.markdown("""
    <style>
        .stChatMessage {
            padding: 12px 16px;
            border-radius: 8px;
            margin-bottom: 12px;
        }
        .stChatMessage.user {
            background-color: #f0f2f6;
        }
        .stChatMessage.assistant {
            background-color: #ffffff;
            border: 1px solid #e1e4e8;
        }
        .plot-expander {
            margin-top: 12px;
        }
        .file-uploader {
            margin-bottom: 20px;
        }
        .sidebar .sidebar-content {
            padding: 1rem;
        }
        .stButton>button {
            border-radius: 4px;
            padding: 0.25rem 0.75rem;
        }
        .stDownloadButton>button {
            width: 100%;
            justify-content: center;
        }
        .plot-thumbnail {
            max-width: 100%;
            border-radius: 4px;
            cursor: pointer;
            transition: transform 0.2s;
        }
        .plot-thumbnail:hover {
            transform: scale(1.02);
        }
    </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# Initialize chat history with logging
if "messages" not in st.session_state:
    log_info("Initializing new chat session")
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! I'm your data analysis assistant. You can:\n\n"
                      "1. Upload data files (CSV/Excel) in the sidebar\n"
                      "2. Ask questions about your data\n"
                      "3. View generated plots and analysis steps\n\n"
                      "What would you like to analyze today?",
        },
    ]
    log_debug(f"Initial messages: {st.session_state.messages}")

def save_plot_from_response(response_data: Dict[str, Any]) -> Optional[Path]:
    """Save plot from API response and return its path."""
    try:
        if isinstance(response_data, dict) and "plot" in response_data:
            # Handle base64 encoded plot
            plot_data = response_data["plot"]
            plot_format = response_data.get("plot_format", ".png")

            if not plot_format.startswith("."):
                plot_format = f".{plot_format}"

            if plot_format not in SUPPORTED_PLOT_FORMATS:
                plot_format = ".png"

            timestamp = int(time.time())
            plot_name = f"plot_{timestamp}{plot_format}"
            plot_path = PLOTS_FOLDER / plot_name

            if plot_data.startswith("data:"):
                # Handle data URI
                header, plot_data = plot_data.split(",", 1)
                plot_bytes = base64.b64decode(plot_data)
                with open(plot_path, "wb") as f:
                    f.write(plot_bytes)
            else:
                # Direct base64 data
                plot_bytes = base64.b64decode(plot_data)
                with open(plot_path, "wb") as f:
                    f.write(plot_bytes)

            log_success(f"Saved plot: {plot_name}")
            return plot_path

        if isinstance(response_data, dict) and "plot_file" in response_data:
            # Handle file path from API
            plot_file = response_data["plot_file"]
            plot_name = Path(plot_file).name
            plot_path = PLOTS_FOLDER / plot_name

            # If the API provided the actual file content
            if "plot_content" in response_data:
                with open(plot_path, "wb") as f:
                    f.write(base64.b64decode(response_data["plot_content"]))
            # Assume the file is already in the plots folder
            elif not plot_path.exists():
                log_warning(f"Plot file not found: {plot_path}")
                return None

            return plot_path

    except Exception as e:
        log_error(f"Failed to save plot: {e}")
        return None

def display_plot(plot_path: Path, expander_title: str = "📊 View Plot") -> None:
    """Display a plot with download and expand options."""
    try:
        if plot_path.exists():
            col1, col2 = st.columns([4, 1])

            with col1:
                if plot_path.suffix == ".svg":
                    with open(plot_path) as f:
                        svg = f.read()
                    st.image(svg, use_column_width=True)
                else:
                    st.image(str(plot_path), use_column_width=True)

            with col2:
                # Create download button
                mime_types = {
                    ".png": "image/png",
                    ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg",
                    ".svg": "image/svg+xml",
                }
                mime_type = mime_types.get(plot_path.suffix.lower(), "application/octet-stream")

                with open(plot_path, "rb") as f:
                    st.download_button(
                        label="Download",
                        data=f,
                        file_name=plot_path.name,
                        mime=mime_type,
                        key=f"dl_{plot_path.name}",
                        use_container_width=True,
                    )

            log_success(f"Displayed plot: {plot_path.name}")
        else:
            st.warning(f"Plot file not found at: {plot_path}")
            log_warning(f"Plot not found: {plot_path}")
    except Exception as e:
        st.error(f"Failed to display plot: {e!s}")
        log_error(f"Plot display failed: {e!s}")

def display_message(message: Dict[str, Any]) -> None:
    """Display a chat message with logging and optional plot."""
    try:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            log_debug(f"Displayed {message['role']} message")

            # Show plot if available
            if message.get("plot_path"):
                plot_path = Path(message["plot_path"])
                with st.expander("📊 View Plot", expanded=True):
                    display_plot(plot_path)

            # Show steps if available
            if message.get("steps"):
                with st.expander("🔍 See analysis steps", expanded=False):
                    for i, step in enumerate(message["steps"], 1):
                        st.markdown(f"**Step {i}**")
                        st.code(step, language="python")
                    log_info("Displayed analysis steps")
    except Exception as e:
        log_error(f"Failed to display message: {e!s}")
        st.error(f"Error displaying message: {e!s}")

def handle_file_upload() -> None:
    """Handle file uploads with validation and feedback."""
    uploaded_files = st.sidebar.file_uploader(
        "Upload CSV/Excel files (Max 10MB each)",
        type=["csv", "xlsx"],
        accept_multiple_files=True,
        key="file_uploader",
        help="Upload your data files for analysis",
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_size_mb = convert_bytes_to_mb(uploaded_file.size)
            if file_size_mb > MAX_FILE_SIZE_MB:
                st.sidebar.error(f"File {uploaded_file.name} is too large ({file_size_mb:.2f}MB). Max size is {MAX_FILE_SIZE_MB}MB.")
                continue

            file_path = DATA_FOLDER / uploaded_file.name
            try:
                with st.spinner(f"Saving {uploaded_file.name}..."):
                    with file_path.open("wb") as f:
                        f.write(uploaded_file.getbuffer())
                st.sidebar.success(f"Saved: {uploaded_file.name} ({file_size_mb:.2f}MB)")
                log_success(f"File uploaded: {uploaded_file.name}")
            except Exception as e:
                st.sidebar.error(f"Failed to save {uploaded_file.name}: {e!s}")
                log_error(f"File upload failed: {uploaded_file.name} - {e}")

def display_file_management() -> None:
    """Display file management interface in sidebar."""
    st.sidebar.header("📂 File Management")
    handle_file_upload()

    # Display available files with delete option
    available_files = [f.name for f in DATA_FOLDER.iterdir() if f.suffix in (".csv", ".xlsx")]
    if available_files:
        st.sidebar.subheader("Your Files")
        st.sidebar.caption("Click 🗑️ to delete a file")

        for file in available_files:
            cols = st.sidebar.columns([4, 1])
            cols[0].markdown(f"📄 {file}")

            if cols[1].button("🗑️", key=f"del_{file}", help=f"Delete {file}"):
                try:
                    file_path = DATA_FOLDER / file
                    file_size_mb = convert_bytes_to_mb(file_path.stat().st_size)
                    file_path.unlink()
                    st.sidebar.success(f"Deleted: {file}")
                    log_info(f"File deleted: {file}")
                    time.sleep(0.5)  # Let the user see the success message
                    st.rerun()
                except Exception as e:
                    st.sidebar.error(f"Failed to delete {file}: {e!s}")
                    log_error(f"File deletion failed: {file} - {e}")
    else:
        st.sidebar.info("No data files found. Upload files to get started.")
        log_debug("No data files found in directory")

def display_plot_management() -> None:
    """Display plot management interface in sidebar."""
    st.sidebar.header("📊 Plot Management")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.sidebar.button("🔄 Refresh", help="Refresh the plot list"):
            st.rerun()

    with col2:
        if st.sidebar.button("🧹 Clear All", help="Delete all generated plots"):
            try:
                plot_count = len(list(PLOTS_FOLDER.glob("*")))
                for plot_file in PLOTS_FOLDER.glob("*"):
                    plot_file.unlink()
                st.sidebar.success(f"Cleared {plot_count} plots")
                log_info(f"Cleared {plot_count} plot files")
                time.sleep(0.5)
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Failed to clear plots: {e!s}")
                log_error(f"Plot cleanup failed: {e}")

    # Display available plots as thumbnails
    plot_files = sorted(PLOTS_FOLDER.iterdir(), key=os.path.getmtime, reverse=True)
    if plot_files:
        st.sidebar.subheader(f"Recent Plots (Last {MAX_PLOTS_TO_DISPLAY})")

        for plot_file in plot_files[:MAX_PLOTS_TO_DISPLAY]:
            try:
                # Create a thumbnail preview
                cols = st.sidebar.columns([3, 1])

                with cols[0]:
                    if plot_file.suffix == ".svg":
                        with open(plot_file) as f:
                            svg = f.read()
                        st.image(svg, use_column_width=True)
                    else:
                        st.image(str(plot_file), use_column_width=True)

                with cols[1]:
                    if st.button("👀", key=f"view_{plot_file.name}", help=f"View {plot_file.name}"):
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"Showing plot: {plot_file.name}",
                            "plot_path": str(plot_file),
                        })
                        st.rerun()
            except Exception as e:
                st.sidebar.error(f"Couldn't display {plot_file.name}")
                log_error(f"Failed to display plot thumbnail {plot_file.name}: {e}")
    else:
        st.sidebar.info("No plots generated yet. Ask a question to generate plots.")

def process_user_query(prompt: str) -> None:
    """Process user query and display response."""
    try:
        log_info(f"New user input received: {prompt[:50]}...")

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        display_message({"role": "user", "content": prompt})

        # Prepare API request with file context
        available_files = [f.name for f in DATA_FOLDER.iterdir() if f.suffix in (".csv", ".xlsx")]
        file_context = f"Available files: {', '.join(available_files)}" if available_files else "No files available"
        full_prompt = f"{file_context}\n\nQuestion: {prompt}"

        request_data = {
            "query": full_prompt,
            "session_id": f"streamlit_session_{time.time()}",
            "plot_format": "png",  # Default format
        }
        log_debug(f"Prepared API request with files: {available_files}")

        # Display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            status_placeholder = st.empty()
            full_response = ""
            plot_path = None
            steps = []

            try:
                # Show loading indicator
                with status_placeholder.status("Analyzing your data...", expanded=True) as status:
                    st.write("🔍 Understanding your question...")
                    time.sleep(0.5)

                    # Call API with timing
                    log_info("Making API request to /execute-query")
                    start_time = time.time()
                    st.write("📡 Connecting to analysis engine...")

                    response = requests.post(
                        f"{API_URL}/execute-query",
                        json=request_data,
                        timeout=TIMEOUT,
                    )

                    processing_time = time.time() - start_time
                    log_info(f"API response received in {processing_time:.2f} seconds")

                    if response.status_code == HTTP_OK:
                        # Handle response
                        try:
                            result = response.json()
                            full_response = result.get("output", "No response")
                            steps = result.get("steps", [])
                            plot_path = save_plot_from_response(result)
                        except ValueError:
                            result = response.text
                            full_response = result

                        # Display the text response
                        message_placeholder.markdown(full_response)

                        status.update(label="Analysis complete!", state="complete", expanded=False)
                        success_msg = f"Query processed successfully in {processing_time:.2f}s"
                        log_success(success_msg)
                    else:
                        error_msg = f"API Error {response.status_code}: {response.text[:200]}..."
                        status.error(error_msg)
                        full_response = error_msg
                        log_error(f"API request failed: {error_msg}")

            except requests.Timeout:
                error_msg = "Request timed out. Please try again with a simpler query."
                status_placeholder.error(error_msg)
                full_response = error_msg
                log_warning("API request timeout")

            except requests.ConnectionError:
                error_msg = "Could not connect to analysis service. Please try again later."
                status_placeholder.error(error_msg)
                full_response = error_msg
                log_error("API connection failed")

            except requests.RequestException as e:
                error_msg = f"Unexpected error: {e!s}"
                status_placeholder.error(error_msg)
                full_response = error_msg
                log_error(f"Query processing failed: {e!s}")

        # Add assistant response to chat history
        assistant_message = {
            "role": "assistant",
            "content": full_response,
            "steps": steps,
            "plot_path": str(plot_path) if plot_path else None,
        }
        st.session_state.messages.append(assistant_message)
        log_debug("Added assistant response to chat history")

    except Exception as e:
        log_error(f"Chat input processing failed: {e!s}")
        st.error(f"System error: {e!s}")

# Main application layout
def main():
    """Main application layout and logic."""
    # Sidebar sections
    with st.sidebar:
        display_file_management()
        st.sidebar.divider()
        display_plot_management()

    # Main chat area
    st.subheader("Chat with your Data")

    # Display all messages
    for message in st.session_state.messages:
        display_message(message)

    # Chat input with comprehensive logging
    if prompt := st.chat_input("Ask about your data...", key="chat_input"):
        process_user_query(prompt)

if __name__ == "__main__":
    main()
    log_debug("Streamlit application running")
