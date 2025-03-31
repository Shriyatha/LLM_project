import os

DATA_FOLDER = "./data"
os.makedirs(DATA_FOLDER, exist_ok=True)

def validate_file_path(filename: str) -> str:
    """Validates and sanitizes file paths to prevent directory traversal."""
    filename = os.path.basename(filename.strip())
    filepath = os.path.join(DATA_FOLDER, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File '{filename}' not found in {DATA_FOLDER}")
    return filepath