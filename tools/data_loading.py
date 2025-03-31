import os
import pandas as pd
from typing import Dict, List, Union
from config import DATA_FOLDER, validate_file_path

def list_files() -> Dict:
    """Lists available CSV/Excel files in the data folder."""
    try:
        files = [f for f in os.listdir(DATA_FOLDER) 
                if f.endswith(('.csv', '.xlsx', '.xls'))]
        return {
            "output": f"Available data files: {', '.join(files)}" if files else "No data files found.",
            "files": files,
            "should_stop": True
        }
    except Exception as e:
        return {
            "output": f"Error listing files: {str(e)}",
            "should_stop": True
        }

def load_data(filename: str) -> Union[pd.DataFrame, str]:
    """Improved data loading with better error handling."""
    try:
        filepath = validate_file_path(filename)
        
        if filename.endswith('.csv'):
            return pd.read_csv(filepath)
        elif filename.endswith(('.xlsx', '.xls')):
            return pd.read_excel(filepath)
        return f"Unsupported file format for {filename}"
    except Exception as e:
        return f"Error loading file: {str(e)}"