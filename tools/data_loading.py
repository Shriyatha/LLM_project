import os
import pandas as pd
from typing import Dict, Union
from config import DATA_FOLDER, validate_file_path
from logging_client import log_info, log_error, log_warning, log_debug

def list_files() -> Dict:
    """Lists available CSV/Excel files in the data folder with logging."""
    log_info("Listing available data files")
    try:
        files = [f for f in os.listdir(DATA_FOLDER) 
                if f.endswith(('.csv', '.xlsx', '.xls'))]
        
        if not files:
            log_warning("No data files found in directory")
            return {
                "output": "No data files found.",
                "files": [],
                "should_stop": True
            }
        
        log_debug(f"Found {len(files)} files: {files}")
        return {
            "output": f"Available data files: {', '.join(files)}",
            "files": files,
            "should_stop": True
        }
    except Exception as e:
        log_error(f"Error listing files in {DATA_FOLDER}: {str(e)}")
        return {
            "output": f"Error listing files: {str(e)}",
            "should_stop": True
        }

def load_data(filename: str) -> Union[pd.DataFrame, str]:
    """Improved data loading with better error handling and logging."""
    log_info(f"Loading data file: {filename}")
    try:
        filepath = validate_file_path(filename)
        log_debug(f"Validated file path: {filepath}")
        
        if filename.endswith('.csv'):
            log_debug("Loading CSV file")
            df = pd.read_csv(filepath)
        elif filename.endswith(('.xlsx', '.xls')):
            log_debug("Loading Excel file")
            df = pd.read_excel(filepath)
        else:
            error_msg = f"Unsupported file format for {filename}"
            log_error(error_msg)
            return error_msg
            
        log_info(f"Successfully loaded {filename} with shape {df.shape}")
        log_debug(f"Sample data:\n{df.head(2)}")
        return df
        
    except FileNotFoundError as e:
        error_msg = f"File not found: {filename}"
        log_error(error_msg)
        return error_msg
    except pd.errors.EmptyDataError as e:
        error_msg = f"Empty file: {filename}"
        log_error(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"Error loading file {filename}: {str(e)}"
        log_error(error_msg)
        return error_msg
