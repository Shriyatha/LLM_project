import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from config import validate_file_path
from tools.data_loading import load_data
from typing import List, Tuple, Any

def get_columns(file_name: str) -> list:
    """Return column names of a CSV file."""
    df = pd.read_csv(file_name)
    return df.columns.tolist()

def show_sample_rows(file_name: str, num_rows: int = 3) -> str:
    """
    Displays sample rows from a CSV file.
    Input: {'file_name': 'str', 'num_rows': 'int'}
    """
    try:
        df = pd.read_csv(file_name)
        sample = df.head(num_rows).to_string()
        return f"Sample from {file_name}:\n{sample}"
    except Exception as e:
        return f"Error reading {file_name}: {str(e)}"
    
def filter_data(file_name: str, conditions: List[List]) -> Dict:
    """Enhanced filter with better type handling."""
    try:
        df = load_data(file_name)  # Load the dataset

        if isinstance(df, str):  # If load_data returns an error string
            return {"output": df, "should_stop": True}

        # Standardize column names
        df.columns = df.columns.str.strip().str.lower()

        for condition in conditions:
            if len(condition) != 3:
                return {"error": f"Invalid condition format: {condition}. Expected [column, operator, value]"}

            column, operator, value = condition
            column = column.strip().lower()

            if column not in df.columns:
                return {"error": f"Column '{column}' not found. Available: {list(df.columns)}"}

            try:
                # Convert numeric columns for comparisons
                if operator in ['>', '<', '>=', '<=']:
                    value = float(value)
                    df[column] = pd.to_numeric(df[column], errors='coerce')

                # Apply filtering
                if operator == '==': 
                    df = df[df[column] == value]
                elif operator == '!=': 
                    df = df[df[column] != value]
                elif operator == '>': 
                    df = df[df[column] > value]
                elif operator == '<': 
                    df = df[df[column] < value]
                elif operator == '>=': 
                    df = df[df[column] >= value]
                elif operator == '<=': 
                    df = df[df[column] <= value]
                elif operator.lower() == 'contains':
                    df = df[df[column].astype(str).str.contains(str(value), case=False, na=False)]
                else:
                    return {"error": f"Unsupported operator: {operator}"}
            except Exception as e:
                return {"error": f"Error applying filter {column} {operator} {value}: {str(e)}"}

        # Handle empty results
        filtered_records = df.to_dict(orient="records")
        if not filtered_records:
            return {"output": "No matching records found.", "count": 0, "should_stop": True}

        return {
            "output": f"Found {len(filtered_records)} matching records, Sample: {filtered_records[:5]}",
            "count": len(filtered_records),
            "sample": filtered_records[:5],  # Return only first 5 records
            "should_stop": True
        }

    except Exception as e:
        return {"output": f"Filter error: {str(e)}", "should_stop": True}

def aggregate_data(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate statistics on a column.
    Input should be a dictionary with 'file_name' (str), 'column' (str), and 'agg_funcs' (list of str).
    """
    file_name = input_dict['file_name']
    column = input_dict['column']
    agg_funcs = input_dict['agg_funcs']

    try:
        df = pd.read_csv(file_name)
        if column not in df.columns:
            available_columns = list(df.columns)
            return {
                "status": "error",
                "error": f"Column '{column}' not found. Available columns: {available_columns}",
                "available_columns": available_columns
            }

        results = {}
        for func in agg_funcs:
            if func == "mean":
                results["mean"] = df[column].mean()
            elif func == "sum":
                results["sum"] = df[column].sum()
            # Add other aggregation functions as needed

        return {
            "status": "success",
            "results": results,
            "output": f"Aggregation results for '{column}': {results}",
            "should_stop": True
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}
    

def correlation_analysis(file_name: str, cols: List[str]) -> Dict:
    """Calculates correlation between specified columns."""
    try:
        df = load_data(file_name)
        if isinstance(df, str):
            return {"error": df}

        # Standardize column names
        df.columns = df.columns.str.strip().str.lower()
        cols = [col.strip().lower() for col in cols]

        # Validate columns
        missing_cols = [col for col in cols if col not in df.columns]
        if missing_cols:
            return {"error": f"Columns not found: {missing_cols}. Available: {list(df.columns)}"}

        # Convert to numeric
        for col in cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=cols)

        # Calculate correlation
        corr_matrix = df[cols].corr()

        return {
            "status": "success",
            "correlation_matrix": corr_matrix.to_dict(),
            "should_stop": True
        }
    except Exception as e:
        return {"error": f"Correlation analysis error: {str(e)}"}