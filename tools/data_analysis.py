import pandas as pd
import numpy as np
from typing import Dict, List, Optional ,Union
from config import validate_file_path
from tools.data_loading import load_data
from typing import List, Tuple, Any
import json


def get_columns(file_name: str) -> Dict[str, Any]:
    """List columns and data types with better formatting"""
    try:
        df = load_data(file_name)
        if isinstance(df, str):
            return {"output": f"Data loading error: {df}", "should_stop": True}
            
        # Format column information
        column_info = [f"Columns in {file_name}:"]
        for col, dtype in df.dtypes.items():
            column_info.append(f"- {col}: {str(dtype)}")
            
        # Get sample data
        sample = df.head(3).to_dict('records')
        
        return {
            "output": "\n".join(column_info),
            "columns": list(df.columns),
            "dtypes": dict(df.dtypes),
            "sample": sample,
            "should_stop": True  # Stop after showing columns
        }
    except Exception as e:
        return {"output": f"Error getting columns: {str(e)}", "should_stop": True}

def check_missing_values(file_name: str) -> Dict[str, Any]:
    """Analyze and report missing values in a data file"""
    try:
        df = load_data(file_name)
        if isinstance(df, str):
            return {"output": f"Data loading error: {df}", "should_stop": True}

        # Calculate missing values
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        
        # Format results
        results = []
        for col in df.columns:
            results.append(
                f"- {col}: {missing[col]} missing ({missing_pct[col]:.1f}%)"
            )
        
        return {
            "output": f"Missing values in {file_name}:\n" + "\n".join(results),
            "missing_counts": missing.to_dict(),
            "missing_percentages": missing_pct.to_dict(),
            "should_stop": True
        }
    except Exception as e:
        return {"output": f"Error checking missing values: {str(e)}", "should_stop": True}

def detect_outliers(file_name: str, column: str, method: str = "iqr", threshold: float = 1.5) -> dict:
    """Detect outliers in specified column of a data file.
    
    Returns:
        Dictionary containing:
            - output: str, result description
            - should_stop: bool, always True for final answer
    """
    try:
        # Load data
        df = pd.read_csv(file_name)
        
        # Standardize column names
        df.columns = df.columns.str.strip().str.lower()
        column = column.strip().lower()
        
        # Validate column exists
        if column not in df.columns:
            return {
                "output": f"Column '{column}' not found. Available columns: {list(df.columns)}",
                "should_stop": True
            }
        
        # Convert to numeric
        try:
            data = pd.to_numeric(df[column], errors='raise')
        except ValueError:
            return {
                "output": f"Column '{column}' contains non-numeric values",
                "should_stop": True
            }
        
        # Calculate outliers
        if method == "iqr":
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            outliers = df[(data < lower_bound) | (data > upper_bound)]
            stats = f"IQR: {iqr:.2f}, Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]"
        elif method == "zscore":
            z_scores = (data - data.mean()) / data.std()
            outliers = df[abs(z_scores) > threshold]
            stats = f"Mean: {data.mean():.2f}, Std: {data.std():.2f}"
        else:
            return {
                "output": f"Invalid method '{method}'. Use 'iqr' or 'zscore'",
                "should_stop": True
            }
        
        # Format results
        if outliers.empty:
            return {
                "output": f"No outliers found in '{column}' using {method} (threshold: {threshold})",
                "should_stop": True
            }
        
        output = (
            f"Found {len(outliers)} outliers in '{column}':\n"
            f"Method: {method} (threshold: {threshold})\n"
            f"Statistics: {stats}\n"
            f"Sample outliers:\n{outliers.head().to_string()}"
        )
        
        return {
            "output": output,
            "outliers": outliers.to_dict('records'),
            "should_stop": True
        }
        
    except Exception as e:
        return {
            "output": f"Error detecting outliers: {str(e)}",
            "should_stop": True
        }


def show_data_sample(
    file_name: str, 
    num_rows: int = 5,
    columns: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Show sample rows from a data file"""
    try:
        df = load_data(file_name)
        if isinstance(df, str):
            return {"output": f"Data loading error: {df}", "should_stop": True}
            
        if columns:
            missing = [col for col in columns if col not in df.columns]
            if missing:
                return {
                    "output": f"Columns not found: {missing}",
                    "should_stop": True
                }
            df = df[columns]
            
        sample = df.head(num_rows)
        sample_list = sample.to_dict('records')
        
        # Create formatted output
        output_lines = [f"First {num_rows} rows of {file_name}:"]
        for record in sample_list:
            output_lines.append("\n".join(f"{k}: {v}" for k, v in record.items()))
            output_lines.append("-" * 20)
            
        return {
            "output": "\n".join(output_lines),
            "sample": sample_list,
            "should_stop": True  # We typically want to stop after showing data
        }
    except Exception as e:
        return {"output": f"Error showing data: {str(e)}", "should_stop": True}
    

def match_column(available: List[str], target: str) -> str:
    """Case-insensitive column matching"""
    target = target.lower()
    for col in available:
        if col.lower() == target:
            return col
    raise ValueError(f"Column '{target}' not found")

def transform_data(
    file_name: str,
    operations: List[str],
    output_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Transform data by creating new columns or modifying existing ones.
    
    Args:
        file_name: Name of the input file
        operations: List of operations like ["salary*0.1 as bonus"]
        output_file: Optional name for output file
        
    Returns:
        Dictionary with operation results
    """
    try:
        df = load_data(file_name)
        if isinstance(df, str):
            return {"output": f"Data loading error: {df}", "should_stop": True}
            
        for operation in operations:
            try:
                # Simple implementation for operations like "salary*0.1 as bonus"
                if " as " in operation:
                    expr, new_col = operation.split(" as ")
                    expr = expr.strip()
                    new_col = new_col.strip()
                    df[new_col] = df.eval(expr)
                else:
                    return {
                        "output": f"Invalid operation format: {operation}",
                        "should_stop": True
                    }
            except Exception as e:
                return {
                    "output": f"Error executing '{operation}': {str(e)}",
                    "should_stop": True
                }
                
        if output_file:
            save_path = f"data/{output_file}"
            df.to_csv(save_path, index=False)
            return {
                "output": f"Transformed data saved to {save_path}",
                "file_path": save_path,
                "should_stop": False
            }
        else:
            sample = df.head().to_dict('records')
            return {
                "output": f"Transformation complete. Sample: {json.dumps(sample, indent=2)}",
                "sample": sample,
                "should_stop": True
            }
            
    except Exception as e:
        return {
            "output": f"Transformation error: {str(e)}",
            "should_stop": True
        }
    
def filter_data(file_name: str, conditions: list) -> dict:
    """Filter data rows based on conditions.
    
    Args:
        file_name: Name of the file to filter (must be string)
        conditions: List of conditions where each is [column, operator, value]
                  (operator must be one of: '==', '!=', '>', '<', '>=', '<=')
        
    Returns:
        Dictionary with:
            - output: str (result description)
            - filtered_data: list (filtered records)
            - should_stop: bool (always True)
    """
    try:
        # Load data
        df = load_data(file_name)
        if isinstance(df, str):
            return {"output": f"Data loading error: {df}", "should_stop": True}

        # Validate conditions
        if not conditions or not all(isinstance(cond, list) and len(cond) == 3 for cond in conditions):
            return {"output": "Each condition must be [column, operator, value]", "should_stop": True}

        # Apply filters
        filtered_df = df.copy()
        for column, operator, value in conditions:
            if column not in filtered_df.columns:
                return {"output": f"Column '{column}' not found", "should_stop": True}
            
            try:
                col_data = filtered_df[column]
                if operator == '>':
                    filtered_df = filtered_df[col_data > value]
                elif operator == '>=':
                    filtered_df = filtered_df[col_data >= value]
                elif operator == '<':
                    filtered_df = filtered_df[col_data < value]
                elif operator == '<=':
                    filtered_df = filtered_df[col_data <= value]
                elif operator == '==':
                    filtered_df = filtered_df[col_data == value]
                elif operator == '!=':
                    filtered_df = filtered_df[col_data != value]
                else:
                    return {"output": f"Invalid operator: {operator}", "should_stop": True}
            except Exception as e:
                return {"output": f"Filter error on {column}: {str(e)}", "should_stop": True}

        # Format results
        output = f"Found {len(filtered_df)} matching rows"
        if not filtered_df.empty:
            output += ":\n" + filtered_df.head().to_string()

        return {
            "output": output,
            "filtered_data": filtered_df.to_dict('records'),
            "should_stop": True
        }
    except Exception as e:
        return {"output": f"Filter processing error: {str(e)}", "should_stop": True}

def aggregate_data(file_name: str, column: str, agg_funcs: list) -> dict:
    """Perform aggregation operations on a specified column, with automatic numeric conversion.
    
    Args:
        file_name: Name of the file to analyze
        column: Column name to aggregate
        agg_funcs: List of aggregation functions (e.g., ['mean', 'max'])
    
    Returns:
        Dictionary with output, results, and should_stop flag
    """
    try:
        # Load data
        df = load_data(file_name)
        if isinstance(df, str):
            return {"output": f"Data loading error: {df}", "should_stop": True}

        # Standardize column names
        df.columns = df.columns.str.strip().str.lower()
        column = column.strip().lower()

        # Validate column exists
        if column not in df.columns:
            available_cols = list(df.columns)
            return {"output": f"Column '{column}' not found. Available columns: {available_cols}", 
                    "should_stop": True}

        # Convert to numeric if needed (for all numeric columns)
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in df.columns:
            if col not in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='ignore')

        # Check if we have numeric data for aggregation
        if not pd.api.types.is_numeric_dtype(df[column]):
            return {"output": f"Cannot aggregate non-numeric column '{column}'",
                    "should_stop": True}

        # Supported aggregation functions
        valid_funcs = {
            'count': 'count',
            'mean': 'mean',
            'max': 'max',
            'min': 'min',
            'sum': 'sum',
            'std': 'std',
            'median': 'median',
            'nunique': 'nunique'
        }

        # Calculate requested aggregations
        results = {}
        for func in agg_funcs:
            func_lower = func.lower()
            if func_lower not in valid_funcs:
                return {"output": f"Unsupported aggregation function: {func}",
                        "should_stop": True}
            
            try:
                if func_lower == 'count':
                    results[func_lower] = df[column].count()
                else:
                    results[func_lower] = getattr(df[column], valid_funcs[func_lower])()
            except Exception as e:
                results[func_lower] = f"Error calculating {func}: {str(e)}"

        # Format results
        output = f"Aggregation results for column '{column}':\n"
        for func, value in results.items():
            output += f"- {func}: {value}\n"

        return {
            "output": output,
            "results": results,
            "should_stop": True
        }

    except Exception as e:
        return {"output": f"Aggregation error: {str(e)}", "should_stop": True}


def Sort_data(
    file_name: str,
    column: str,
    order: str,
    filter_column: Optional[str] = None,
    filter_operator: Optional[str] = None,  # NEW: Operator for filtering
    filter_value: Optional[Union[str, int, float]] = None
) -> Dict:
    """Sorts data by a specified column in ascending or descending order with optional filtering using operators."""
    try:
        df = load_data(file_name.strip())
        if isinstance(df, str):
            return {"error": df}

        # Standardize column names
        df.columns = df.columns.str.strip().str.lower()
        column = column.strip().lower()

        if column not in df.columns:
            return {"error": f"Column '{column}' not found in {file_name}. Available: {list(df.columns)}"}

        # Apply optional filtering with operators
        if filter_column and filter_operator and filter_value is not None:
            filter_column = filter_column.strip().lower()
            if filter_column not in df.columns:
                return {"error": f"Column '{filter_column}' not found in {file_name}."}

            try:
                filter_value = float(filter_value) if df[filter_column].dtype in ['int64', 'float64'] else filter_value

                if filter_operator == '>':
                    df = df[df[filter_column] > filter_value]
                elif filter_operator == '>=':
                    df = df[df[filter_column] >= filter_value]
                elif filter_operator == '<':
                    df = df[df[filter_column] < filter_value]
                elif filter_operator == '<=':
                    df = df[df[filter_column] <= filter_value]
                elif filter_operator == '==':
                    df = df[df[filter_column] == filter_value]
                elif filter_operator == '!=':
                    df = df[df[filter_column] != filter_value]
                else:
                    return {"error": f"Invalid filter operator '{filter_operator}'"}
            except Exception as e:
                return {"error": f"Error filtering column '{filter_column}': {str(e)}"}

        # Perform sorting
        ascending = order.lower() == 'asc'
        df = df.sort_values(by=column, ascending=ascending)
        output = f"Sorting done with column '{column}'"

        if not df.empty:
            output += ":\n" + df.head().to_string()

        return {
            "output": output,
            "filtered_data": df.to_dict('records'),
            "should_stop": True
        }
    except Exception as e:
        return {"error": f"Sorting error: {str(e)}"}


def summary_statistics(file_name: str) -> dict:
    """Calculate summary statistics for all numeric columns in a file.
    
    Args:
        file_name: Name of the file to analyze
        
    Returns:
        Dictionary with output, results, and should_stop flag
    """
    try:
        df = load_data(file_name)
        if isinstance(df, str):
            return {"output": f"Data loading error: {df}", "should_stop": True}

        # Select only numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if not numeric_cols:
            return {"output": "No numeric columns found", "should_stop": True}

        # Calculate statistics for each column
        results = {}
        for col in numeric_cols:
            results[col] = {
                'mean': df[col].mean(),
                'max': df[col].max(),
                'min': df[col].min(),
                'sum': df[col].sum(),
                'count': df[col].count()
            }

        # Format results
        output = "Summary statistics:\n"
        for col, stats in results.items():
            output += f"\nColumn: {col}\n"
            output += "\n".join([f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}"
                               for k, v in stats.items()])
            output += "\n"

        return {
            "output": output,
            "results": results,
            "should_stop": True
        }

    except Exception as e:
        return {"output": f"Summary statistics error: {str(e)}", "should_stop": True} 

def correlation_analysis(file_name: str, cols: List[str]) -> Dict:
    """Calculates correlation between specified columns (numerical only)."""
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

        # Select only numeric columns
        numeric_cols = [col for col in cols if pd.api.types.is_numeric_dtype(df[col])]
        if not numeric_cols:
            return {"error": "No numeric columns found for correlation analysis"}
            
        # Convert remaining columns to numeric (in case they're stored as strings)
        for col in numeric_cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop rows with NA values in our numeric columns
        df = df.dropna(subset=numeric_cols)

        # Calculate correlation
        corr_matrix = df[numeric_cols].corr()

        return {
            "status": "success",
            "correlation_matrix": corr_matrix.to_dict(),
            "columns_used": numeric_cols,  # Show which columns were actually used
            "should_stop": True
        }
    except Exception as e:
        return {"error": f"Correlation analysis error: {str(e)}"}