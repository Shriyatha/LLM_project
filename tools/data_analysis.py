import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from config import validate_file_path
from tools.data_loading import load_data
import json
#from loguru import logger
from logging_client import log_info, log_debug, log_warning, log_error


def get_columns(file_name: str) -> Dict[str, Any]:
    """List columns and data types with better formatting"""
    log_info(f"Getting columns for file: {file_name}")
    try:
        df = load_data(file_name)
        if isinstance(df, str):
            log_error(f"Data loading error: {df}")
            return {"output": f"Data loading error: {df}", "should_stop": True}
            
        column_info = [f"Columns in {file_name}:"]
        for col, dtype in df.dtypes.items():
            column_info.append(f"- {col}: {str(dtype)}")
            
        sample = df.head(3).to_dict('records')
        log_debug(f"Retrieved sample data: {sample[:1]}")
        
        result = {
            "output": "\n".join(column_info),
            "columns": list(df.columns),
            "dtypes": dict(df.dtypes),
            "sample": sample,
            "should_stop": True
        }
        log_info("Successfully retrieved column information")
        return result
        
    except Exception as e:
        log_error(f"Error getting columns for {file_name}")
        return {"output": f"Error getting columns: {str(e)}", "should_stop": True}

def check_missing_values(file_name: str) -> Dict[str, Any]:
    """Analyze and report missing values in a data file"""
    log_info(f"Checking missing values for file: {file_name}")
    try:
        df = load_data(file_name)
        if isinstance(df, str):
            log_error(f"Data loading error: {df}")
            return {"output": f"Data loading error: {df}", "should_stop": True}

        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        log_debug(f"Missing values calculated: {missing.to_dict()}")
        
        results = []
        for col in df.columns:
            results.append(f"- {col}: {missing[col]} missing ({missing_pct[col]:.1f}%)")
        
        result = {
            "output": f"Missing values in {file_name}:\n" + "\n".join(results),
            "missing_counts": missing.to_dict(),
            "missing_percentages": missing_pct.to_dict(),
            "should_stop": True
        }
        log_info("Missing values analysis completed")
        return result
        
    except Exception as e:
        log_error(f"Error checking missing values in {file_name}")
        return {"output": f"Error checking missing values: {str(e)}", "should_stop": True}

def detect_outliers(file_name: str, column: str, method: str = "iqr", threshold: float = 1.5) -> dict:
    """Detect outliers in specified column of a data file."""
    log_info(f"Detecting outliers in {file_name}, column: {column}, method: {method}, threshold: {threshold}")
    try:
        df = load_data(file_name)
        log_info(f"Data loaded successfully, shape: {df.shape}")
        
        df.columns = df.columns.str.strip().str.lower()
        column = column.strip().lower()
        log_debug(f"Standardized column name: {column}")
        
        if column not in df.columns:
            error_msg = f"Column '{column}' not found. Available columns: {list(df.columns)}"
            log_error(error_msg)
            return {"output": error_msg, "should_stop": True}
        
        try:
            data = pd.to_numeric(df[column], errors='raise')
            log_debug(f"Column converted to numeric successfully")
        except ValueError:
            error_msg = f"Column '{column}' contains non-numeric values"
            log_error(error_msg)
            return {"output": error_msg, "should_stop": True}
        
        if method == "iqr":
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            outliers = df[(data < lower_bound) | (data > upper_bound)]
            stats = f"IQR: {iqr:.2f}, Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]"
            log_debug(f"IQR method stats: {stats}")
        elif method == "zscore":
            z_scores = (data - data.mean()) / data.std()
            outliers = df[abs(z_scores) > threshold]
            stats = f"Mean: {data.mean():.2f}, Std: {data.std():.2f}"
            log_debug(f"Z-score method stats: {stats}")
        else:
            error_msg = f"Invalid method '{method}'. Use 'iqr' or 'zscore'"
            log_error(error_msg)
            return {"output": error_msg, "should_stop": True}
        
        if outliers.empty:
            result = {
                "output": f"No outliers found in '{column}' using {method} (threshold: {threshold})",
                "should_stop": True
            }
            log_info("No outliers found")
            return result
        
        output = (
            f"Found {len(outliers)} outliers in '{column}':\n"
            f"Method: {method} (threshold: {threshold})\n"
            f"Statistics: {stats}\n"
            f"Sample outliers:\n{outliers.head().to_string()}"
        )
        
        result = {
            "output": output,
            "outliers": outliers.to_dict('records'),
            "should_stop": True
        }
        log_info(f"Outlier detection completed: {len(outliers)} outliers found")
        return result
        
    except Exception as e:
        log_error(f"Error detecting outliers in {file_name}")
        return {
            "output": f"Error detecting outliers: {str(e)}",
            "should_stop": True
        }

def show_data_sample(file_name: str, num_rows: int = 5, columns: Optional[List[str]] = None) -> Dict[str, Any]:
    """Show sample rows from a data file"""
    log_info(f"Showing data sample from {file_name}, rows: {num_rows}, columns: {columns}")
    try:
        df = load_data(file_name)
        if isinstance(df, str):
            log_error(f"Data loading error: {df}")
            return {"output": f"Data loading error: {df}", "should_stop": True}
            
        if columns:
            missing = [col for col in columns if col not in df.columns]
            if missing:
                error_msg = f"Columns not found: {missing}"
                log_error(error_msg)
                return {"output": error_msg, "should_stop": True}
            df = df[columns]
            log_debug(f"Filtered to specified columns")
            
        sample = df.head(num_rows)
        sample_list = sample.to_dict('records')
        log_debug(f"Retrieved sample data")
        
        output_lines = [f"First {num_rows} rows of {file_name}:"]
        for record in sample_list:
            output_lines.append("\n".join(f"{k}: {v}" for k, v in record.items()))
            output_lines.append("-" * 20)
            
        result = {
            "output": "\n".join(output_lines),
            "sample": sample_list,
            "should_stop": True
        }
        log_info("Data sample retrieved successfully")
        return result
    except Exception as e:
        log_error(f"Error showing data from {file_name}")
        return {"output": f"Error showing data: {str(e)}", "should_stop": True}

def transform_data(file_name: str, operations: List[str], output_file: Optional[str] = None) -> Dict[str, Any]:
    """Transform data by creating new columns or modifying existing ones."""
    log_info(f"Transforming data in {file_name}, operations: {operations}, output_file: {output_file}")
    try:
        df = load_data(file_name)
        if isinstance(df, str):
            log_error(f"Data loading error: {df}")
            return {"output": f"Data loading error: {df}", "should_stop": True}
            
        for operation in operations:
            try:
                if " as " in operation:
                    expr, new_col = operation.split(" as ")
                    expr = expr.strip()
                    new_col = new_col.strip()
                    df[new_col] = df.eval(expr)
                    log_debug(f"Applied transformation: {operation}")
                else:
                    error_msg = f"Invalid operation format: {operation}"
                    log_error(error_msg)
                    return {"output": error_msg, "should_stop": True}
            except Exception as e:
                error_msg = f"Error executing '{operation}': {str(e)}"
                log_error(error_msg)
                return {"output": error_msg, "should_stop": True}
                
        if output_file:
            save_path = f"data/{output_file}"
            df.to_csv(save_path, index=False)
            result = {
                "output": f"Transformed data saved to {save_path}",
                "file_path": save_path,
                "should_stop": False
            }
            log_info(f"Data transformed and saved to {save_path}")
        else:
            sample = df.head().to_dict('records')
            result = {
                "output": f"Transformation complete. Sample: {json.dumps(sample, indent=2)}",
                "sample": sample,
                "should_stop": True
            }
            log_info("Data transformation completed")
        return result
            
    except Exception as e:
        log_error(f"Error transforming data in {file_name}")
        return {
            "output": f"Transformation error: {str(e)}",
            "should_stop": True
        }

def filter_data(file_name: str, operations: list[dict]) -> dict:
    """Filter data rows based on operations."""
    log_info(f"Filtering data in {file_name}, operations: {operations}")
    try:
        df = load_data(file_name)
        log_debug(f"Data loaded successfully, shape: {df.shape}")
        
        if not operations or not all(isinstance(op, dict) and all(k in op for k in ['column', 'operator', 'value']) for op in operations):
            error_msg = "Each operation must be {column, operator, value}"
            log_error(error_msg)
            return {"output": error_msg, "should_stop": True}

        filtered_df = df.copy()
        for op in operations:
            column = op['column']
            operator = op['operator']
            value = op['value']
            
            if column not in filtered_df.columns:
                error_msg = f"Column '{column}' not found. Available: {list(filtered_df.columns)}"
                log_error(error_msg)
                return {"output": error_msg, "should_stop": True}
            
            try:
                col_data = pd.to_numeric(filtered_df[column], errors='ignore')
                
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
                    error_msg = f"Invalid operator '{operator}'. Use >, <, >=, <=, ==, !="
                    log_error(error_msg)
                    return {"output": error_msg, "should_stop": True}
                log_debug(f"Applied filter: {column} {operator} {value}")
            except Exception as e:
                error_msg = f"Error filtering {column} {operator} {value}: {str(e)}"
                log_error(error_msg)
                return {"output": error_msg, "should_stop": True}

        result = {
            "output": f"Found {len(filtered_df)} matching rows",
            "filtered_data": filtered_df.to_dict('records'),
            "should_stop": True
        }
        
        if not filtered_df.empty:
            result["output"] += ":\n" + filtered_df.head().to_string()
            log_debug(f"Filter results sample: {filtered_df.head().to_string()}")
            
        log_info("Data filtering completed successfully")
        return result
        
    except Exception as e:
        log_error(f"Error filtering data in {file_name}")
        return {
            "output": f"Data processing error: {str(e)}",
            "should_stop": True
        }

def aggregate_data(file_name: str, column: str, agg_funcs: list) -> dict:
    """Perform aggregation operations on a specified column."""
    log_info(f"Aggregating data in {file_name}, column: {column}, functions: {agg_funcs}")
    try:
        df = load_data(file_name)
        if isinstance(df, str):
            log_error(f"Data loading error: {df}")
            return {"output": f"Data loading error: {df}", "should_stop": True}

        df.columns = df.columns.str.strip().str.lower()
        column = column.strip().lower()
        log_debug(f"Standardized column name: {column}")

        if column not in df.columns:
            available_cols = list(df.columns)
            error_msg = f"Column '{column}' not found. Available columns: {available_cols}"
            log_error(error_msg)
            return {"output": error_msg, "should_stop": True}

        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in df.columns:
            if col not in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='ignore')

        if not pd.api.types.is_numeric_dtype(df[column]):
            error_msg = f"Cannot aggregate non-numeric column '{column}'"
            log_error(error_msg)
            return {"output": error_msg, "should_stop": True}

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

        results = {}
        for func in agg_funcs:
            func_lower = func.lower()
            if func_lower not in valid_funcs:
                error_msg = f"Unsupported aggregation function: {func}"
                log_error(error_msg)
                return {"output": error_msg, "should_stop": True}
            
            try:
                if func_lower == 'count':
                    results[func_lower] = df[column].count()
                else:
                    results[func_lower] = getattr(df[column], valid_funcs[func_lower])()
                log_debug(f"Calculated {func_lower}: {results[func_lower]}")
            except Exception as e:
                results[func_lower] = f"Error calculating {func}: {str(e)}"
                log_error(f"Error calculating {func}: {str(e)}")

        output = f"Aggregation results for column '{column}':\n"
        for func, value in results.items():
            output += f"- {func}: {value}\n"

        result = {
            "output": output,
            "results": results,
            "should_stop": True
        }
        log_info("Aggregation completed successfully")
        return result

    except Exception as e:
        log_error(f"Error aggregating data in {file_name}")
        return {"output": f"Aggregation error: {str(e)}", "should_stop": True}

def Sort_data(file_name: str, column: str, order: str, filter_column: Optional[str] = None, 
             filter_operator: Optional[str] = None, filter_value: Optional[Union[str, int, float]] = None) -> Dict:
    """Sorts data by a specified column with optional filtering."""
    log_info(f"Sorting data in {file_name}, column: {column}, order: {order}, filter: {filter_column} {filter_operator} {filter_value}")
    try:
        df = load_data(file_name.strip())
        if isinstance(df, str):
            log_error(f"Data loading error: {df}")
            return {"error": df}

        df.columns = df.columns.str.strip().str.lower()
        column = column.strip().lower()
        log_debug(f"Standardized column name: {column}")

        if column not in df.columns:
            error_msg = f"Column '{column}' not found in {file_name}. Available: {list(df.columns)}"
            log_error(error_msg)
            return {"error": error_msg}

        if filter_column and filter_operator and filter_value is not None:
            filter_column = filter_column.strip().lower()
            if filter_column not in df.columns:
                error_msg = f"Column '{filter_column}' not found in {file_name}."
                log_error(error_msg)
                return {"error": error_msg}

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
                    error_msg = f"Invalid filter operator '{filter_operator}'"
                    log_error(error_msg)
                    return {"error": error_msg}
                log_debug(f"Applied filter: {filter_column} {filter_operator} {filter_value}")
            except Exception as e:
                error_msg = f"Error filtering column '{filter_column}': {str(e)}"
                log_error(error_msg)
                return {"error": error_msg}

        ascending = order.lower() == 'asc'
        df = df.sort_values(by=column, ascending=ascending)
        log_debug(f"Data sorted by {column} in {'ascending' if ascending else 'descending'} order")

        output = f"Sorting done with column '{column}'"
        if not df.empty:
            output += ":\n" + df.head().to_string()
            log_debug(f"Sorted data sample:\n{df.head().to_string()}")

        result = {
            "output": output,
            "filtered_data": df.to_dict('records'),
            "should_stop": True
        }
        log_info("Data sorting completed successfully")
        return result
    except Exception as e:
        log_error(f"Error sorting data in {file_name}: {str(e)}")
        return {"error": f"Sorting error: {str(e)}"}

def summary_statistics(file_name: str) -> dict:
    """Calculate summary statistics for all numeric columns in a file."""
    log_info(f"Calculating summary statistics for {file_name}")
    try:
        df = load_data(file_name)
        if isinstance(df, str):
            log_error(f"Data loading error: {df}")
            return {"output": f"Data loading error: {df}", "should_stop": True}

        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if not numeric_cols:
            log_warning("No numeric columns found")
            return {"output": "No numeric columns found", "should_stop": True}

        results = {}
        for col in numeric_cols:
            results[col] = {
                'mean': df[col].mean(),
                'max': df[col].max(),
                'min': df[col].min(),
                'sum': df[col].sum(),
                'count': df[col].count()
            }
            log_debug(f"Calculated stats for {col}: {results[col]}")

        output = "Summary statistics:\n"
        for col, stats in results.items():
            output += f"\nColumn: {col}\n"
            output += "\n".join([f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}"
                                   for k, v in stats.items()])
            output += "\n"

        result = {
            "output": output,
            "results": results,
            "should_stop": True
        }
        log_info("Summary statistics calculated successfully")
        return result

    except Exception as e:
        log_error(f"Error calculating summary statistics for {file_name}: {str(e)}")
        return {"output": f"Summary statistics error: {str(e)}", "should_stop": True}

def correlation_analysis(file_name: str, cols: List[str]) -> Dict:
    """Calculates correlation between specified columns."""
    log_info(f"Calculating correlations in {file_name}, columns: {cols}")
    try:
        df = load_data(file_name)
        if isinstance(df, str):
            log_error(f"Data loading error: {df}")
            return {"error": df}

        df.columns = df.columns.str.strip().str.lower()
        cols = [col.strip().lower() for col in cols]
        log_debug(f"Standardized column names: {cols}")

        missing_cols = [col for col in cols if col not in df.columns]
        if missing_cols:
            error_msg = f"Columns not found: {missing_cols}. Available: {list(df.columns)}"
            log_error(error_msg)
            return {"error": error_msg}
            
        numeric_cols = [col for col in cols if pd.api.types.is_numeric_dtype(df[col])]
        if not numeric_cols:
            log_warning("No numeric columns found for correlation analysis")
            return {"error": "No numeric columns found for correlation analysis"}
            
        for col in numeric_cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=numeric_cols)
        corr_matrix = df[numeric_cols].corr()
        log_debug(f"Correlation matrix:\n{corr_matrix}")

        result = {
            "status": "success",
            "correlation_matrix": corr_matrix.to_dict(),
            "columns_used": numeric_cols,
            "should_stop": True
        }
        log_info("Correlation analysis completed successfully")
        return result
    except Exception as e:
        log_error(f"Error calculating correlations in {file_name}: {str(e)}")
        return {"error": f"Correlation analysis error: {str(e)}"}