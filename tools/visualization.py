import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, Optional, Union, List, Any
import pandas as pd
import os
from pathlib import Path
from tools.data_loading import load_data
from logging_client import log_info, log_debug, log_error

def visualize_data(
    file_name: str, 
    plot_type: str, 
    y_col: str, 
    x_col: Optional[str] = None,
    group_by: Optional[str] = None,
    title: Optional[str] = None,
    **kwargs
) -> Dict[str, Union[str, bool, None]]:
    """
    Creates visualizations from data with robust error handling and centralized logging.
    """
    log_info(f"Starting visualization: {plot_type} plot for {file_name}")
    
    # Initialize response
    response = {
        "output": "",
        "plot_file": None,
        "should_stop": True
    }

    try:
        # 1. Data Loading
        log_debug(f"Loading data from {file_name}")
        df = load_data(file_name)
        if isinstance(df, str):
            log_error(f"Data loading failed: {df}")
            response["output"] = f"Data loading error: {df}"
            return response

        log_info(f"Data loaded successfully. Shape: {df.shape}")
        log_debug(f"Data sample:\n{df.head(2).to_string()}")

        # 2. Input Validation
        if not isinstance(df, pd.DataFrame):
            error_msg = "Loaded data is not a pandas DataFrame"
            log_error(error_msg)
            response["output"] = error_msg
            return response

        # Standardize column names
        df.columns = df.columns.str.strip().str.lower()
        y_col = y_col.strip().lower()
        x_col = x_col.strip().lower() if x_col else None
        group_by = group_by.strip().lower() if group_by else None
        log_debug(f"Standardized columns - y_col: {y_col}, x_col: {x_col}, group_by: {group_by}")

        # 3. Plot Type Validation
        plot_type = plot_type.lower().strip()
        valid_plot_types = {
            'histogram', 'hist', 'bar', 'bar chart', 'scatter', 
            'scatter plot', 'line', 'line chart', 'box', 'boxplot', 
            'box plot', 'pie', 'pie chart'
        }
        
        if plot_type not in valid_plot_types:
            error_msg = f"Unsupported plot type: {plot_type}"
            log_error(error_msg)
            response["output"] = error_msg
            return response

        # Map to canonical plot types
        plot_type_map = {
            'histogram': 'hist', 'hist': 'hist',
            'bar': 'bar', 'bar chart': 'bar',
            'scatter': 'scatter', 'scatter plot': 'scatter',
            'line': 'line', 'line chart': 'line',
            'box': 'box', 'boxplot': 'box', 'box plot': 'box',
            'pie': 'pie', 'pie chart': 'pie'
        }
        plot_type = plot_type_map[plot_type]
        log_debug(f"Canonical plot type: {plot_type}")

        # 4. Column Validation
        required_cols = [y_col]
        if plot_type in ['scatter', 'line'] and not x_col:
            error_msg = f"x_col is required for {plot_type} plot"
            log_error(error_msg)
            response["output"] = error_msg
            return response
        if x_col:
            required_cols.append(x_col)
        if group_by:
            required_cols.append(group_by)

        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            error_msg = f"Columns not found: {missing_cols}"
            log_error(error_msg)
            response["output"] = error_msg
            return response

        # 5. Data Preparation
        try:
            if plot_type in ['hist', 'box', 'scatter', 'line']:
                df[y_col] = pd.to_numeric(df[y_col], errors='raise')
                if x_col:
                    df[x_col] = pd.to_numeric(df[x_col], errors='raise')
            log_debug("Numeric conversion successful")
        except ValueError as e:
            error_msg = f"Numeric conversion error: {str(e)}"
            log_error(error_msg)
            response["output"] = error_msg
            return response

        # 6. Plot Generation
        plot_dir = Path("plots")
        plot_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_filename = plot_dir / f"{plot_type}_plot_{timestamp}.png"
        log_debug(f"Preparing to save plot to: {plot_filename}")
        plt.figure(figsize=kwargs.get('figsize', (10, 6)))
        plt.grid(True, linestyle='--', alpha=0.7)

        # Generate default title if not provided
        if not title:
            title_map = {
                'bar': f"Average {y_col} by {x_col if x_col else group_by}",
                'hist': f"Distribution of {y_col}",
                'scatter': f"{y_col} vs {x_col}",
                'line': f"{y_col} over {x_col}",
                'box': f"Distribution of {y_col}",
                'pie': f"Proportion of {y_col}"
            }
            title = title_map.get(plot_type, f"{plot_type} plot of {y_col}")
        log_debug(f"Using title: {title}")

        try:
            if plot_type == 'bar':
                if group_by:
                    plot_data = df.groupby(group_by)[y_col].mean()
                elif x_col:
                    plot_data = df.groupby(x_col)[y_col].mean()
                else:
                    plot_data = df[y_col].value_counts()
                
                plot_data.plot(kind='bar', color=kwargs.get('color', 'skyblue'), edgecolor='black')
                plt.ylabel(y_col)

            elif plot_type == 'hist':
                df[y_col].plot(kind='hist', bins=kwargs.get('bins', 'auto'), 
                             color=kwargs.get('color', 'lightgreen'), edgecolor='black')
                plt.xlabel(y_col)

            elif plot_type == 'scatter':
                df.plot.scatter(x=x_col, y=y_col, color=kwargs.get('color', 'coral'), 
                              alpha=kwargs.get('alpha', 0.7))

            elif plot_type == 'line':
                df.plot.line(x=x_col, y=y_col, marker=kwargs.get('marker', 'o'),
                           color=kwargs.get('color', 'royalblue'))

            elif plot_type == 'box':
                if group_by:
                    df.boxplot(column=y_col, by=group_by, patch_artist=True,
                             boxprops=dict(facecolor=kwargs.get('color', 'lightyellow')))
                else:
                    df[y_col].plot(kind='box', patch_artist=True,
                                 boxprops=dict(facecolor=kwargs.get('color', 'lightyellow')))

            elif plot_type == 'pie':
                if group_by:
                    plot_data = df.groupby(group_by)[y_col].sum()
                else:
                    plot_data = df[y_col].value_counts()
                
                plot_data.plot(kind='pie', autopct='%1.1f%%',
                             colors=kwargs.get('colors', plt.cm.Pastel1.colors),
                             startangle=90)
                plt.ylabel('')

            plt.title(title)
            plt.tight_layout()
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close()

            success_msg = f"Successfully created {plot_type} plot: {plot_filename}"
            log_info(success_msg)
            response.update({
                "output": success_msg,
                "plot_file": str(plot_filename),
                "should_stop": False  # Allow chaining with other operations
            })

        except Exception as plot_error:
            plt.close()
            error_msg = f"Plot generation error: {str(plot_error)}"
            log_error(error_msg)
            response["output"] = error_msg
            return response

    except Exception as e:
        error_msg = f"Visualization error: {str(e)}"
        log_error(error_msg)
        response["output"] = error_msg
        return response

    return response

def aggregate_and_visualize(
    file_name: str,
    value_col: str,
    group_by: str,
    plot_type: str = "bar",
    **kwargs
) -> Dict[str, Any]:
    """
    Aggregates data and creates visualization with comprehensive logging.
    """
    log_info(f"Starting aggregate_and_visualize for {file_name}")
    log_debug(f"Params - value_col: {value_col}, group_by: {group_by}, plot_type: {plot_type}")

    try:
        # 1. Load and validate data
        log_debug(f"Loading data from {file_name}")
        df = load_data(file_name)
        if isinstance(df, str):
            log_error(f"Data loading failed: {df}")
            return {"output": df, "plot_file": None, "should_stop": True}
        
        log_info(f"Data loaded successfully. Shape: {df.shape}")

        # 2. Standardize column names
        try:
            df.columns = df.columns.str.strip().str.lower()
            value_col = value_col.strip().lower()
            group_by = group_by.strip().lower()
            log_debug(f"Standardized columns - value_col: {value_col}, group_by: {group_by}")
        except AttributeError:
            error_msg = "Invalid column names provided"
            log_error(error_msg)
            return {"output": error_msg, "plot_file": None, "should_stop": True}

        # 3. Validate columns exist
        missing_cols = [col for col in [value_col, group_by] if col not in df.columns]
        if missing_cols:
            error_msg = f"Columns not found: {missing_cols}"
            log_error(error_msg)
            return {
                "output": error_msg,
                "plot_file": None,
                "should_stop": True
            }

        # 4. Convert to numeric if needed
        try:
            df[value_col] = pd.to_numeric(df[value_col], errors='raise')
            log_debug("Numeric conversion successful")
        except ValueError:
            error_msg = f"Could not convert {value_col} to numeric values"
            log_error(error_msg)
            return {
                "output": error_msg,
                "plot_file": None,
                "should_stop": True
            }

        # 5. Aggregate data
        try:
            aggregated = df.groupby(group_by)[value_col].mean().reset_index()
            aggregated.columns = [group_by, f"avg_{value_col}"]
            log_debug(f"Aggregation successful. Result shape: {aggregated.shape}")
        except Exception as e:
            error_msg = f"Aggregation error: {str(e)}"
            log_error(error_msg)
            return {"output": error_msg, "plot_file": None, "should_stop": True}

        # 6. Generate visualization
        try:
            plot_dir = Path("plots")
            plot_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plot_filename = plot_dir / f"agg_{plot_type}_plot_{timestamp}.png"
            log_debug(f"Preparing to save plot to: {plot_filename}")

            plt.figure(figsize=kwargs.get('figsize', (10, 6)))
            plt.grid(True, linestyle='--', alpha=0.7)

            if plot_type == "bar":
                aggregated.plot.bar(
                    x=group_by,
                    y=f"avg_{value_col}",
                    color=kwargs.get('color', 'skyblue'),
                    edgecolor='black'
                )
                plt.title(f"Average {value_col} by {group_by}")
                plt.ylabel(f"Average {value_col}")
            else:
                error_msg = f"Unsupported plot type for aggregation: {plot_type}"
                log_error(error_msg)
                return {
                    "output": error_msg,
                    "plot_file": None,
                    "should_stop": True
                }

            plt.tight_layout()
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close()

            success_msg = f"Successfully created {plot_type} plot: {plot_filename}"
            log_info(success_msg)
            return {
                "output": success_msg,
                "plot_file": str(plot_filename),
                "should_stop": True
            }

        except Exception as e:
            plt.close()
            error_msg = f"Plot generation error: {str(e)}"
            log_error(error_msg)
            return {"output": error_msg, "plot_file": None, "should_stop": True}

    except Exception as e:
        error_msg = f"Unexpected error in aggregate_and_visualize: {str(e)}"
        log_error(error_msg)
        return {"output": error_msg, "plot_file": None, "should_stop": True}
