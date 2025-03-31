import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, Optional, Union, List
import pandas as pd
import os
from pathlib import Path
from tools.data_loading import load_data
from typing import Dict, Any
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
    Creates visualizations from data with robust error handling and consistent returns.
    
    Args:
        file_name: Name of the data file
        plot_type: Type of plot (histogram, bar, scatter, line, boxplot, pie)
        y_col: Column name for y-axis/data
        x_col: Column name for x-axis (required for scatter/line plots)
        group_by: Column to group data by (for aggregated plots)
        title: Custom plot title
        **kwargs: Additional plot customization options
        
    Returns:
        Dictionary with:
        - output: Status message
        - plot_file: Path to saved plot (if successful)
        - should_stop: Boolean indicating if execution should stop
    """
    
    # Initialize response with default values
    response = {
        "output": "",
        "plot_file": None,
        "should_stop": True  # Default to True to stop on errors
    }

    try:
        # 1. Data Loading
        df = load_data(file_name)
        if isinstance(df, str):
            response["output"] = f"Data loading error: {df}"
            return response

        # 2. Input Validation
        if not isinstance(df, pd.DataFrame):
            response["output"] = "Loaded data is not a pandas DataFrame"
            return response

        # Standardize column names
        df.columns = df.columns.str.strip().str.lower()
        y_col = y_col.strip().lower()
        x_col = x_col.strip().lower() if x_col else None
        group_by = group_by.strip().lower() if group_by else None

        # 3. Plot Type Validation
        plot_type = plot_type.lower().strip()
        valid_plot_types = {
            'histogram', 'hist', 'bar', 'bar chart', 'scatter', 
            'scatter plot', 'line', 'line chart', 'box', 'boxplot', 
            'box plot', 'pie', 'pie chart'
        }
        
        if plot_type not in valid_plot_types:
            response["output"] = (
                f"Unsupported plot type: {plot_type}. "
                f"Supported types: {sorted(valid_plot_types)}"
            )
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

        # 4. Column Validation
        required_cols = [y_col]
        if plot_type in ['scatter', 'line'] and not x_col:
            response["output"] = f"x_col is required for {plot_type} plot"
            return response
        if x_col:
            required_cols.append(x_col)
        if group_by:
            required_cols.append(group_by)

        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            response["output"] = (
                f"Columns not found: {missing_cols}. "
                f"Available columns: {list(df.columns)}"
            )
            return response

        # 5. Data Preparation
        try:
            if plot_type in ['hist', 'box', 'scatter', 'line']:
                df[y_col] = pd.to_numeric(df[y_col], errors='raise')
                if x_col:
                    df[x_col] = pd.to_numeric(df[x_col], errors='raise')
        except ValueError as e:
            response["output"] = f"Numeric conversion error: {str(e)}"
            return response

        # 6. Plot Generation
        plot_dir = Path("plots")
        plot_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_filename = plot_dir / f"{plot_type}_plot_{timestamp}.png"

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

        try:
            if plot_type == 'bar':
                # Handle grouped or ungrouped bar plots
                if group_by:
                    plot_data = df.groupby(group_by)[y_col].mean()
                elif x_col:
                    plot_data = df.groupby(x_col)[y_col].mean()
                else:
                    plot_data = df[y_col].value_counts()
                
                plot_data.plot(
                    kind='bar',
                    color=kwargs.get('color', 'skyblue'),
                    edgecolor='black'
                )
                plt.ylabel(y_col)

            elif plot_type == 'hist':
                df[y_col].plot(
                    kind='hist',
                    bins=kwargs.get('bins', 'auto'),
                    color=kwargs.get('color', 'lightgreen'),
                    edgecolor='black'
                )
                plt.xlabel(y_col)

            elif plot_type == 'scatter':
                df.plot.scatter(
                    x=x_col,
                    y=y_col,
                    color=kwargs.get('color', 'coral'),
                    alpha=kwargs.get('alpha', 0.7)
                )

            elif plot_type == 'line':
                df.plot.line(
                    x=x_col,
                    y=y_col,
                    marker=kwargs.get('marker', 'o'),
                    color=kwargs.get('color', 'royalblue')
                )

            elif plot_type == 'box':
                if group_by:
                    df.boxplot(column=y_col, by=group_by, patch_artist=True,
                             boxprops=dict(facecolor=kwargs.get('color', 'lightyellow')))
                else:
                    df[y_col].plot(
                        kind='box',
                        patch_artist=True,
                        boxprops=dict(facecolor=kwargs.get('color', 'lightyellow'))
                    )

            elif plot_type == 'pie':
                if group_by:
                    plot_data = df.groupby(group_by)[y_col].sum()
                else:
                    plot_data = df[y_col].value_counts()
                
                plot_data.plot(
                    kind='pie',
                    autopct='%1.1f%%',
                    colors=kwargs.get('colors', plt.cm.Pastel1.colors),
                    startangle=90
                )
                plt.ylabel('')

            plt.title(title)
            plt.tight_layout()
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close()

            response.update({
                "output": f"Successfully created {plot_type} plot: {plot_filename}",
                "plot_file": str(plot_filename),
                "should_stop": True  # Allow chaining with other operations
            })

        except Exception as plot_error:
            plt.close()
            response["output"] = f"Plot generation error: {str(plot_error)}"
            return response

    except Exception as e:
        response["output"] = f"Visualization error: {str(e)}"
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
    Complete solution that aggregates data and creates visualization in one step.
    """
    # 1. Load and validate data
    df = load_data(file_name)
    if isinstance(df, str):
        return {"output": df, "plot_file": None, "should_stop": True}
    
    # 2. Standardize column names
    try:
        df.columns = df.columns.str.strip().str.lower()
        value_col = value_col.strip().lower()
        group_by = group_by.strip().lower()
    except AttributeError:
        return {"output": "Invalid column names provided", "plot_file": None, "should_stop": True}

    # 3. Validate columns exist
    missing_cols = [col for col in [value_col, group_by] if col not in df.columns]
    if missing_cols:
        return {
            "output": f"Columns not found: {missing_cols}. Available: {list(df.columns)}",
            "plot_file": None,
            "should_stop": True
        }

    # 4. Convert to numeric if needed
    try:
        df[value_col] = pd.to_numeric(df[value_col], errors='raise')
    except ValueError:
        return {
            "output": f"Could not convert {value_col} to numeric values",
            "plot_file": None,
            "should_stop": True
        }

    # 5. Aggregate data
    try:
        aggregated = df.groupby(group_by)[value_col].mean().reset_index()
        aggregated.columns = [group_by, f"avg_{value_col}"]
    except Exception as e:
        return {"output": f"Aggregation error: {str(e)}", "plot_file": None, "should_stop": True}

    # 6. Generate visualization
    try:
        plot_dir = Path("plots")
        plot_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_filename = plot_dir / f"agg_{plot_type}_plot_{timestamp}.png"

        plt.figure(figsize=kwargs.get('figsize', (10, 6)))
        plt.grid(True, linestyle='--', alpha=0.7)

        # Create the plot
        if plot_type == "bar":
            aggregated.plot.bar(
                x=group_by,
                y=f"avg_{value_col}",
                color=kwargs.get('color', 'skyblue'),
                edgecolor='black'
            )
            plt.title(f"Average {value_col} by {group_by}")
        else:
            return {
                "output": f"Unsupported plot type for aggregation: {plot_type}",
                "plot_file": None,
                "should_stop": True
            }

        plt.ylabel(f"Average {value_col}")
        plt.tight_layout()
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()

        return {
            "output": f"Successfully created {plot_type} plot: {plot_filename}",
            "plot_file": str(plot_filename),
            "should_stop": False
        }

    except Exception as e:
        plt.close()
        return {"output": f"Plot generation error: {str(e)}", "plot_file": None, "should_stop": True}
