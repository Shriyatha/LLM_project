import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, Optional
import pandas as pd
import os
from config import validate_file_path
from tools.data_loading import load_data

def visualize_data(file_name: str, plot_type: str, y_col: str, x_col: Optional[str] = None) -> Dict:
    """Creates visualizations from data with robust error handling and consistent returns."""
    try:
        # Load and validate data
        df = load_data(file_name)
        if isinstance(df, str):
            return {"output": f"Data loading error: {df}", "should_stop": True}

        # Standardize column names
        df.columns = df.columns.str.strip().str.lower()
        plot_type = plot_type.lower().strip()
        
        # Plot type normalization mapping
        plot_type_mapping = {
            'histogram': 'hist',
            'hist': 'hist',
            'bar': 'bar',
            'bar chart': 'bar',
            'scatter': 'scatter',
            'scatter plot': 'scatter',
            'line': 'line',
            'line chart': 'line',
            'box': 'box',
            'boxplot': 'box',
            'box plot': 'box'
        }
        
        # Normalize plot type
        normalized_plot_type = plot_type_mapping.get(plot_type)
        if normalized_plot_type is None:
            return {
                "output": f"Unsupported plot type: {plot_type}. Supported: histogram, bar, scatter, line, boxplot",
                "should_stop": True
            }

        y_col = y_col.strip().lower()
        x_col = x_col.strip().lower() if x_col else None

        # Validate columns exist
        required_cols = [y_col] + ([x_col] if x_col else [])
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return {
                "output": f"Columns not found: {missing_cols}. Available: {list(df.columns)}",
                "should_stop": True
            }

        # Validate numeric columns where needed
        numeric_plots = ['hist', 'box', 'scatter', 'line']
        if normalized_plot_type in numeric_plots:
            try:
                df[y_col] = pd.to_numeric(df[y_col], errors='raise')
                if x_col and normalized_plot_type in ['scatter', 'line']:
                    df[x_col] = pd.to_numeric(df[x_col], errors='raise')
            except ValueError as e:
                return {
                    "output": f"Numeric conversion error for {normalized_plot_type} plot: {str(e)}",
                    "should_stop": True
                }

        # Create plot directory if it doesn't exist
        os.makedirs('plots', exist_ok=True)

        # Generate plot filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_filename = f"plots/{normalized_plot_type}_plot_{timestamp}.png"
        
        plt.figure(figsize=(10, 6))
        plt.grid(True, linestyle='--', alpha=0.7)
        
        try:
            if normalized_plot_type == 'bar':
                if not x_col:
                    return {
                        "output": "x_col is required for bar plot",
                        "should_stop": True
                    }
                df.groupby(x_col)[y_col].mean().plot(kind='bar', color='skyblue')
                plt.title(f"Average {y_col} by {x_col}")
                plt.ylabel(y_col)
                
            elif normalized_plot_type == 'hist':
                df[y_col].plot(kind='hist', bins=20, color='lightgreen', edgecolor='black')
                plt.title(f"Distribution of {y_col}")
                plt.xlabel(y_col)
                
            elif normalized_plot_type == 'scatter':
                if not x_col:
                    return {
                        "output": "x_col is required for scatter plot",
                        "should_stop": True
                    }
                df.plot.scatter(x=x_col, y=y_col, color='coral', alpha=0.7)
                plt.title(f"{y_col} vs {x_col}")
                
            elif normalized_plot_type == 'line':
                if not x_col:
                    return {
                        "output": "x_col is required for line plot",
                        "should_stop": True
                    }
                df.plot.line(x=x_col, y=y_col, marker='o', color='royalblue')
                plt.title(f"{y_col} over {x_col}")
                
            elif normalized_plot_type == 'box':
                df[y_col].plot(kind='box', patch_artist=True, 
                             boxprops=dict(facecolor='lightyellow'))
                plt.title(f"Boxplot of {y_col}")
                
            else:
                return {
                    "output": f"Unsupported plot type: {normalized_plot_type}",
                    "should_stop": True
                }

            plt.tight_layout()
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close()

            return {
                "output": f"Successfully created {plot_type} plot: {plot_filename}",
                "plot_file": plot_filename,
                "should_stop": False  # Allow chaining with other operations
            }

        except Exception as plot_error:
            plt.close()
            return {
                "output": f"Plot generation error: {str(plot_error)}",
                "should_stop": True
            }

    except Exception as e:
        return {
            "output": f"Visualization error: {str(e)}",
            "should_stop": True
        }