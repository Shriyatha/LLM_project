import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional
from config import validate_file_path
from tools.data_loading import load_data

def time_series_analysis(file_name: str, date_col: str, value_col: str, freq: str = 'D') -> Dict:
    """Analyzes time series data with improved date handling."""
    try:
        df = load_data(file_name)
        if isinstance(df, str):
            return {"error": df}

        # Standardize column names
        df.columns = df.columns.str.strip().str.lower()
        date_col = date_col.strip().lower()
        value_col = value_col.strip().lower()

        # Validate columns
        missing_cols = [col for col in [date_col, value_col] if col not in df.columns]
        if missing_cols:
            return {"error": f"Columns not found: {missing_cols}. Available: {list(df.columns)}"}

        # Convert to datetime and numeric
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
        df = df.dropna(subset=[date_col, value_col])

        # Set date as index and resample
        df.set_index(date_col, inplace=True)
        resampled = df[value_col].resample(freq).mean()

        # Create visualization
        plt.figure(figsize=(12, 6))
        resampled.plot(title=f"Time Series of {value_col} ({freq})")
        plot_filename = f"timeseries_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
        plt.savefig(plot_filename)
        plt.close()

        # Calculate statistics
        stats = {
            'mean': float(resampled.mean()),
            'max': float(resampled.max()),
            'min': float(resampled.min()),
            'last': float(resampled.iloc[-1]),
            'change': float(resampled.iloc[-1] - resampled.iloc[0]),
            'pct_change': float(((resampled.iloc[-1] - resampled.iloc[0]) / resampled.iloc[0]) * 100)
        }

        return {
            "status": "success",
            "output": f"Time series analysis saved as {plot_filename}. Statistics: {stats}",
            "plot_file": plot_filename,
            "stats": stats,
            "should_stop": True
        }
    except Exception as e:
        return {"error": f"Time series analysis error: {str(e)}"}