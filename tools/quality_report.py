import json
import numpy as np
import pandas as pd
from typing import Dict
from config import validate_file_path
from tools.data_loading import load_data

def data_quality_report(file_name: str) -> Dict:
    """Generates a comprehensive data quality report."""
    try:
        df = load_data(file_name)
        if isinstance(df, str):
            return {"output": df, "should_stop": True}

        # Convert numpy types to native Python for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return int(obj) if isinstance(obj, np.integer) else float(obj)
            return obj

        # Basic statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        report = {
            'file_name': file_name,
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': {col: int(df[col].isnull().sum()) for col in df.columns},
            'missing_percentage': {col: float(df[col].isnull().mean() * 100) for col in df.columns},
            'duplicate_rows': int(df.duplicated().sum()),
            'data_types': {col: str(dtype) for col, dtype in df.dtypes.to_dict().items()},
            'numeric_stats': {col: {
                'mean': convert_to_serializable(df[col].mean()),
                'median': convert_to_serializable(df[col].median()),
                'std': convert_to_serializable(df[col].std()),
                'min': convert_to_serializable(df[col].min()),
                'max': convert_to_serializable(df[col].max()),
                'skew': convert_to_serializable(df[col].skew()),
                'kurtosis': convert_to_serializable(df[col].kurtosis())
            } for col in numeric_cols},
            'unique_values': {col: int(df[col].nunique()) for col in df.columns},
            'memory_usage': convert_to_serializable(df.memory_usage(deep=True).sum()),
            'sample_data': {col: convert_to_serializable(df[col].head().tolist()) for col in df.columns}
        }

        # Generate a summary of data quality issues
        quality_issues = []
        for col in df.columns:
            issues = []
            null_count = df[col].isnull().sum()
            if null_count > 0:
                issues.append(f"{null_count} missing values ({null_count/len(df)*100:.2f}%)")
            
            if col in numeric_cols:
                if df[col].nunique() == 1:
                    issues.append("Constant value (no variation)")
                elif df[col].nunique() < 5 and df[col].nunique() < len(df)/2:
                    issues.append("Low cardinality (few unique values)")
            
            if issues:
                quality_issues.append({
                    'column': col,
                    'issues': issues,
                    'data_type': str(df[col].dtype)
                })

        report['quality_issues'] = quality_issues
        report['quality_score'] = 100 - (len(quality_issues) / len(df.columns)) * 50

        return {
            "output": f"Data Quality Report for {file_name}:\n{json.dumps(report, indent=2)}",
            "report": report,
            "should_stop": True
        }
    except Exception as e:
        return {"output": f"Quality report error: {str(e)}", "should_stop": True}