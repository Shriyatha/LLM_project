import json
import numpy as np
import pandas as pd
from typing import Dict
from config import validate_file_path
from tools.data_loading import load_data
from logging_client import log_info, log_debug, log_critical, log_error, log_warning

def data_quality_report(file_name: str) -> Dict:
    """Generates a comprehensive data quality report with centralized logging."""
    log_info(f"Starting data quality assessment for: {file_name}")
    
    try:
        # Data loading phase
        log_debug(f"Loading dataset: {file_name}")
        df = load_data(file_name)
        if isinstance(df, str):
            log_error(f"Data loading failed: {df}")
            return {"output": df, "should_stop": True}
        
        log_info(f"Data loaded successfully. Dimensions: {df.shape}")
        log_debug(f"Columns detected: {list(df.columns)}")
        log_debug(f"Data sample:\n{df.head(2).to_string()}")  # Detailed debug

        # Data conversion helper
        def convert_to_serializable(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return int(obj) if isinstance(obj, np.integer) else float(obj)
            return obj

        # Report generation
        log_debug("Calculating quality metrics")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        log_debug(f"Numeric columns identified: {list(numeric_cols)}")

        # Core metrics calculation
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

        # Quality issues detection
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

        log_info(f"Quality report generated. Score: {report['quality_score']:.1f}")
        log_debug(f"Quality issues found: {len(quality_issues)}")

        return {
            "output": f"Data Quality Report for {file_name}:\n{json.dumps(report, indent=2)}",
            "report": report,
            "should_stop": True
        }

    except Exception as e:
        log_critical(f"Critical error during quality assessment: {str(e)}")
        return {"output": f"Quality report error: {str(e)}", "should_stop": True}
