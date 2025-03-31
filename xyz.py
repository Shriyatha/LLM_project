import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
from typing import Dict, List, Union, Optional, Any
from langchain_community.llms import Ollama
from langchain.agents import Tool, AgentExecutor, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.chains import LLMMathChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.tools import StructuredTool
from langchain.schema import SystemMessage
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.agents import OpenAIFunctionsAgent

# Improved data folder handling with path validation
DATA_FOLDER = "./data"
os.makedirs(DATA_FOLDER, exist_ok=True)

def validate_file_path(filename: str) -> str:
    """Validates and sanitizes file paths to prevent directory traversal."""
    filename = os.path.basename(filename.strip())
    filepath = os.path.join(DATA_FOLDER, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File '{filename}' not found in {DATA_FOLDER}")
    return filepath

def list_files() -> Dict:
    """Lists available CSV/Excel files in the data folder."""
    try:
        files = [f for f in os.listdir(DATA_FOLDER) 
                if f.endswith(('.csv', '.xlsx', '.xls'))]
        return {
            "output": f"Available data files: {', '.join(files)}" if files else "No data files found.",
            "files": files,
            "should_stop": True
        }
    except Exception as e:
        return {
            "output": f"Error listing files: {str(e)}",
            "should_stop": True
        }

def load_data(filename: str) -> Union[pd.DataFrame, str]:
    """Improved data loading with better error handling."""
    try:
        filepath = validate_file_path(filename)
        
        if filename.endswith('.csv'):
            return pd.read_csv(filepath)
        elif filename.endswith(('.xlsx', '.xls')):
            return pd.read_excel(filepath)
        return f"Unsupported file format for {filename}"
    except Exception as e:
        return f"Error loading file: {str(e)}"

def filter_data(file_name: str, conditions: List[List[str]]) -> Dict:
    """Enhanced filter with better type handling."""
    try:
        df = load_data(file_name)
        if isinstance(df, str):
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
                if operator in ['>', '<', '>=', '<=']:
                    value = float(value)
                    df[column] = pd.to_numeric(df[column], errors='coerce')
                
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

        return {
            "output": f"Found {len(df)} matching records. Sample: {df.head().to_dict('records')}",
            "count": len(df),
            "sample": df.head().to_dict('records'),
            "should_stop": True
        }
    except Exception as e:
        return {"output": f"Filter error: {str(e)}", "should_stop": True}

def aggregate_data(file_name: str, column: str, agg_funcs: List[str]) -> Dict:
    """Enhanced aggregation with multiple metrics support."""
    try:
        df = load_data(file_name)
        if isinstance(df, str):
            return {"error": df}

        # Standardize column names
        df.columns = df.columns.str.strip().str.lower()
        column = column.strip().lower()

        if column not in df.columns:
            return {"error": f"Column '{column}' not found. Available: {list(df.columns)}"}

        # Convert to numeric and clean
        df[column] = pd.to_numeric(df[column], errors='coerce')
        df = df.dropna(subset=[column])

        results = {}
        supported_funcs = ['sum', 'mean', 'count', 'max', 'min', 'median', 'std', 'range', 'variance']
        
        for func in agg_funcs:
            func = func.lower()
            if func not in supported_funcs:
                return {"error": f"Unsupported function: {func}. Use: {', '.join(supported_funcs)}"}
            
            if func == 'range':
                results['range'] = df[column].max() - df[column].min()
            elif func == 'variance':
                results['variance'] = df[column].var()
            else:
                results[func] = float(getattr(df[column], func)())  # Convert numpy types to native Python

        return {
            "status": "success",
            "results": results,
            "output": f"Aggregation results for '{column}': {results}",
            "should_stop": True
        }
    except Exception as e:
        return {"error": f"Aggregation error: {str(e)}"}

def visualize_data(file_name: str, plot_type: str, y_col: str, x_col: Optional[str] = None) -> Dict:
    """Creates visualizations from data with improved error handling."""
    try:
        df = load_data(file_name)
        if isinstance(df, str):
            return {"error": df}

        # Standardize column names
        df.columns = df.columns.str.strip().str.lower()
        plot_type = plot_type.lower()
        y_col = y_col.strip().lower()
        if x_col:
            x_col = x_col.strip().lower()

        # Validate columns
        required_cols = [y_col]
        if x_col:
            required_cols.append(x_col)
            
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return {"error": f"Columns not found: {missing_cols}. Available: {list(df.columns)}"}

        plt.figure(figsize=(10, 6))
        plot_title = f"{plot_type.capitalize()} plot"

        if plot_type == 'bar':
            if not x_col:
                return {"error": "x_col is required for bar plot"}
            df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
            df.groupby(x_col)[y_col].mean().plot(kind='bar')
            plot_title += f" of {y_col} by {x_col}"
            plt.ylabel(y_col)
        elif plot_type == 'hist':
            df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
            df[y_col].plot(kind='hist', bins=20)
            plt.xlabel(y_col)
            plot_title += f" of {y_col}"
        elif plot_type == 'scatter':
            if not x_col:
                return {"error": "x_col is required for scatter plot"}
            df.plot(x=x_col, y=y_col, kind='scatter')
            plot_title += f" of {y_col} vs {x_col}"
        elif plot_type == 'line':
            if not x_col:
                return {"error": "x_col is required for line plot"}
            df.plot(x=x_col, y=y_col, kind='line')
            plot_title += f" of {y_col} over {x_col}"
        elif plot_type == 'box':
            df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
            df[y_col].plot(kind='box')
            plot_title += f" of {y_col}"
        else:
            return {"error": f"Unsupported plot type: {plot_type}. Use: bar, hist, scatter, line, box"}

        plt.title(plot_title)
        plot_filename = f"plot_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
        plt.savefig(plot_filename)
        plt.close()

        return {
            "status": "success",
            "output": f"Visualization saved as {plot_filename}",
            "plot_file": plot_filename,
            "should_stop": True
        }
    except Exception as e:
        return {"error": f"Visualization error: {str(e)}"}

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
        report['quality_score'] = 100 - (len(quality_issues) / len(df.columns) * 50)  # Simple quality score

        return {
            "output": f"Data Quality Report for {file_name}:\n{json.dumps(report, indent=2)}",
            "report": report,
            "should_stop": True
        }
    except Exception as e:
        return {"output": f"Quality report error: {str(e)}", "should_stop": True}

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

        # Create heatmap visualization
        plt.figure(figsize=(10, 8))
        plt.matshow(corr_matrix, fignum=1)
        plt.xticks(range(len(cols)), cols, rotation=45)
        plt.yticks(range(len(cols)), cols)
        plt.colorbar()
        plt.title("Correlation Matrix")
        
        plot_filename = f"correlation_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
        plt.savefig(plot_filename)
        plt.close()

        return {
            "status": "success",
            "output": f"Correlation analysis saved as {plot_filename}. Correlation matrix:\n{corr_matrix.to_string()}",
            "plot_file": plot_filename,
            "correlation_matrix": corr_matrix.to_dict(),
            "should_stop": True
        }
    except Exception as e:
        return {"error": f"Correlation analysis error: {str(e)}"}

# Initialize LLM with streaming callback
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
llm = Ollama(
    model="llama3:8b",
    temperature=0,
    callback_manager=callback_manager
)

# Math tool
math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)

# Improved tool definitions using StructuredTool for better handling
tools = [
    StructuredTool.from_function(
        func=list_files,
        name="ListFiles",
        description="List all available data files in the data directory. No input required. Returns a list of filenames."
    ),
    StructuredTool.from_function(
        func=filter_data,
        name="FilterData",
        description="Filter data based on conditions. Input should be a dictionary with 'file_name' (str) and 'conditions' (list of lists, each containing [column, operator, value])."
    ),
    StructuredTool.from_function(
        func=aggregate_data,
        name="AggregateData",
        description="Calculate statistics on a column. Input should be a dictionary with 'file_name' (str), 'column' (str), and 'agg_funcs' (list of str)."
    ),
    StructuredTool.from_function(
        func=visualize_data,
        name="VisualizeData",
        description="Create visualizations. Input should be a dictionary with 'file_name' (str), 'plot_type' (str), 'y_col' (str), and optionally 'x_col' (str)."
    ),
    StructuredTool.from_function(
        func=data_quality_report,
        name="DataQualityReport",
        description="Generate a data quality report. Input should be a dictionary with 'file_name' (str)."
    ),
    StructuredTool.from_function(
        func=time_series_analysis,
        name="TimeSeriesAnalysis",
        description="Analyze time series data. Input should be a dictionary with 'file_name' (str), 'date_col' (str), 'value_col' (str), and optionally 'freq' (str: D, W, M, Q, Y)."
    ),
    StructuredTool.from_function(
        func=correlation_analysis,
        name="CorrelationAnalysis",
        description="Calculate correlations between columns. Input should be a dictionary with 'file_name' (str) and 'cols' (list of str)."
    ),
    Tool(
        name="Calculator",
        func=math_chain.run,
        description="Useful for math calculations. Input should be a math expression as a string."
    ),
    Tool(
        name="FinalAnswer",
        func=lambda x: {"output": str(x), "should_stop": True},
        description="Use this to provide the final answer to the user."
    )
]

# Custom agent executor to handle should_stop
class CustomAgentExecutor(AgentExecutor):
    def _call(self, inputs: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        result = super()._call(inputs, **kwargs)
        
        # Check intermediate steps for should_stop
        if 'intermediate_steps' in result:
            for step in result['intermediate_steps']:
                if isinstance(step, tuple) and len(step) > 1:
                    observation = step[1]
                    if isinstance(observation, dict) and observation.get('should_stop', False):
                        return {"output": observation.get('output', str(observation)), "should_stop": True}
        
        return result

# Initialize agent with proper configuration
# Initialize the agent with the correct configuration
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5,
    early_stopping_method="generate",
    return_intermediate_steps=True
)

# Custom function to handle agent execution with should_stop checking
def execute_agent_query(agent, query):
    try:
        response = agent({"input": query})
        
        # Check intermediate steps for should_stop
        if 'intermediate_steps' in response:
            for step in response['intermediate_steps']:
                if isinstance(step, tuple) and len(step) > 1:
                    observation = step[1]
                    if isinstance(observation, dict) and observation.get('should_stop', False):
                        return {"output": observation.get('output', str(observation))}
        
        return {"output": response.get('output', str(response))}
    except Exception as e:
        return {"output": f"Error processing query: {str(e)}"}

# Update the test function to use our custom executor
def run_test_queries():
    """Test function with improved error handling."""
    test_queries = [
        "What files are available for analysis?",
        "Show me a data quality report for 'test.csv'",
        "What is the average and maximum salary in 'test.csv'?",
        #"Filter 'test.csv' where age > 30",
        #"Create a histogram of salaries from 'test.csv'",
        #"What is 25% of the average salary from 'test.csv'?",
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*50}\nQuery {i}: {query}\n{'='*50}")
        response = execute_agent_query(agent, query)
        print("Agent response:", response['output'])

run_test_queries()