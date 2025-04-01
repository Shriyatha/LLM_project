import json
from langchain_community.llms import Ollama
from langchain.agents import Tool, AgentExecutor, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.chains import LLMMathChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.tools import StructuredTool
from langchain.schema import SystemMessage
from typing import List, Optional
from agent.executor import CustomAgentExecutor

from pydantic import BaseModel, Field

class DataQualityInput(BaseModel):
    file_name: str = Field(..., description="Name/path of the CSV file to analyze")

# Import functions from tools.py
from tools import (
    list_files,
    visualize_data, time_series_analysis,
    correlation_analysis
)
from tools.data_analysis import (
    get_columns, transform_data, show_data_sample, check_missing_values, 
    detect_outliers, aggregate_data, summary_statistics, filter_data, sort_data
)
from tools.visualization import visualize_data as viz_data, aggregate_and_visualize
from tools.quality_report import data_quality_report
def log(message: str) -> None:
    """Force debug messages to console."""
    print(f"DEBUG: {message}", flush=True)

def final_answer(output: str) -> str:
    """Stops execution and returns the final answer."""
    log(f"FinalAnswer invoked with output: {output}")
    return output

def initialize_custom_agent() -> AgentExecutor:
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llm = Ollama(
        model="llama3:8b",
        temperature=0,
        callback_manager=callback_manager
    )

    # Math tool
    math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)

    tools = [
        StructuredTool.from_function(
            func=list_files,
            name="ListFiles",
            description="List all available data files in the data directory. No input required. Returns a list of filenames."
        ),
        StructuredTool.from_function(
            func=check_missing_values,
            name="CheckMissingValues",
            description="Analyze missing values in a data file. Input should be a string with the filename.",
            args_schema=None  # Remove schema to accept simple string input
        ),
        StructuredTool.from_function(
            func=detect_outliers,
            name="DetectOutliers",
            description=(
                "Identify outliers in data columns using statistical methods. "
                "Input should be a JSON string with: "
                "'file_name' (string, required), "
                "'column' (string, required), "
                "'method' (string, optional: 'iqr' or 'zscore', default='iqr'), "
                "'threshold' (number, optional: default 1.5 for IQR, 3 for Z-score)"
            ),
            args_schema=None
        ),
        StructuredTool.from_function(
            func=filter_data,
            name="FilterData",
            description=(
                "Filter data based on conditions. "
                "Input should be a JSON string with: "
                "'file_name' (string, required), "
                "'conditions' (list of lists, required: each containing "
                "[column (string), operator (string), value])"
            ),
            args_schema=None
        ),
        StructuredTool.from_function(
            func=aggregate_data,
            name="AggregateData",
            description=(
                "Calculate statistics on data columns using aggregation methods. "
                "Input should be a JSON string with: "
                "'file_name' (string, required), "
                "'column' (string, required), "
                "'agg_funcs' (list of strings, required: supported functions are "
                "'mean', 'max', 'min', 'sum', 'count')"
            ),
            args_schema=None
        ),
        StructuredTool.from_function(
            func=summary_statistics,
            name="SummaryStatistics",
            description=(
                "Calculate summary statistics for all numeric columns in a file. "
                "Input should be a string with the filename."
            ),
            args_schema=None
        ),
        StructuredTool.from_function(
            func=aggregate_and_visualize,
            name="AggregateAndVisualize",
            description=(
                "Aggregates data by specified column and creates visualization. "
                "Input should be a JSON string with: "
                "'file_name' (string, required), "
                "'value_col' (string, required), "
                "'group_by' (string, required), "
                "'plot_type' (string, optional: default 'bar')"
            ),
            args_schema=None
        ),
        StructuredTool.from_function(
            func=viz_data,
            name="VisualizeData",
            description=(
                "Create visualizations. Input should be a JSON string with: "
                "'file_name' (string, required), "
                "'plot_type' (string, required), "
                "'x_col' (string, optional), "
                "'y_col' (string, required if x_col provided)"
            ),
            args_schema=None
        ),
        StructuredTool.from_function(
            func=data_quality_report,
            name="DataQualityReportTool",
            description="Generate a comprehensive data quality report for a given dataset.Input should be a JSON string with: "
                "'file_name' (string, required), "
        ),

        StructuredTool.from_function(
            func=time_series_analysis,
            name="TimeSeriesAnalysis",
            description=(
                "Analyze time series data. Input should be a JSON string with: "
                "'file_name' (string, required), "
                "'date_col' (string, required), "
                "'value_col' (string, required), "
                "'freq' (string, optional: D, W, M, Q, Y)"
            ),
            args_schema=None
        ),
        StructuredTool.from_function(
            func=correlation_analysis,
            name="CorrelationAnalysis",
            description=(
                "Calculate correlations between columns. Input should be a JSON string with: "
                "'file_name' (string, required), "
                "'cols' (list of strings, required)"
            ),
            args_schema=None
        ),
        Tool(
            name="Calculator",
            func=math_chain.run,
            description="Useful for math calculations. Input should be a math expression as a string."
        ),
        StructuredTool.from_function(
            func=transform_data,
            name="TransformData",
            description="Create new columns or modify data. Input: dict with 'file_name', 'operations' (list of transforms like 'salary*0.1 as bonus')"
        ),

        StructuredTool.from_function(
            func=show_data_sample,
            name="ShowDataSample",
            description=(
                "Show sample rows from a file. Input should be a JSON string with: "
                "'file_name' (string, required), "
                "'num_rows' (integer, optional), "
                "'columns' (list of strings, optional)"
            ),
            args_schema=None
        ),
        StructuredTool.from_function(
            func=get_columns,
            name="GetColumns",
            description="List columns and data types for a file. Input should be a string with the filename.",
            args_schema=None
        ),
        StructuredTool.from_function(
            func=final_answer,
            name="FinalAnswer",
            description="Stops execution and returns the final answer. Input should be the final answer string.",
            return_direct=True
        )
    ]

    # Initialize the agent with proper prompt template
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=7,
        early_stopping_method="generate",
        agent_kwargs={
            "input_variables": ["input", "agent_scratchpad"],
            "memory_prompts": [],
            "system_message": SystemMessage(
                content=(
                    "You are a helpful AI assistant that performs data analysis tasks. "
                    "When asked to perform operations on data files, you should:"
                    "\n1. Use simple string inputs for filenames (not complex objects)"
                    "\n2. For DataQualityReport, just provide the filename as a input dictionary"
                    "\n3. For tools requiring multiple parameters, use JSON strings"
                    "\n4. Always verify the file exists before operating on it"
                    "\n5. First check available columns using GetColumns before specifying columns for analysis."
                    "\n6. Operations should be in format: 'expression as new_column_name' (e.g., salary*0.1 as bonus)"
                    "\n7. Column must be numeric for aggregation functions. Use GetColumns first to check data types."
                )
            )
        }
    )
    

    
    return CustomAgentExecutor.from_agent_and_tools(
        agent=agent.agent,
        tools=tools,
        verbose=True,
        max_iterations=7,
        handle_parsing_errors=True
    )