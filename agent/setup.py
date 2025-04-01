import json
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
from agent.executor import CustomAgentExecutor
# Import functions from tools.py
from tools import (
    list_files,
    visualize_data, data_quality_report, time_series_analysis,
    correlation_analysis
)
from tools.data_analysis import get_columns, transform_data, show_data_sample, check_missing_values, detect_outliers, aggregate_data,Sort_data, summary_statistics, filter_data
from tools.visualization import visualize_data, aggregate_and_visualize
def log(message):
    """Force debug messages to console."""
    print(f"DEBUG: {message}", flush=True)

# ✅ Wrap final_answer as a function
def final_answer(output: str) -> str:
    """Stops execution and returns the final answer."""
    log(f"FinalAnswer invoked with output: {output}")
    return output

final_answer_tool = StructuredTool.from_function(
    func=final_answer,
    name="FinalAnswer",
    description="Stops execution and returns the final answer."
)

def initialize_custom_agent():
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
            description=(
                "Analyze missing values in a data file. "
                "Input should be a JSON object with: "
                "'file_name' (string, required)"
            ),
            return_direct=True
        ),
        
        StructuredTool.from_function(
            func=detect_outliers,
            name="DetectOutliers",
            description=(
                "Identify outliers in data columns using statistical methods. "
                "Input should be a JSON object with: "
                "'file_name' (string, required), "
                "'column' (string, required), "
                "'method' (string, optional: 'iqr' or 'zscore'), "
                "'threshold' (number, required)."
            ),
            return_direct=True
        ),

        StructuredTool.from_function(
            func=filter_data,
            name="FilterData",
            description=(
                "Filter data based on conditions. "
                "Input should be a JSON object with: "
                "'file_name' (string, required), "
                "'conditions' (array of arrays, required: each containing "
                "[column (string), operator (string: '==', '!=', '>', '<', '>=', '<='), value]"
            ),
            return_direct=True
        ),

        StructuredTool.from_function(
            func=aggregate_data,
            name="AggregateData",
            description=(
                "Calculate statistics on data columns using aggregation methods. "
                "Input should be a JSON object with: "
                "'file_name' (string, required), "
                "'column' (string, required), "
                "'agg_funcs' (array of strings, required: supported functions are "
                "'mean', 'max', 'min', 'sum', 'count')"
            ),
            return_direct=True
        ),
        StructuredTool.from_function(
            func=Sort_data,
            name="SortData",
            description=(
            "Sort data by a specified column in ascending or descending order with optional filtering. "
            "Input should be a JSON object with: "
            "'file_name' (string, required), "
            "'column' (string, required), "
            "'order' (string, required: 'asc' for ascending, 'desc' for descending), "
            "'filter_column' (string, optional), "
            "'filter_operator' (str ,optional)"
            "'filter_value' (string or integer, optional)"
            ),
            return_direct=True
        ),
        StructuredTool.from_function(
            func=summary_statistics,
            name="SummaryStatistics",
            description=(
                "Calculate summary statistics for all numeric columns in a file. "
                "Input should be a JSON object with: "
                "'file_name' (string, required)"
            ),
            return_direct=True
        ),
        StructuredTool.from_function(
            func=aggregate_and_visualize,
            name="AggregateAndVisualize",
            description="Aggregates data by specified column and creates visualization. "
                    "Required parameters: file_name, value_col (column to average), "
                    "group_by (column to group by). Optional: plot_type (default: 'bar').",
            return_direct=True
        ),

            StructuredTool.from_function(
            func=visualize_data,
            name="VisualizeData",
            description="Create visualizations. Input should be a dictionary with 'file_name' (str), 'plot_type' (str), 'y_col' (str), and optionally 'x_col' (str)."
        ),
        StructuredTool.from_function(
            func=data_quality_report,
            name="DataQualityReport",
            description="Generate a data quality report. Input should be a dictionary with 'file_name' (str).",
            return_direct=True  
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
        StructuredTool.from_function(
            func=transform_data,
            name="TransformData",
            description="Create new columns or modify data. Input: dict with 'file_name', 'operations' (list of transforms like 'salary*0.1 as bonus')"
        ),
        StructuredTool.from_function(
            func=show_data_sample,
            name="ShowDataSample",
            description="Show sample rows from a file. Input: dict with 'file_name' and optionally 'num_rows', 'columns'"
        ),

        StructuredTool.from_function(
            func=get_columns,
            name="GetColumns",
            description="List columns and data types for a file. Input: dict with 'file_name'",
            return_direct=True
        ),
        StructuredTool.from_function(
            func=final_answer,
            name="FinalAnswer",
            description="Stops execution and returns the final answer. Input should be the final answer string.",
            return_direct=True
        )
    ]

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=7,
        early_stopping_method="generate",
        return_intermediate_steps=True
    )
    
    return CustomAgentExecutor.from_agent_and_tools(
        agent=agent.agent,
        tools=tools,
        verbose=True,
        max_iterations=7,
        handle_parsing_errors=True
    )
    