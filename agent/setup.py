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
    list_files, aggregate_data, filter_data,
    visualize_data, data_quality_report, time_series_analysis,
    correlation_analysis
)


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
            description="Generate a data quality report. Input should be a dictionary with 'file_name' (str).",
            return_direct=True  # This will make it return directly
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
        max_iterations=5,
        early_stopping_method="generate",
        return_intermediate_steps=True
    )
    
    # Wrap the agent with our custom executor
    return CustomAgentExecutor.from_agent_and_tools(
        agent=agent.agent,
        tools=tools,
        verbose=True,
        max_iterations=5,
        handle_parsing_errors=True,
        return_intermediate_steps=True
    )