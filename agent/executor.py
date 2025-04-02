from typing import Dict, Any, List, Union
from langchain.agents import AgentExecutor
from loguru import logger
from logging_client import log_info, log_debug, log_warning, log_error


class CustomAgentExecutor(AgentExecutor):
    def _execute_tool(self, tool_name: str, tool_input: Union[str, Dict], run_manager=None) -> Any:
        """Execute the specified tool with the given input."""
        log_info(f"Executing tool: {tool_name} with input: {tool_input}")
        
        for tool in self.tools:
            if tool.name == tool_name:
                try:
                    result = tool.run(tool_input, verbose=self.verbose)
                    log_info(f"Execution result for {tool_name}: {result}")
                    return result
                except Exception as e:
                    log_error(f"Error executing tool {tool_name}: {str(e)}")
                    raise
        
        log_warning(f"Tool '{tool_name}' not found")
        raise ValueError(f"Tool '{tool_name}' not found")

    def _call(self, inputs: Dict[str, Any], run_manager=None) -> Dict[str, Any]:
        """Run the agent loop until we get a final answer or reach max iterations."""
        log_info(f"Agent execution started with inputs: {inputs}")
        
        intermediate_steps: List[tuple] = []
        
        for step in range(self.max_iterations):
            try:
                output = self.agent.plan(
                    intermediate_steps,
                    callbacks=run_manager.get_child() if run_manager else None,
                    **inputs,
                )
                log_debug(f"Step {step}: Plan output - {output}")
            except Exception as e:
                log_error(f"Error during planning at step {step}: {str(e)}")
                return {"output": f"Error at step {step}: {str(e)}", "intermediate_steps": intermediate_steps}
            
            if isinstance(output, dict) and output.get("should_stop", False):
                log_info("Agent stopping due to should_stop flag.")
                return {"output": output.get("output", ""), "intermediate_steps": intermediate_steps}
            
            if getattr(output, "tool", None) == "FinalAnswer":
                log_info("FinalAnswer tool selected. Returning output.")
                return {"output": getattr(output, "tool_input", ""), "intermediate_steps": intermediate_steps}
            
            intermediate_steps.append((output, None))
            
            if getattr(output, "tool", None):
                try:
                    observation = self._execute_tool(output.tool, output.tool_input, run_manager)
                    intermediate_steps[-1] = (output, observation)
                    log_debug(f"Step {step}: Tool {output.tool} executed with observation {observation}")
                except Exception as e:
                    log_error(f"Error executing tool {output.tool}: {str(e)}")
                    return {"output": f"Error executing tool {output.tool}: {str(e)}", "intermediate_steps": intermediate_steps}
                
                if isinstance(observation, dict) and observation.get("should_stop", False):
                    log_info("Stopping execution based on observation.")
                    return {"output": observation.get("output", ""), "intermediate_steps": intermediate_steps}
            else:
                observation = output.tool_input
                
            if run_manager:
                run_manager.on_agent_action(output, verbose=self.verbose)
                
        log_warning("Maximum iterations reached without final answer.")
        return {"output": "Maximum iterations reached without final answer.", "intermediate_steps": intermediate_steps}


def execute_agent_query(agent, query):
    try:
        log_info(f"Executing agent query: {query}")
        response = agent.invoke({"input": query})
        log_info(f"Agent response: {response}")
        
        if 'intermediate_steps' in response:
            for step in response['intermediate_steps']:
                if isinstance(step, tuple) and len(step) > 1:
                    observation = step[1]
                    if isinstance(observation, dict) and observation.get('should_stop', False):
                        log_info("Agent execution stopping due to should_stop flag in intermediate steps.")
                        return {"output": observation.get('output', str(observation))}
        
        return {"output": response.get('output', str(response))}
    except Exception as e:
        log_error(f"Error processing query: {str(e)}")
        return {"output": f"Error processing query: {str(e)}"}