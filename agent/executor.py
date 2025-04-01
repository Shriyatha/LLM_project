from typing import Dict, Any, List, Union
from langchain.agents import AgentExecutor


class CustomAgentExecutor(AgentExecutor):
    def _execute_tool(self, tool_name: str, tool_input: Union[str, Dict], run_manager=None) -> Any:
        """Execute the specified tool with the given input."""
        for tool in self.tools:
            if tool.name == tool_name:
                return tool.run(tool_input, verbose=self.verbose)
        raise ValueError(f"Tool '{tool_name}' not found")

    def _call(self, inputs: Dict[str, Any], run_manager=None) -> Dict[str, Any]:
        """Run the agent loop until we get a final answer or reach max iterations."""
        intermediate_steps: List[tuple] = []
        
        for step in range(self.max_iterations):
            output = self.agent.plan(
                intermediate_steps,
                callbacks=run_manager.get_child() if run_manager else None,
                **inputs,
            )

            if isinstance(output, dict) and output.get("should_stop", False):
                return {
                    "output": output.get("output", ""),
                    "intermediate_steps": intermediate_steps,
                }

            if getattr(output, "tool", None) == "FinalAnswer":
                return {
                    "output": getattr(output, "tool_input", ""),
                    "intermediate_steps": intermediate_steps,
                }

            intermediate_steps.append((output, None))

            if getattr(output, "tool", None):
                observation = self._execute_tool(output.tool, output.tool_input, run_manager)
                intermediate_steps[-1] = (output, observation)

                if isinstance(observation, dict) and observation.get("should_stop", False):
                    return {
                        "output": observation.get("output", ""),
                        "intermediate_steps": intermediate_steps,
                    }
            else:
                observation = output.tool_input

            if run_manager:
                run_manager.on_agent_action(output, verbose=self.verbose)

        return {
            "output": "Maximum iterations reached without final answer.",
            "intermediate_steps": intermediate_steps,
        }


def execute_agent_query(agent, query: str) -> Dict[str, Any]:
    """Executes a query on the agent and handles the response."""
    try:
        response = agent.invoke({"input": query})

        if isinstance(response, dict) and "intermediate_steps" in response:
            for step in response["intermediate_steps"]:
                if isinstance(step, tuple) and len(step) > 1:
                    observation = step[1]
                    if isinstance(observation, dict) and observation.get("should_stop", False):
                        return {"output": observation.get("output", str(observation))}

        return {"output": response.get("output", str(response))}
    except Exception as e:
        return {"output": f"Error processing query: {str(e)}"}
