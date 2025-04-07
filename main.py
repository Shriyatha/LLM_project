from agent import execute_agent_query, initialize_custom_agent


def run_test_queries():
    """Test function with comprehensive query coverage."""
    agent = initialize_custom_agent()

    test_queries = [
        "Show the average salary for employees aged over 35 in employees.csv.",
        "List all available data files.",
        "Show summary statistics for employees.csv.",
        "Identify missing values in sales_data.csv.",
        "Detect outliers in the salary column of employees.csv.",
        "Filter sales_data.csv for orders where region is West.",
        "Aggregate amount of sales for each region in sales_data.csv.",
        "Sort employees.csv by salary in descending order.",
        "Show a sample of 3 rows from test.csv.",
        "List columns and data types for sales_data.csv.",
        "Generate a data quality report for test.csv.",
        "Calculate correlation between age and salary in employees.csv.",
        "Create a new column bonus = salary * 0.1 in employees.csv.",
        "Create a bar chart showing the count of employees per department in employees.csv.",
        "Visualize the total sales amount per region using sales_data.csv.",
        "Compute the average salary in employees.csv.",
        "Count the total number of orders in sales_data.csv.",
        "Compute the highest salary in employees.csv.",
        "Convert salary column in test.csv to thousands (e.g., 75000 → 75K).",
        "Find the total sales revenue for the 'West' region in sales_data.csv.",

        # Visualization queries
        # "Create a histogram of salaries from 'test.csv'",
        # "Plot a bar chart of average salary by department in 'employees.csv'",
        # "Generate a scatter plot of age vs. salary in 'test.csv'",

        # # Advanced analysis
        # "Run regression analysis on price vs. features",
        # "Identify top 5 most frequent values in each column"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*50}\nQuery {i}: {query}\n{'='*50}")
        try:
            result = execute_agent_query(agent, query)
            output = result.get("output", "No output returned")
            print(f"Final Output: {output}")

            # Print intermediate steps if available
            if result.get("intermediate_steps"):
                print("\nExecution Steps:")
                for step in result["intermediate_steps"]:
                    print(f"- {step[0] if isinstance(step, tuple) else step}")
        except Exception as e:
            print(f"Error executing query: {e!s}")


if __name__ == "__main__":
    print("Starting Data Analysis Agent Test Suite\n")
    run_test_queries()
    print("\nTest suite completed")