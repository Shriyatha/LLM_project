from agent import execute_agent_query, initialize_custom_agent


def run_test_queries():
    """Test function with comprehensive query coverage."""
    agent = initialize_custom_agent()

    test_queries = [
        # Basic file operations
        "What files are available for analysis?",
        "Show me the first 5 rows of 'test.csv'",
        "List all columns in 'test.csv' with their data types",

        # # # Data quality checks
        "Show me a data quality report for 'test.csv'",
        "Check for missing values in 'test.csv'",
        "Identify outliers in the salary column of 'test.csv'",

        # # Descriptive statistics
        "What is the average and maximum salary in 'test.csv'?",
        "Show summary statistics for numeric columns in 'test.csv'",
        "Calculate the correlation matrix for 'test.csv'",

        # Data filtering and manipulation
        "Filter 'test.csv' where age > 30",
        "Sort 'test.csv' by salary in descending order",
        "Create a new column 'bonus' as 10% of salary in 'test.csv'",

        # Aggregation operations
        "Count employees by department in 'employees.csv'",

        # Sorting query
        "sort the users by age in test.csv?"
        "sort the users by date in stock_data.csv with price more than 153?",

        # Visualization queries
        # "Create a histogram of salaries from 'test.csv'",
        # "Plot a bar chart of average salary by department in 'employees.csv'",
        # "Generate a scatter plot of age vs. salary in 'test.csv'"

        # Advanced analysis
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
