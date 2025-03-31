from agent import initialize_custom_agent, execute_agent_query

def run_test_queries():
    """Test function with improved error handling."""
    agent = initialize_custom_agent()
    
    test_queries = [
        #"What files are available for analysis?",
        #"Show me a data quality report for 'test.csv'",
        #"What is the average and maximum salary in 'test.csv'?",
        #"Filter 'test.csv' where age > 30",
        #"Create a histogram of salaries from 'test.csv'",
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*50}\nQuery {i}: {query}\n{'='*50}")
        result = execute_agent_query(agent, query)
        print("Final Output:", result['output'])


if __name__ == "__main__":
    run_test_queries()