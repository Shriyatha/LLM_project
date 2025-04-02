from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Optional
from logging_client import log_info, log_debug, log_error, log_critical, setup_logging_client, log_trace
from agent import initialize_custom_agent, execute_agent_query
from pydantic import BaseModel
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
import base64
from io import BytesIO
import uvicorn
from pathlib import Path

# Load environment variables
load_dotenv()

# Initialize logging client
config_path = os.getenv('LOGGING_CONFIG', 'logging_config.yaml')
setup_logging_client(config_path)

app = FastAPI(
    title="Data Analysis Agent API",
    description="API for data analysis queries",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None  # For session tracking

class TestQueryResponse(BaseModel):
    query: str
    output: Optional[str]
    plot: Optional[str]
    success: bool
    error: Optional[str]

# Initialize agent at startup with logging
@app.on_event("startup")
async def startup_event():
    """Initialize the agent and setup logging context."""
    try:
        log_info("Starting API initialization")
            
        # Initialize agent
        log_debug("Initializing custom agent")
        app.state.agent = initialize_custom_agent()
            
        # Verify agent initialization
        if not app.state.agent:
            raise RuntimeError("Agent initialization returned None")
            
        log_info("Agent initialized successfully")
            
    except Exception as e:
        log_critical(f"API startup failed: {str(e)}")
        raise RuntimeError(f"API startup failed: {str(e)}")

@app.post("/query", response_model=Dict[str, Any])
async def process_query(request: QueryRequest):
    """Process a data analysis query through the agent."""
    try:
        log_info(f"Processing new query: {request.query}, session_id: {request.session_id}")
            
        # Execute query
        log_debug("Executing agent query")
        result = execute_agent_query(app.state.agent, request.query)
            
        # Handle plot results
        if 'plot' in result:
            log_debug("Processing plot result")
            buf = BytesIO()
            result['plot'].savefig(buf, format='png', bbox_inches='tight')
            plt.close(result['plot'])
            plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            result['plot'] = plot_base64
            
        log_info("Query processed successfully")
        return {
            "output": result.get("output", "No output returned"),
            "steps": result.get("intermediate_steps", []),
            "status": "success"
        }
            
    except Exception as e:
        log_error(f"Query processing failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Query processing failed",
                "message": str(e)
            }
        )

@app.get("/test-queries", response_model=Dict[str, List[TestQueryResponse]])
async def run_test_queries():
    """Execute a suite of test queries for validation."""
    test_queries = [
        "Check for missing values in 'test.csv'",
        "Identify outliers in the salary column of 'test.csv'",
        "Show summary statistics for 'test.csv'"
    ]
    
    results = []
    for query in test_queries:
        try:
            log_debug(f"Executing test query: {query}")
                
            result = execute_agent_query(app.state.agent, query)
                
            response = {
                "query": query,
                "output": result.get("output"),
                "success": True
            }
                
            if 'plot' in result:
                buf = BytesIO()
                result['plot'].savefig(buf, format='png', bbox_inches='tight')
                plt.close(result['plot'])
                response['plot'] = base64.b64encode(buf.getvalue()).decode('utf-8')
                
            log_debug("Test query completed successfully")
            results.append(response)
                
        except Exception as e:
            log_error(f"Test query failed: {str(e)}")
            results.append({
                "query": query,
                "error": str(e),
                "success": False
            })
    
    log_info(f"Completed test queries with {len([r for r in results if r['success']])} successes")
    return {"results": results}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    log_trace("Health check requested")
    return {"status": "healthy", "version": app.version}

if __name__ == "__main__":
    # Configure logging for production
    log_info("Starting API server")
    try:
        uvicorn.run(
            app,
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", 8000)),
            log_config=None,  # Disable default uvicorn logging
            access_log=False  # We handle logging ourselves
        )
    except Exception as e:
        log_critical(f"API server failed to start: {str(e)}")
        raise
