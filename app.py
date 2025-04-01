from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Dict, Any, List
import logging
from agent import initialize_custom_agent, execute_agent_query
from pydantic import BaseModel
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Data Analysis Agent API",
    description="API for data analysis queries",
    version="0.1.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agent at startup
@app.on_event("startup")
async def startup_event():
    try:
        app.state.agent = initialize_custom_agent()
        logger.info("Agent initialized successfully")
    except Exception as e:
        logger.error(f"Agent initialization failed: {str(e)}")
        raise

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def process_query(request: QueryRequest):
    try:
        logger.info(f"Processing query: {request.query}")
        result = execute_agent_query(app.state.agent, request.query)
        
        return {
            "output": result.get("output", "No output returned"),
            "steps": result.get("intermediate_steps", []),
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Query processing failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": str(e)}
        )

@app.get("/test-queries")
async def run_test_queries():
    test_queries = [
        "Check for missing values in 'test.csv'",
        "Identify outliers in the salary column of 'test.csv'"
    ]
    
    results = []
    for query in test_queries:
        try:
            result = execute_agent_query(app.state.agent, query)
            results.append({
                "query": query,
                "output": result.get("output"),
                "success": True
            })
        except Exception as e:
            results.append({
                "query": query,
                "error": str(e),
                "success": False
            })
    
    return {"results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)