import os
import time
import logging
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.api import api_router
from app.core.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("vectorless_rag")

app = FastAPI(
    title="Vectorless RAG API",
    description="A FastAPI backend for Vectorless RAG using Neo4j and Gemini",
    version="1.0.0",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    # Log request arrival
    logger.info(f"REQUEST START: {request.method} {request.url.path}")
    
    try:
        response = await call_next(request)
    except Exception as e:
        logger.error(f"REQUEST ERROR: {request.method} {request.url.path} | Error: {str(e)}")
        raise
    
    process_time = (time.time() - start_time) * 1000
    formatted_process_time = "{0:.2f}".format(process_time)
    
    logger.info(f"REQUEST COMPLETE: {request.method} {request.url.path} | Status: {response.status_code} | Time: {formatted_process_time}ms")
    
    return response

# Include the main API router
app.include_router(api_router, prefix="/api/v1")

@app.get("/")
async def root():
    return {
        "status": "online",
        "message": "Vectorless RAG Backend API",
        "version": settings.VERSION,
        "docs_url": "/docs",
        "health_url": "/health"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": settings.VERSION,
        "service": settings.PROJECT_NAME
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
