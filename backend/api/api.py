"""
FastAPI Server for RAG-powered Chatbot
Handles chat requests and returns AI-generated responses
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import logging
import sys
from pathlib import Path

# Add parent directory to path to import rag_engine
sys.path.append(str(Path(__file__).parent.parent))

from rag_engine import ExamRAG

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Chatbot RAG API",
    description="AI-powered chatbot with document retrieval",
    version="2.0.0"
)

# Add CORS middleware to allow Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG engine instance
rag_engine: Optional[ExamRAG] = None


# Request/Response models
class ChatRequest(BaseModel):
    message: str
    include_sources: bool = False
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "Làm thế nào để tạo khóa học mới?",
                "include_sources": False
            }
        }


class Source(BaseModel):
    file: str
    page: str
    content_preview: str


class ChatResponse(BaseModel):
    response: str
    sources: Optional[List[Source]] = None
    num_sources: Optional[int] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "response": "Để tạo khóa học mới, bạn cần...",
                "sources": [],
                "num_sources": 0
            }
        }


class HealthResponse(BaseModel):
    status: str
    rag_initialized: bool
    message: str


# Startup event: Initialize RAG engine
@app.on_event("startup")
async def startup_event():
    """
    Initialize the RAG engine when server starts
    This loads PDFs and builds the vector index
    """
    global rag_engine
    
    logger.info("=" * 50)
    logger.info("🚀 Starting FastAPI RAG Server")
    logger.info("=" * 50)
    
    try:
        logger.info("Initializing RAG Engine...")
        # Use absolute path from project root
        project_root = Path(__file__).parent.parent.parent
        pdf_directory = project_root / "data" / "pdf"
        index_path = Path(__file__).parent.parent / "faiss_index"
        
        rag_engine = ExamRAG(
            pdf_directory=str(pdf_directory),
            index_path=str(index_path)
        )
        
        # Initialize (will load existing index or build new one)
        rag_engine.initialize(force_rebuild=False)
        
        logger.info("✅ RAG Engine initialized successfully!")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize RAG Engine: {e}")
        logger.error("Server will start but /chat endpoint will not work")
        rag_engine = None


@app.get("/", response_model=Dict[str, str])
async def root():
    """
    Root endpoint - API information
    """
    return {
        "message": "Chatbot RAG API",
        "version": "2.0.0",
        "endpoints": {
            "health": "/health",
            "chat": "/chat (POST)"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    Returns server status and RAG initialization state
    """
    return {
        "status": "healthy",
        "rag_initialized": rag_engine is not None,
        "message": "RAG engine is ready" if rag_engine else "RAG engine not initialized"
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint
    
    Args:
        request: ChatRequest with user message
        
    Returns:
        ChatResponse with AI-generated answer and optional sources
    """
    if not rag_engine:
        raise HTTPException(
            status_code=503,
            detail="RAG engine not initialized. Please check server logs."
        )
    
    if not request.message or not request.message.strip():
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty"
        )
    
    logger.info(f"📨 Received chat request: {request.message[:100]}...")
    
    try:
        # Get answer from RAG engine
        result = rag_engine.get_answer(request.message)
        
        # Prepare response
        response = ChatResponse(
            response=result["answer"],
            sources=result["sources"] if request.include_sources else None,
            num_sources=result["num_sources"] if request.include_sources else None
        )
        
        logger.info(f"✅ Generated response with {result['num_sources']} sources")
        return response
        
    except Exception as e:
        logger.error(f"❌ Error processing chat request: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating response: {str(e)}"
        )


@app.post("/rebuild-index")
async def rebuild_index():
    """
    Rebuild the vector index from scratch
    Useful when PDF files are updated
    """
    global rag_engine
    
    if not rag_engine:
        raise HTTPException(
            status_code=503,
            detail="RAG engine not initialized"
        )
    
    logger.info("🔄 Rebuilding vector index...")
    
    try:
        rag_engine.initialize(force_rebuild=True)
        logger.info("✅ Index rebuilt successfully")
        
        return {
            "status": "success",
            "message": "Vector index rebuilt successfully"
        }
        
    except Exception as e:
        logger.error(f"❌ Error rebuilding index: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error rebuilding index: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    
    # Run server
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes
        log_level="info"
    )
