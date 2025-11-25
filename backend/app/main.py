"""
Main entry point for the Personal AI Assistant Backend.
Run with: python -m app.main or uvicorn app.main:app --reload
"""
from app.server import app

# Re-export the FastAPI app for uvicorn
__all__ = ["app"]

if __name__ == "__main__":
    import uvicorn
    from app.config import settings
    
    print("🤖 Personal AI Assistant - Starting Server...")
    print(f"📍 URL: http://{settings.host}:{settings.port}")
    print(f"📚 Docs: http://{settings.host}:{settings.port}/docs")
    
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )
