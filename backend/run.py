"""
Run the FastAPI server 
"""
import sys
from pathlib import Path

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    import uvicorn
    from app.core.config import settings
    
    print(f"Project root: {project_root}")
    print(f"Data file: {settings.data_file}")
    
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
        log_level="info"
    )