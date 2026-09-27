import os
import sys

# Prevent OpenMP runtime conflict between PyTorch and XGBoost on macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["JOBLIB_MULTIPROCESSING"] = "0"
os.environ["JOBLIB_START_METHOD"] = "threading"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import uvicorn

if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.abspath(__file__))
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)
        
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    enable_docs = os.getenv("ENABLE_DOCS", "false").lower() in ("true", "1", "yes")

    print("=" * 65)
    print("🚀 TelcoPulse CRM — Enterprise Customer Retention Platform")
    print(f"🌐 Full-Stack App URL : http://{host}:{port}")
    if enable_docs:
        print(f"📚 Interactive API Docs: http://{host}:{port}/docs")
    else:
        print("🔒 Interactive API Docs: Disabled for Security (set ENABLE_DOCS=true to enable)")
    print("=" * 65)
    
    from Frontend_BackEnd.server import app
    uvicorn.run(app, host=host, port=port, log_level="info")
