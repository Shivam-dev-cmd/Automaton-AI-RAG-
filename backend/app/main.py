import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"

from backend.app.config import settings
from backend.app.schemas import (
    ChatRequest, ChatResponse,
    SearchRequest, SearchResponse, SearchResult,
    HealthResponse, KnowledgeSummaryResponse
)
from backend.app.rag.pipeline import RAGPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("main")

# Global pipeline instance
rag_pipeline: RAGPipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_pipeline
    logger.info("Initializing Automaton AI RAG Pipeline...")
    rag_pipeline = RAGPipeline()
    logger.info(f"RAG Pipeline initialized with {len(rag_pipeline.doc_store.chunks)} chunks.")
    yield
    logger.info("Shutting down Automaton AI Chatbot server.")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="High-precision, Zero-Hallucination Enterprise RAG Chatbot API for Automaton AI Infosystem Pvt. Ltd.",
    lifespan=lifespan
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["Frontend"])
async def root():
    """Serves the standalone floating chatbot frontend."""
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "online"
    }

@app.get("/api", tags=["Root"])
async def api_info():
    return {
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "online",
        "docs": "/docs",
        "endpoints": ["/api/chat", "/api/search", "/api/health", "/api/knowledge/summary", "/api/eval"]
    }

@app.get("/api/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    chunks_count = len(rag_pipeline.doc_store.chunks) if rag_pipeline else 0
    return HealthResponse(
        status="healthy",
        version=settings.VERSION,
        chunks_count=chunks_count,
        llm_provider=settings.LLM_PROVIDER,
        knowledge_base_loaded=chunks_count > 0
    )

@app.post("/api/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Main conversational endpoint:
    - Pure async request handling
    - Pre-retrieval intent guardrail (pricing non-disclosure, out-of-scope rejection, careers)
    - Hybrid BM25 + keyword boosted retrieval
    - Grounded generation with citation verification
    - Post-generation anti-hallucination sanitization
    """
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG Pipeline not ready")
    
    try:
        response = await rag_pipeline.process_chat(
            message=request.message,
            history=request.history
        )
        return response
    except Exception as e:
        logger.error(f"Error processing chat request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal chat processing error: {str(e)}")

@app.post("/api/search", response_model=SearchResponse, tags=["Search"])
async def search(request: SearchRequest):
    """
    Direct inspection endpoint to search and debug retrieved chunks.
    """
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG Pipeline not ready")
        
    retrieved = rag_pipeline.retriever.search(request.query, top_k=request.top_k)
    results = [
        SearchResult(
            id=chunk.id,
            title=chunk.title,
            category=chunk.category,
            score=score,
            content=chunk.content,
            url=chunk.url,
            caveats=chunk.caveats
        )
        for chunk, score in retrieved
    ]
    return SearchResponse(query=request.query, results=results)

@app.get("/api/knowledge/summary", response_model=KnowledgeSummaryResponse, tags=["Knowledge"])
async def knowledge_summary():
    """
    Provides a high-level summary of loaded knowledge base categories and entities.
    """
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG Pipeline not ready")
        
    doc_store = rag_pipeline.doc_store
    cat_counts = {cat: len(chunks) for cat, chunks in doc_store._category_map.items()}
    
    products = [c.title.replace("Product: ", "") for c in doc_store.get_by_category("product")]
    services = [c.title.replace("Service: ", "") for c in doc_store.get_by_category("service")]
    
    # Model gallery items
    model_chunk = doc_store.get_by_id("model-gallery-catalog")
    models = []
    if model_chunk:
        for line in model_chunk.content.split("\n"):
            if line.startswith("- "):
                models.append(line.split(":")[0].replace("- ", "").strip())
                
    # Datasets
    data_chunk = doc_store.get_by_id("data-zoo-catalog")
    datasets = []
    if data_chunk:
        for line in data_chunk.content.split("\n"):
            if line.startswith("- "):
                datasets.append(line.split(":")[0].replace("- ", "").strip())

    return KnowledgeSummaryResponse(
        company_name="Automaton AI Infosystem Pvt. Ltd.",
        tagline="AI with Purpose, Progress with Precision",
        categories=cat_counts,
        total_chunks=len(doc_store.chunks),
        products=products,
        services=services,
        datasets=datasets,
        models=models
    )

@app.post("/api/eval", tags=["Evaluation"])
async def run_evaluation():
    """
    Runs automated evaluation test suite on the live RAG pipeline.
    Tests faithfulness, pricing guardrails, 404 product handling, out-of-scope rejection, and attribution.
    """
    from backend.app.tests.eval_suite import run_evaluations
    results = await run_evaluations(rag_pipeline)
    return results

# Mount static frontend directory at the end so all /api endpoints take precedence
from pathlib import Path
from fastapi.staticfiles import StaticFiles

FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host=settings.HOST, port=settings.PORT, reload=True)
