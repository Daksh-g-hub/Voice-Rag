import os
import uvicorn
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from fastapi.staticfiles import StaticFiles

from .config import settings
from .orchestration.pipeline import orchestrator, RAGPipelineResponse
from .services.vector_store import vector_store
from .services.cache_service import cache_service

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Low-Latency Voice-Enabled RAG System with MSMARCO-XI Hierarchical Indexing and Grounding Guardrails."
)

# Enable CORS for local React/Vite development and deployed frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount frontend static files if directory exists
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/app", StaticFiles(directory=frontend_dir, html=True), name="frontend")


class TextQueryRequest(BaseModel):
    query: str
    language_code: Optional[str] = "en-IN"
    use_cache: bool = True

@app.get("/")
async def root():
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "indexed_vectors": vector_store.count(),
        "embedding_model": settings.EMBEDDING_MODEL_NAME,
        "llm_model": settings.LLM_MODEL
    }

@app.get("/health")
async def health_check():
    return {
        "status": "online",
        "indexed_points": vector_store.count(),
        "storage_path": settings.QDRANT_STORAGE_PATH,
        "reranker_enabled": settings.ENABLE_RERANKER
    }

@app.post("/api/query/text", response_model=RAGPipelineResponse)
async def query_text(payload: TextQueryRequest):
    """
    Direct text query endpoint with complete pipeline execution and latency metrics.
    """
    try:
        response = await orchestrator.execute_query(
            query=payload.query,
            language_code=payload.language_code,
            use_cache=payload.use_cache
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline processing error: {str(e)}")

@app.post("/api/query/voice", response_model=RAGPipelineResponse)
async def query_voice(
    audio: UploadFile = File(...),
    language_code: Optional[str] = Form("en-IN"),
    use_cache: Optional[bool] = Form(True)
):
    """
    Voice query endpoint: receives raw audio (WAV/PCM/WebM), performs STT via Sarvam AI,
    and runs the full RAG pipeline with stage-by-stage latency tracking.
    """
    try:
        audio_bytes = await audio.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio payload received.")

        response = await orchestrator.execute_query(
            audio_bytes=audio_bytes,
            language_code=language_code,
            use_cache=use_cache
        )
        return response
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Voice processing error: {str(e)}")

@app.post("/api/cache/clear")
async def clear_cache():
    cache_service.clear()
    return {"message": "Query cache cleared successfully."}

@app.post("/api/admin/reindex")
async def reindex_dataset():
    """Rebuilds the vector index from MSMARCO-XI dataset live without restarting server."""
    try:
        from scripts.download_msmarco import prepare_msmarco_sample
        from backend.app.rag.chunker import HierarchicalChunker
        from backend.app.services.embedding_service import embedding_service
        import json

        data_file = prepare_msmarco_sample()
        with open(data_file, "r", encoding="utf-8") as f:
            docs = json.load(f)

        chunker = HierarchicalChunker(parent_chunk_size=350, parent_overlap=40, child_chunk_size=90, child_overlap=25)
        parents = []
        child_texts = []
        for d in docs:
            p_chunks = chunker.chunk_document(d["doc_id"], d["text"], d.get("language", "en"), {"title": d.get("title", ""), "url": d.get("url", "")})
            for p in p_chunks:
                parents.append(p)
                for c in p.children:
                    child_texts.append(c.text)

        embeddings = embedding_service.embed_documents(child_texts, batch_size=128)
        inserted = vector_store.insert_parent_chunks(parents, embeddings, batch_size=256)
        cache_service.clear()

        return {
            "status": "reindexed",
            "documents_loaded": len(docs),
            "parent_chunks": len(parents),
            "child_vectors_indexed": inserted,
            "total_points_in_db": vector_store.count()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reindexing failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("backend.app.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
