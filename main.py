"""FastAPI RAG application."""

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Security, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader

from models import QueryRequest, QueryResponse
from cache import QueryCache
from rag.loader import load_documents
from rag.chunker import chunk_documents
from rag.embedder import Embedder
from rag.faiss_index import build_index, save_index, load_index, empty_index
from rag.retriever import retrieve
from rag.prompt import build_messages
from rag.generator import Generator, GeneratorError

# ---------------------------------------------------------------------------
# Load .env file if present
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Configuration via environment variables
# ---------------------------------------------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "3"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DATA_PATH = os.getenv("DATA_PATH", os.path.join(os.path.dirname(__file__), "data"))
INDEX_PATH = os.getenv("INDEX_PATH", os.path.join(os.path.dirname(__file__), "index"))
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

# Optional API key auth (set RAG_API_KEY env var to enable)
RAG_API_KEY = os.getenv("RAG_API_KEY", "")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional API key authentication
# ---------------------------------------------------------------------------
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str | None = Security(api_key_header)) -> None:
    """If RAG_API_KEY is set, require matching X-API-Key header."""
    if not RAG_API_KEY:
        return  # auth disabled
    if api_key != RAG_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ---------------------------------------------------------------------------
# Index build helper (reused by lifespan and rebuild endpoint)
# ---------------------------------------------------------------------------
async def _build_index_from_docs(embedder: Embedder):
    """Load docs, chunk, embed, build+save FAISS index. Returns (index, chunks)."""
    docs = load_documents(DATA_PATH)
    texts = [doc.text for doc in docs]
    chunks = chunk_documents(texts, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    if chunks:
        embeddings = await asyncio.to_thread(embedder.embed_batch, chunks)
        index = build_index(embeddings)
        save_index(index, chunks, INDEX_PATH)
        logger.info("Built and saved index with %d chunks from %d documents", len(chunks), len(docs))
    else:
        logger.warning("No chunks produced — using empty index")
        index = empty_index(embedder.dimension)
        chunks = []
    return index, chunks


# ---------------------------------------------------------------------------
# Lifespan: load models & index once
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
    embedder = Embedder(model_name=EMBEDDING_MODEL)
    app.state.embedder = embedder

    # Try loading existing index
    loaded = load_index(INDEX_PATH)
    if loaded is not None:
        index, chunks = loaded
        logger.info("Loaded existing index (%d vectors, %d chunks)", index.ntotal, len(chunks))
    else:
        logger.info("No existing index found — building from documents in %s", DATA_PATH)
        index, chunks = await _build_index_from_docs(embedder)

    app.state.index = index
    app.state.chunks = chunks
    app.state.cache = QueryCache()
    app.state.generator = Generator(model=OPENAI_MODEL)

    yield  # app runs

    logger.info("Shutting down")


app = FastAPI(title="RAG API", lifespan=lifespan)

# ---------------------------------------------------------------------------
# CORS middleware
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "index_vectors": app.state.index.ntotal,
        "chunks_loaded": len(app.state.chunks),
    }


# ---------------------------------------------------------------------------
# POST /index/rebuild
# ---------------------------------------------------------------------------
@app.post("/index/rebuild", dependencies=[Depends(verify_api_key)])
async def rebuild_index():
    """Rebuild the FAISS index from documents in data/ without restarting."""
    embedder: Embedder = app.state.embedder
    index, chunks = await _build_index_from_docs(embedder)
    app.state.index = index
    app.state.chunks = chunks
    app.state.cache = QueryCache()  # clear stale cache
    return {
        "status": "rebuilt",
        "chunks": len(chunks),
        "vectors": index.ntotal,
    }


# ---------------------------------------------------------------------------
# POST /query
# ---------------------------------------------------------------------------
@app.post("/query", response_model=QueryResponse, dependencies=[Depends(verify_api_key)])
async def query(req: QueryRequest):
    cache: QueryCache = app.state.cache
    embedder: Embedder = app.state.embedder
    gen: Generator = app.state.generator

    t_start = time.perf_counter()

    # Cache check
    cached = cache.get(req.question, req.top_k)
    if cached is not None:
        total_ms = (time.perf_counter() - t_start) * 1000
        return QueryResponse(
            answer=cached,
            chunks=[],
            latency_ms=round(total_ms, 2),
            embedding_ms=0,
            retrieval_ms=0,
            generation_ms=0,
            top_k=req.top_k,
            cache_hit=True,
        )

    # Embed
    t0 = time.perf_counter()
    query_vec = await asyncio.to_thread(embedder.embed_text, req.question)
    embedding_ms = (time.perf_counter() - t0) * 1000

    # Retrieve
    t0 = time.perf_counter()
    top_chunks = await asyncio.to_thread(
        retrieve, query_vec, app.state.index, app.state.chunks, req.top_k
    )
    retrieval_ms = (time.perf_counter() - t0) * 1000

    # Generate
    messages = build_messages(top_chunks, req.question)
    t0 = time.perf_counter()
    try:
        answer = await gen.generate(messages)
    except GeneratorError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    generation_ms = (time.perf_counter() - t0) * 1000

    total_ms = (time.perf_counter() - t_start) * 1000

    logger.info(
        "query | embed=%.1fms  retrieve=%.1fms  generate=%.1fms  total=%.1fms",
        embedding_ms, retrieval_ms, generation_ms, total_ms,
    )

    cache.set(req.question, req.top_k, answer)

    return QueryResponse(
        answer=answer,
        chunks=top_chunks,
        latency_ms=round(total_ms, 2),
        embedding_ms=round(embedding_ms, 2),
        retrieval_ms=round(retrieval_ms, 2),
        generation_ms=round(generation_ms, 2),
        top_k=req.top_k,
        cache_hit=False,
    )


# ---------------------------------------------------------------------------
# WebSocket /stream
# ---------------------------------------------------------------------------
@app.websocket("/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    embedder: Embedder = app.state.embedder
    gen: Generator = app.state.generator

    try:
        while True:
            question = await ws.receive_text()
            t_start = time.perf_counter()

            # Embed
            t0 = time.perf_counter()
            query_vec = await asyncio.to_thread(embedder.embed_text, question)
            embedding_ms = (time.perf_counter() - t0) * 1000

            # Retrieve
            t0 = time.perf_counter()
            top_chunks = await asyncio.to_thread(
                retrieve, query_vec, app.state.index, app.state.chunks, DEFAULT_TOP_K
            )
            retrieval_ms = (time.perf_counter() - t0) * 1000

            # Stream tokens
            messages = build_messages(top_chunks, question)
            t0 = time.perf_counter()
            try:
                async for token in gen.generate_stream(messages):
                    await ws.send_text(token)
            except GeneratorError as exc:
                await ws.send_text(json.dumps({"error": str(exc)}))
                continue
            generation_ms = (time.perf_counter() - t0) * 1000

            total_ms = (time.perf_counter() - t_start) * 1000

            # Final latency payload
            await ws.send_text(
                json.dumps({
                    "done": True,
                    "latency_ms": round(total_ms, 2),
                    "embedding_ms": round(embedding_ms, 2),
                    "retrieval_ms": round(retrieval_ms, 2),
                    "generation_ms": round(generation_ms, 2),
                })
            )
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
