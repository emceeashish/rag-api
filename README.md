# RAG API

Backend-only RAG service with FAISS retrieval, OpenAI generation, streaming, and latency profiling.

Retrieval-Augmented Generation (RAG) backend built with FastAPI, FAISS, sentence-transformers, and OpenAI.

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

Set environment variable:

```bash
# Windows PowerShell
$env:OPENAI_API_KEY="sk-..."
# Linux/macOS
export OPENAI_API_KEY="sk-..."
```

## Add documents

Place `.txt` files in `rag_api/data/`:

```
rag_api/
  data/
    notes.txt
    handbook.txt
```

## Build / rebuild index

The index is built automatically on startup if `rag_api/index/` does not contain an existing index.

To rebuild, delete saved index files and restart:

```bash
rm -f rag_api/index/index.faiss rag_api/index/chunks.json
```

Then restart the server.

## Run the server

```bash
cd rag_api
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Configuration (environment variables)

| Variable | Default |
|---|---|
| `CHUNK_SIZE` | `1000` |
| `CHUNK_OVERLAP` | `200` |
| `DEFAULT_TOP_K` | `3` |
| `EMBEDDING_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` |
| `DATA_PATH` | `./data` |
| `INDEX_PATH` | `./index` |
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` |
| `OPENAI_API_KEY` | *(required)* |

## API

### POST /query

`top_k` is optional and defaults to `DEFAULT_TOP_K` (3).

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question":"What is this repo about?","top_k":3}'
```

Example response:

```json
{
  "answer": "string",
  "latency_ms": 1234.5,
  "embedding_ms": 12.3,
  "retrieval_ms": 1.2,
  "generation_ms": 1220.0,
  "top_k": 3,
  "cache_hit": false
}
```

Generation latency dominates due to external LLM call; retrieval is typically sub-5 ms.

### WebSocket /stream

Connect with any WebSocket client (example with [websocat](https://github.com/vi/websocat)):

```bash
websocat ws://localhost:8000/stream
```

Send one plain-text question per message. The server keeps the connection open and repeats: receive a text message → stream tokens → send final latency JSON.
