# Expert Network Search Copilot

Search 10,000 candidate profiles using natural language, powered by vector search and an LLM.

---

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

uvicorn main:app --reload
```

Open **http://localhost:8000/docs** to use the Swagger UI.

---

## Usage

**Step 1 — Index the candidates** (run once)
```bash
curl -X POST http://localhost:8000/ingest
```

**Step 2 — Search**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Python engineers with fintech experience in Dubai"}'
```

**Step 3 — Follow-up in the same conversation**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Filter those to only Arabic speakers",
    "conversation_id": "paste-id-from-previous-response"
  }'
```

---

## Endpoints

| Method | Endpoint  | Description                         |
|--------|-----------|-------------------------------------|
| POST   | /ingest   | Embed and index all candidates      |
| POST   | /chat     | Natural language search             |
| GET    | /health   | Check DB + vector store status      |

---

## Project Structure

```
main.py          # FastAPI app — all routes
models.py        # Pydantic request/response models
database.py      # PostgreSQL queries
embeddings.py    # Text → vectors (LangChain)
vectorstore.py   # ChromaDB (store + search vectors)
llm.py           # OpenRouter LLM calls
config.py        # Settings from .env
```

---

## Design Decisions

**Embeddings: `text-embedding-ada-002` via OpenRouter**

**Vector DB: ChromaDB**
Zero setup — runs in-process and persists to `./chroma_db`. No Docker needed.

**One chunk per candidate**
Each candidate is embedded as a single text block combining all fields. Profiles are short enough that splitting by field would hurt more than help.

**Query rewriting**
Before searching, the LLM expands the query with synonyms and regional variants — e.g. "Gulf" becomes "UAE, Dubai, Saudi Arabia, Qatar". This dramatically improves recall.

**Conversation context**
Each `/chat` call can include a `conversation_id` to continue a session. Prior turns are passed to the LLM so follow-up queries like "filter those to only Arabic speakers" work correctly.

**LLM: Mistral-7B via OpenRouter**
Fast and cheap for the two tasks it does here: query rewriting and match explanation. Easy to swap via `LLM_MODEL` in `.env`.
