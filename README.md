# DocQABot

A document question-answering API built with FastAPI, PostgreSQL + pgvector, and OpenAI. Upload text, Markdown, or PDF files, then ask questions in natural language. Answers are grounded in your documents and cite which files they came from.

## How it works

1. **Upload**: `POST /upload` extracts text from the file (PDFs via `pdfplumber`) and creates a `documents` row with status `processing`.
2. **Chunk**: in a background task, the text is split into ~400-word chunks along paragraph boundaries, with a 50-word overlap between neighboring chunks.
3. **Embed**: all chunks are embedded in one batch with OpenAI `text-embedding-3-small` (1536 dimensions) and stored in the `chunks` table. The document is then marked `ready`, or `failed` on error.
4. **Retrieve**: `POST /ask` embeds the question and runs a cosine-similarity search (IVFFlat index) over chunks from ready documents. It keeps the top 5 matches with similarity above 0.3.
5. **Generate**: the retrieved chunks, plus the last 4 chat-history messages, are sent to `gpt-4o-mini`. The model is told to answer only from that context and to name its sources.

## Project structure

```
app/
├── main.py         # FastAPI app, endpoints, background document processing
├── chunker.py      # Paragraph-aware chunking with overlap
├── embeddings.py   # OpenAI embedding helpers (single + batch)
├── rag.py          # Retrieval (pgvector) and answer generation
└── database.py     # Connection helper and schema initialization
requirements.txt
```

## Requirements

- Python 3.10+
- PostgreSQL with the [pgvector](https://github.com/pgvector/pgvector) extension
- An OpenAI API key

## Setup

1. **Start PostgreSQL with pgvector.** The easiest way is Docker:

   ```bash
   docker run -d --name docqabot-db -e POSTGRES_PASSWORD=postgres -p 5432:5432 pgvector/pgvector:pg16
   ```

2. **Install dependencies:**

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Create a `.env` file** in the project root:

   ```env
   DATABASE_URL=postgresql://postgres:postgres@localhost:5432/postgres
   OPENAI_API_KEY=sk-...
   ```

4. **Run the server:**

   ```bash
   uvicorn app.main:app --reload
   ```

   On startup, the app enables the `vector` extension and creates the `documents` and `chunks` tables if they don't exist. Interactive API docs are at http://localhost:8000/docs.

## API

### `POST /upload`

Upload a `.txt`, `.md`, or `.pdf` file (multipart form field `file`). Processing runs in the background.

```bash
curl -F "file=@handbook.pdf" http://localhost:8000/upload
```

```json
{ "message": "Document uploaded", "document_id": 1, "status": "processing" }
```

Returns `400` for unsupported file types, text files that aren't UTF-8, or files with no extractable text.

### `GET /documents`

List uploaded documents with their processing status (`processing`, `ready`, or `failed`), newest first.

```bash
curl http://localhost:8000/documents
```

### `POST /ask`

Ask a question about the documents that are ready. `chat_history` is optional and takes OpenAI-style messages for follow-up questions.

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
        "question": "What is the vacation policy?",
        "chat_history": [
          {"role": "user", "content": "Who is eligible for benefits?"},
          {"role": "assistant", "content": "Full-time employees after 30 days..."}
        ]
      }'
```

```json
{
  "answer": "Employees get 20 days of paid vacation per year (source: handbook.pdf).",
  "sources": ["handbook.pdf"]
}
```

If no chunk is similar enough to the question, the bot says it couldn't find relevant information. It doesn't guess.

## Configuration

These defaults are set in code:

| Setting | Value | Location |
| --- | --- | --- |
| Chunk size / overlap | 400 / 50 words | `app/chunker.py` |
| Embedding model | `text-embedding-3-small` | `app/embeddings.py` |
| Retrieval top-k / similarity threshold | 5 / 0.3 | `app/rag.py` |
| Chat model | `gpt-4o-mini` (temperature 0.3, max 500 tokens) | `app/rag.py` |
| Chat history kept | last 4 messages | `app/rag.py` |
