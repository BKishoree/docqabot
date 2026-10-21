#Load .env before app modules create their OpenAI clients / DB connections
from dotenv import load_dotenv
load_dotenv()

from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, HTTPException, BackgroundTasks
from pydantic import BaseModel
from app.database import get_connection, init_db
from app.chunker import chunk_document
from app.embeddings import get_embeddings_batch
from app.rag import ask
import pdfplumber
import io

#Initialize database on startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

app = FastAPI(title="Document Q&A Bot", lifespan=lifespan)

#Request/Response models
class QuestionRequest(BaseModel):
    question:str
    chat_history: list[dict] | None = None

class AnswerResponse(BaseModel):
    answer: str
    sources: list[str]

#Background task for document processing
def process_document(doc_id: int, content: str, filename: str):
    conn = get_connection()

    try:
        #Chunk the document (skip empty chunks, the embeddings API rejects them)
        chunks = [c for c in chunk_document(content) if c.strip()]

        #Generate embeddings in batch
        embeddings = get_embeddings_batch(chunks)

        #Store chunks with embeddings
        with conn.cursor() as curr:
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                curr.execute(
                    """INSERT INTO chunks (document_id, content, embedding, chunk_index)
                    VALUES (%s, %s, %s, %s)""",
                    (doc_id, chunk, embedding, i)
                )

            #Mark document as ready
            curr.execute(
                "UPDATE documents SET status = 'ready' WHERE id = %s",
                (doc_id,)
            )

        conn.commit()
        print(f"Processed {filename}: {len(chunks)} chunks")

    except Exception as e:
        #Mark as failed (rollback first, the transaction is aborted after an error)
        conn.rollback()
        with conn.cursor() as curr:
            curr.execute(
                """UPDATE documents SET status = 'failed' WHERE id = %s""",
                (doc_id,)
            )
        conn.commit()
        print(f"Failed to process {filename}: {e}")

    finally:
        conn.close()

@app.post("/upload")
async def upload_document(file: UploadFile, background_tasks: BackgroundTasks):
    """Upload a document for processing."""

    #Validate file type
    allowed_types = [".txt", ".md", ".pdf"]
    filename = (file.filename or "").lower()
    if not any (filename.endswith(t) for t in allowed_types):
        raise HTTPException(400, f"File type not supported. Use: {allowed_types}")

    #Read content
    content = await file.read()

    #Handle different file types
    if filename.endswith(".pdf"):
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            text = "\n\n".join(page.extract_text() or "" for page in pdf.pages)
    else:
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(400, "File must be UTF-8 encoded text")

    if not text.strip():
        raise HTTPException(400, "No text could be extracted from the file")

    # Create document record
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO documents (filename) VALUES (%s) RETURNING id",
                (file.filename,)
            )
            doc_id = cur.fetchone()["id"]
        conn.commit()
    finally:
        conn.close()

    # Process in background
    background_tasks.add_task(process_document, doc_id, text, file.filename)

    return {"message": "Document uploaded", "document_id": doc_id, "status": "processing"}

@app.get("/documents")
def list_documents():
    """List all uploaded documents and their status."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, filename, status, created_at FROM documents ORDER BY created_at DESC")
            docs = cur.fetchall()
    finally:
        conn.close()
    return docs

@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    """Ask a question about the uploaded documents."""
    result = ask(request.question, request.chat_history)
    return AnswerResponse(
        answer=result["answer"],
        sources=result["sources"]
    )
