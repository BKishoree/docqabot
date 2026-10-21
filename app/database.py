import psycopg2
from psycopg2.extras import RealDictCursor
import os

def get_connection():
    return psycopg2.connect(
        os.getenv("DATABASE_URL"), 
        cursor_factory=RealDictCursor
    )

def init_db():
    conn = get_connection()
    with conn.cursor() as curr:
        #Enable pgvector
        curr.execute("CREATE EXTENSION IF NOT EXISTS vector")
        
        #Documents table
        curr.execute(
            """
            CREATE TABLE IF NOT EXISTS documents(
                id SERIAL PRIMARY KEY, 
                filename TEXT NOT NULL,
                status TEXT DEFAULT 'processing',
                created_at TIMESTAMP DEFAULT now()
            )
            """
        )
        
        #CHUNKS table with embeddings
        curr.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks(
                id SERIAL PRIMARY KEY, 
                document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE, 
                content TEXT NOT NULL, 
                embedding vector(1536), 
                chunk_index INTEGER 
            )
            """
        )
        
        #Index for fast similarity search
        curr.execute(
            """
            CREATE INDEX IF NOT EXISTS chunks_embedding_idx
            ON chunks USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100)
            """
        )
    conn.commit()
    conn.close()