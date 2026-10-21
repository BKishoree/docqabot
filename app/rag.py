from openai import OpenAI
from app.database import get_connection
from app.embeddings import get_embeddings
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def retrieve_context(query: str, top_k: int = 5, threshold: float=0.3) -> list[dict]:
    """Find relevant chunks for a query."""
    query_embedding = get_embeddings(query)
    conn = get_connection()
    
    with conn.cursor() as curr:
        curr.execute(
            """
            SELECT 
            c.content, 
            d.filename,
            1 - (c.embedding <=> %s::vector) as similarity 
            FROM chunks c
            JOIN documents d ON c.document_id = d.id 
            WHERE d.status = 'ready' 
            AND 1 - (c.embedding <=> %s::vector) > %s 
            ORDER BY c.embedding <=> %s::vector 
            LIMIT %s 
            """, (query_embedding, query_embedding, threshold, query_embedding, top_k)
        )
        
        results = [
            {"content": row["content"], "source": row["filename"], "similarity": row["similarity"]} 
            for row in curr.fetchall()
        ]
    conn.close()
    return results

def generate_answer(query: str, context_chunks: list[dict], chat_history: list = None) -> str:
    """Generate answer using retrieved context."""
    if not context_chunks:
        return "I couldn't find relevant information in the uploaded documents to answer your question."
    
    #Build context string 
    context = "\n\n---\n\n".join([
        f"[Source: {c['source']}]\n{c['content']}"
        for c in context_chunks
    ])
    
    #Build Messages
    messages=[
        {
            "role": "system", 
            "content": """You are a helpful assistant that answers questions based on the provided documents.
            Rules.
            - Only answer based on the provided context
            - If the context doesn't contain the answer, say so 
            - Be concise but complete 
            - Mention which document(s) the information came from 
            - If asked a follow-up question, use the conversation history for context
            """
        }
    ]
    
    #Add Chat History for follow-ups
    if chat_history:
        messages.extend(chat_history[-4:]) #Last 2 exchanes
    
    #Add current query with context
    messages.append({
        "role": "user", 
        "content": f"""Based on these documents: 
        {context}
        
        ---
        
        Question: {query}
        
        """
    })
    
    response = client.chat.completions.create(
        model="gpt-4o-mini", 
        messages=messages, 
        temperature=0.3, #more focused
        max_tokens=500
    )
    
    return response.choices[0].message.content

def ask(query: str, chat_history: list = None) -> dict:
    """Main RAG function: retrieve and generate."""
    #Retrieve
    chunks = retrieve_context(query)
    
    #Generate
    answer = generate_answer(query, chunks, chat_history)
    
    #Return with metadata
    return {
        "answer": answer, 
        "sources": list(set([c["source"] for c in chunks])), 
        "chunks_used": len(chunks)
    }