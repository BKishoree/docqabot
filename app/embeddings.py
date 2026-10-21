from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embeddings(text: str) -> list[float]:
    """Generate embedding for a single text."""
    response = client.embeddings.create(
        model = "text-embedding-3-small", 
        input = text
    )
    return response.data[0].embedding

def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for multiple texts (more efficient)."""
    response = client.embeddings.create(
        model = "text-embedding-3-small", 
        input = texts
    )
    return [item.embedding for item in response.data]