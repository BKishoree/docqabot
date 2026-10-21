import re

def chunk_document(text: str, chunk_size: int = 400, overlap: int = 50) -> list[str]:
    """
    Split document into chunks, trying to respect paragraph boundaries.
    """
    #Clean the text
    text = re.sub(r'\n{3,}', '\n\n', text) #Normalize multiple in new lines
    text = re.sub(r' {2,}', ' ', text) #Normalize spaces
    
    #Split the paragraphs first
    paragraphs = text.split('\n\n')
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for para in paragraphs:
        para_words = para.split()
        para_length = len(para_words)
        
        if current_length + para_length <= chunk_size:
            current_chunk.append(para)
            current_length+=para_length
        else:
            #Saves current chunk
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
            #Start new chunk (with overlap from previous)
            if chunks and overlap > 0:
                #Get last N words from previous chunk
                prev_words = chunks[-1].split()[-overlap:]
                current_chunk = [' '.join(prev_words), para]
                current_length = overlap + para_length
            else:
                current_chunk = [para]
                current_length = para_length
    
    #Don't forget to add the last_chunk
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
    
    return chunks