from langchain_google_genai import GoogleGenerativeAIEmbeddings
from typing import List, Union
import numpy as np
import time

def generate_embeddings(
    chunks: List[str],
    api_key: str,
    title: str = "Document",
    batch_size: int = 100,
    max_retries: int = 3,
    backoff_factor: float = 2.0
) -> List[np.ndarray]:
    """
    Tạo embeddings cho các chunks văn bản sử dụng GoogleGenerativeAIEmbeddings của LangChain
    """
    embedding_function = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        task_type="retrieval_document",
        title=title,
        google_api_key=api_key
    )
    
    embeddings = []
    total_chunks = len(chunks)

    for i in range(0, total_chunks, batch_size):
        batch = chunks[i:i + batch_size]

        batch = [
            f"[TABLE] {chunk}" if "[BẢNG" in chunk else chunk
            for chunk in batch
        ]
        
        batch_embeddings = None
        attempt = 0
        
        while attempt < max_retries:
            try:

                batch_embeddings = embedding_function.embed_documents(batch)
                break
            except Exception as e:
                attempt += 1
                if attempt >= max_retries:
                    raise RuntimeError(
                        f"Embedding generation failed after {max_retries} attempts: {str(e)}"
                    ) from e

                wait_time = backoff_factor ** attempt
                print(f"Error: {str(e)}. Retrying in {wait_time:.1f} seconds...")
                time.sleep(wait_time)

        if batch_embeddings:
            embeddings.extend([np.array(emb) for emb in batch_embeddings])
        
        processed = min(i + batch_size, total_chunks)
        print(f"Processed {processed}/{total_chunks} chunks ({processed/total_chunks:.1%})")
    
    return embeddings


def generate_query_embedding(
    query: str,
    api_key: str
) -> np.ndarray:
    """
    Tạo embedding cho một câu hỏi (query) duy nhất.
    
    Args:
        query: Câu hỏi của người dùng dưới dạng một chuỗi string.
        api_key: API key cho Google Generative AI.
        
    Returns:
        Một vector embedding dưới dạng numpy array.
    """

    embedding_function = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        task_type="retrieval_query", 
        google_api_key=api_key
    )
    

    embedding_vector = embedding_function.embed_query(query)
    
    return np.array(embedding_vector)