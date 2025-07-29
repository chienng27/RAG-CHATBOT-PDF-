import heapq
from typing import List, Tuple, Optional
from sentence_transformers import CrossEncoder
from functools import lru_cache

@lru_cache(maxsize=1)
def load_reranker(model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2') -> CrossEncoder:
    """Cache model to avoid reloading on every function call"""
    return CrossEncoder(model_name, max_length=512, device='cpu')

def rerank_with_cross_encoder(
    query: str, 
    candidate_texts: List[str], 
    k: int = 1, 
    reranker: Optional[CrossEncoder] = None,
    batch_size: int = 64
) -> List[Tuple[str, float]]:
    """
    Optimized CPU version with:
    - Model caching
    - Heap-based top-k selection
    - Memory-efficient processing
    
    Parameters:
    - query: Input query
    - candidate_texts: Candidate texts from Qdrant
    - k: Number of results to return
    - reranker: Pre-loaded model instance (optional)
    - batch_size: CPU-optimized batch size (64)
    
    Returns:
    - Top-k results as [(text, score)] sorted by relevance
    """
    if not candidate_texts or k <= 0:
        return []

    if reranker is None:
        reranker = load_reranker()
 
    def generate_inputs():
        for text in candidate_texts:
            yield (query, text)

    top_results = []
    min_score = -10  
    
    for i in range(0, len(candidate_texts), batch_size):
        batch = list(generate_inputs())[i:i+batch_size]
        batch_scores = reranker.predict(
            batch,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        

        for j, score in enumerate(batch_scores):
            idx = i + j
            text = candidate_texts[idx]
            if len(top_results) < k:
                heapq.heappush(top_results, (score, text))
                min_score = top_results[0][0]
            elif score > min_score:
                heapq.heapreplace(top_results, (score, text))
                min_score = top_results[0][0]

    sorted_results = sorted(top_results, key=lambda x: x[0], reverse=True)
    return [(text, score) for score, text in sorted_results]