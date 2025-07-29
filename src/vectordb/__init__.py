from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Dict, Any
import uuid
import time
import numpy as np
def upload_to_qdrant(
    chunks: List[str],
    embeddings: List[np.ndarray],
    collection_name: str,
    metadata_list: List[Dict[str, Any]] = None,
    client: QdrantClient = None,
    batch_size: int = 100,
    max_retries: int = 3,
    backoff_factor: float = 2.0
):
    """
    Đưa dữ liệu đã được embedding vào Qdrant
    
    Args:
        chunks: Danh sách các đoạn văn bản
        embeddings: Danh sách các vector embedding tương ứng
        collection_name: Tên collection trong Qdrant
        metadata_list: Danh sách metadata cho từng chunk (optional)
        qdrant_host: Địa chỉ host Qdrant
        qdrant_port: Port Qdrant
        batch_size: Số lượng điểm dữ liệu upload mỗi lần
        max_retries: Số lần thử lại tối đa khi gặp lỗi
        backoff_factor: Hệ số tăng thời gian chờ khi retry
    """
    
    try:
        client.get_collection(collection_name)
    except Exception:
        vector_size = len(embeddings[0]) if embeddings else 768
        
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE
            )
        )
        print(f"Created new collection: {collection_name}")

    points = []
    for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):

        metadata = metadata_list[idx] if metadata_list and idx < len(metadata_list) else {}
        metadata["text"] = chunk
        

        point = models.PointStruct(
            id=str(uuid.uuid4()),  
            vector=embedding.tolist(),  
            payload=metadata
        )
        points.append(point)

    total_points = len(points)
    for i in range(0, total_points, batch_size):
        batch = points[i:i + batch_size]
        attempt = 0
        
        while attempt < max_retries:
            try:

                client.upsert(
                    collection_name=collection_name,
                    points=batch,
                    wait=True
                )
                break
            except Exception as e:
                attempt += 1
                if attempt >= max_retries:
                    raise RuntimeError(
                        f"Upload to Qdrant failed after {max_retries} attempts: {str(e)}"
                    ) from e
                

                wait_time = backoff_factor ** attempt
                print(f"Error: {str(e)}. Retrying in {wait_time:.1f} seconds...")
                time.sleep(wait_time)

        processed = min(i + batch_size, total_points)
        print(f"Uploaded {processed}/{total_points} points ({processed/total_points:.1%})")
    
    print(f"Successfully uploaded {total_points} points to collection '{collection_name}'")


def delete_collection(collection_name: str, host: str = "localhost", port: int = 6333):
    """
    Xóa hoàn toàn collection trong Qdrant
    
    Args:
        collection_name: Tên collection cần xóa
        host: Địa chỉ Qdrant server
        port: Port Qdrant server
    """
    client = QdrantClient(host=host, port=port)
    
    try:
        client.delete_collection(collection_name)
        print(f"✅ Collection '{collection_name}' đã được xóa hoàn toàn")
    except Exception as e:
        print(f" Lỗi xóa collection: {str(e)}")

delete_collection("academic_documents")

