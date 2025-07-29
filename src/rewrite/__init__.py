import google.generativeai as genai
import logging

logging.basicConfig(level=logging.INFO)

def enhance_query(
    query: str, 
    model_name: str = "gemini-1.5-flash-latest",
    api_key: str = None,
    rewrite_count: int = 0,
    max_rewrites: int = 3
) -> str:
    if rewrite_count >= max_rewrites:
        logging.warning(f"Đạt giới hạn rewrite tối đa ({max_rewrites})")
        return query

    prompt = f"""
**Hệ thống RAG Query Optimization**
**Phiên bản rewrite**: {rewrite_count + 1}/{max_rewrites}

**Nhiệm vụ**: Cải thiện truy vấn tìm kiếm sau bằng cách:
1. Bổ sung từ khoá quan trọng tiềm năng
2. Tăng tính cụ thể và rõ ràng
3. Giữ nguyên ý định gốc
4. Không thay đổi ngữ nghĩa gốc

**Truy vấn gốc**: "{query}"

**Yêu cầu**:
- Giữ cấu trúc tự nhiên
- Trả về TRUY VẤN TỐI ƯU DUY NHẤT không có định dạng
- Trả về thành câu hỏi hoàn chỉnh, không cần thêm chú thích

**Kết quả**:
"""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.2
            )
        )
        rewritten = response.text.strip().strip('"')
        
        if not rewritten or rewritten.lower() == query.lower():
            logging.info("Không có cải thiện sau rewrite")
            return query
        
        return rewritten

    except Exception as e:
        logging.error(f"Lỗi Gemini: {str(e)}")
        return query
