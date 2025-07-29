import os
from tempfile import NamedTemporaryFile
from __init__ import extract_and_process_pdf

def test_extract_and_process_pdf():
    # Đường dẫn đến file PDF mẫu
    pdf_path = "C:/Users/Chien/Downloads/cde58-Bang-so-xac-xuat-thong-ke.pdf"  
    if not os.path.exists(pdf_path):
        print(f"File PDF không tồn tại: {pdf_path}")
        return

    # Gọi hàm extract_and_process_pdf
    try:
        print("Đang xử lý file PDF...")
        chunks = extract_and_process_pdf(pdf_path, chunk_size=1500, chunk_overlap=200)
        print(f"Đã xử lý xong. Tổng số chunks: {len(chunks)}")

        with NamedTemporaryFile(delete=False, mode="w", encoding="utf-8", suffix=".txt") as temp_file:
            for i, chunk in enumerate(chunks):
                temp_file.write(f"Chunk {i + 1}:\n{chunk}\n{'-' * 50}\n")
            temp_file_path = temp_file.name

        print(f"Kết quả đã được ghi vào file tạm: {temp_file_path}")

    except Exception as e:
        print(f"Lỗi khi xử lý file PDF: {str(e)}")

if __name__ == "__main__":
    test_extract_and_process_pdf()