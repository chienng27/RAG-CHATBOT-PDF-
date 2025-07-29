import re
from typing import List, Dict, Tuple, Any, Union
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTChar, LAParams, LTFigure, LTRect, LTLine, LTTextBox
from langchain.text_splitter import RecursiveCharacterTextSplitter
import camelot
import pandas as pd
import numpy as np

def extract_and_process_pdf(
    pdf_path: str, 
    chunk_size: int = 3000, 
    chunk_overlap: int = 200,
    table_flavor: str = 'lattice'
) -> List[str]:
    """
    Trích xuất và xử lý nội dung PDF tiếng Việt, kết hợp văn bản và bảng biểu
    
    Tham số:
        pdf_path: Đường dẫn tới file PDF
        chunk_size: Kích thước tối đa của mỗi đoạn văn bản
        chunk_overlap: Độ chồng lấp giữa các đoạn văn bản
        table_flavor: Phương pháp trích xuất bảng ('stream' hoặc 'lattice')
    
    Trả về:
        Danh sách các đoạn văn bản đã được xử lý
    """
    print(f"[BẮT ĐẦU] Xử lý file: {pdf_path}")
    print("[BẢNG] Đang trích xuất bảng từ PDF...")
    try:
        tables = camelot.read_pdf(
            pdf_path, 
            pages="all", 
            flavor=table_flavor,
            strip_text='\n',
            suppress_stdout=True
        )
        print(f"[BẢNG] Đã phát hiện {len(tables)} bảng")
    except Exception as e:
        print(f"[LỖI] Không thể trích xuất bảng: {str(e)}")
        tables = []

    table_bboxes: List[Tuple[int, Tuple[float, float, float, float]]] = []
    table_chunks: List[str] = []
    
    for i, table in enumerate(tables):
        page_num = table.page
        print(f"[BẢNG] Xử lý bảng {i+1} trên trang {page_num}")
        if hasattr(table, '_bbox'):
            table_bboxes.append((page_num, table._bbox))
        try:
            markdown_table = table.df.to_markdown(index=False)
            table_chunks.append(f"\n\n[BẢNG {i+1} - TRANG {page_num}]\n{markdown_table}\n[/BẢNG]\n\n")
        except Exception as e:
            print(f"[LỖI] Xử lý bảng {i+1} thất bại: {str(e)}")
    print("[VĂN BẢN] Đang trích xuất nội dung văn bản...")
    laparams = LAParams(
        line_margin=0.5,
        word_margin=0.1,
        boxes_flow=0.7
    )
    
    all_text_content = []
    page_elements: List[Dict[str, Any]] = []
    
    for page_num, page_layout in enumerate(extract_pages(pdf_path, laparams=laparams), start=1):
        page_width = page_layout.width
        page_height = page_layout.height
        page_texts = []
        header_threshold = page_height * 0.85
        footer_threshold = page_height * 0.15
        
        for element in page_layout:
            if isinstance(element, LTTextBox):
                x0, y0, x1, y1 = element.bbox
                element_text = element.get_text().strip()
                in_table = any(
                    page_num == tb_page and 
                    is_bbox_overlap((x0, y0, x1, y1), tb_bbox, tolerance=5)
                    for tb_page, tb_bbox in table_bboxes
                )
                
                if not element_text or in_table or y1 > header_threshold or y0 < footer_threshold:
                    continue
                

                cleaned_text = clean_vietnamese_text(element_text)
                if cleaned_text:
                    page_texts.append({
                        'text': cleaned_text,
                        'bbox': (x0, y0, x1, y1),
                        'font_size': get_avg_font_size(element)
                    })
        

        page_texts.sort(key=lambda x: (-x['bbox'][3], x['bbox'][0])) 
        

        merged_text = merge_text_elements(page_texts)
        if merged_text:
            all_text_content.append(merged_text)
            print(f"[VĂN BẢN] Trang {page_num}: Trích xuất {len(merged_text.split())} từ")

    print("[TỔNG HỢP] Kết hợp văn bản và bảng biểu...")
    text_content = "\n\n".join(all_text_content)
    table_content = "".join(table_chunks)
    combined_content = table_content + "\n\n" + text_content

    print(f"[CHIA NHỎ] Đang chia văn bản (kích thước: {chunk_size}, chồng lấp: {chunk_overlap})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
    )
    
    final_chunks = text_splitter.split_text(combined_content)
    print(f"[HOÀN TẤT] Đã tạo {len(final_chunks)} đoạn văn bản")
    return final_chunks

def clean_vietnamese_text(text: str) -> str:
    """
    Chuẩn hóa văn bản tiếng Việt:
    - Sửa lỗi khoảng trắng quanh dấu câu
    - Chuẩn hóa dấu câu tiếng Việt
    - Xử lý từ viết tắt thông dụng
    """

    text = re.sub(r'\s+([.,!?;:)\]])', r'\1', text)
    

    text = re.sub(r'([.,!?;:([{])(\w)', r'\1 \2', text)

    text = re.sub(r'(\w)-\s+(\w)', r'\1-\2', text)
    

    abbreviations = {
        r'\bs\s*\.\s*k\s*\.': 's.k.',
        r'\bT\s*\.\s*P\s*\.': 'T.P.',
        r'\bP\s*\.\s*G\s*\.\s*S\s*\.': 'P.G.S.',
        r'\bTS\s*\.': 'TS.',
        r'\bTH\s*\.\s*S\s*\.': 'TH.S.',
        r'\bMr\s*\.': 'Mr.',
        r'\bMrs\s*\.': 'Mrs.',
    }
    
    for pattern, replacement in abbreviations.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    text = re.sub(r"''", '"', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def get_avg_font_size(element: LTTextContainer) -> float:
    """Tính kích thước font trung bình của phần tử"""
    sizes = []
    for text_line in element:
        if hasattr(text_line, "chars"):
            for char in text_line.chars:
                if isinstance(char, LTChar):
                    sizes.append(char.size)
    return np.mean(sizes) if sizes else 12.0

def is_bbox_overlap(
    bbox1: Tuple[float, float, float, float], 
    bbox2: Tuple[float, float, float, float],
    tolerance: float = 5.0
) -> bool:
    """Kiểm tra hai khung chữ nhật có chồng lấp nhau không"""
    x0_1, y0_1, x1_1, y1_1 = bbox1
    x0_2, y0_2, x1_2, y1_2 = bbox2
    

    x_overlap = max(0, min(x1_1, x1_2) - max(x0_1, x0_2))

    y_overlap = max(0, min(y1_1, y1_2) - max(y0_1, y0_2))

    area1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    overlap_area = x_overlap * y_overlap
    
    return overlap_area > (area1 * 0.3) or (x_overlap > tolerance and y_overlap > tolerance)

def merge_text_elements(elements: List[Dict], line_threshold: float = 1.5) -> str:
    """
    Hợp nhất các phần tử văn bản thành đoạn dựa trên:
    - Khoảng cách dòng
    - Kích thước font chữ
    - Căn chỉnh văn bản
    """
    if not elements:
        return ""
    
    paragraphs = []
    current_para = []
    prev_y1 = elements[0]['bbox'][3]
    prev_font_size = elements[0]['font_size']
    
    for elem in elements:
        _, y0, _, y1 = elem['bbox']
        current_font_size = elem['font_size']

        line_spacing = prev_y1 - y1  # Khoảng cách so với dòng trước

        if (line_spacing > prev_font_size * line_threshold or 
            abs(current_font_size - prev_font_size) > prev_font_size * 0.2):
            if current_para:
                paragraphs.append(" ".join(current_para))
                current_para = []
        
        current_para.append(elem['text'])
        prev_y1 = y1
        prev_font_size = current_font_size
    
    if current_para:
        paragraphs.append(" ".join(current_para))
    
    return "\n\n".join(paragraphs)