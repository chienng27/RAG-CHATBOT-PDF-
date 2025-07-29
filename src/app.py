
from qdrant_client import QdrantClient
from embedding import generate_query_embedding, generate_embeddings
from pdf_processing import extract_and_process_pdf
from retrieval import rerank_with_cross_encoder
from vectordb import upload_to_qdrant, delete_collection
from rewrite import enhance_query
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
import os
import numpy as np
import re
import asyncio
import json
import pandas as pd
from io import StringIO

from qdrant_client import QdrantClient
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

qdrant_host = "127.0.0.1"
qdrant_port = 6333
temperature = 0.1

@st.cache_resource
def init_llm(api_key):
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        google_api_key=api_key,
        temperature=temperature
    )

@st.cache_resource
def init_qdrant():
    return QdrantClient(host=qdrant_host, port=qdrant_port)

prompt_template = ChatPromptTemplate.from_template(
"""
Bạn là một RAG AI AGENT thông minh tên là QUIZ_CHAT .
Nếu có thông tin ngữ cảnh bên dưới, hãy sử dụng **chính xác** nội dung trong đó để trả lời câu hỏi.
Nếu ngữ cảnh chứa bảng, hãy phân tích bảng để trả lời câu hỏi dựa trên các hàng và cột tương ứng.
Tên cột và hàng đã được cung cấp trong bảng. Hãy sử dụng chúng để trả lời chính xác.

Lịch sử hội thoại trước đó:
{history}

Ngữ cảnh từ tài liệu (nếu có):
{context}

Câu hỏi hiện tại:
{question}

Yêu cầu:
- Nếu ngữ cảnh là bảng, hãy trả lời dựa trên dữ liệu trong bảng. Trích xuất thông tin từ các hàng và cột liên quan.
- Trả lời đúng trọng tâm, ngắn gọn và bằng tiếng Việt.
- Tuyệt đối không bịa thông tin nếu dùng ngữ cảnh.
- Nếu không có ngữ cảnh, hãy trả lời tự nhiên như một trợ lý AI thân thiện.

Câu trả lời:
"""
)


prompt_direct_template = ChatPromptTemplate.from_template(
"""Dựa vào lịch sử hội thoại dưới đây và kiến thức của bạn, hãy trả lời câu hỏi.
Nếu bạn không biết câu trả lời, hãy thành thật nói 'Tôi không biết'.

Lịch sử hội thoại:
{history}

Câu hỏi:
{question}

Yêu cầu:
- Trả lời ngắn gọn, chính xác bằng tiếng Việt
- Chỉ trả lời 'Tôi không biết' khi thực sự không có thông tin
- Không tự bịa câu trả lời

Trả lời:
"""
)

def collection_exists(collection_name, client):
    try:
        client.get_collection(collection_name)
        return True
    except Exception:
        return False

def sanitize_filename(filename):
    """Chuyển tên file thành tên collection hợp lệ"""
    name = os.path.splitext(filename)[0]
    name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    return name[:50]

st.title("📚 QUIZ_CHAT 📚 ")
st.caption("By NTChien . Được sử dụng trong tra cứu và tổng hợp tài liệu")

st.sidebar.header("Cấu hình API Key")
api_key = st.sidebar.text_input("Nhập Google API Key:", type="password")

if not api_key:
    st.warning("Vui lòng nhập Google API Key ở sidebar để sử dụng chatbot.")
    st.stop()

client = init_qdrant()
llm = init_llm(api_key)

if st.sidebar.button("🗑️ Xóa lịch sử chat"):
    st.session_state.messages = []
    st.rerun()

# Khởi tạo phiên làm việc
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.collection_name = None
    st.session_state.uploaded_file = None
    st.session_state.processed = False

uploaded_file = st.file_uploader("Tải lên file PDF", type="pdf")

if uploaded_file and (uploaded_file != st.session_state.uploaded_file):
    st.session_state.uploaded_file = uploaded_file
    st.session_state.processed = False
    st.session_state.messages = []
    st.session_state.collection_name = None


if uploaded_file and not st.session_state.processed:
    with st.status(f"Đang xử lý {uploaded_file.name}...", expanded=True) as status:

        st.session_state.collection_name = sanitize_filename(uploaded_file.name)
        st.write(f"Tạo collection: {st.session_state.collection_name}")
        

        if collection_exists(st.session_state.collection_name, client):
            st.warning(f"Collection '{st.session_state.collection_name}' đã tồn tại")
            if delete_collection(st.session_state.collection_name):
                st.write("Đang tạo lại collection...")
        

        st.write("📄 Trích xuất nội dung PDF...")
        chunks = extract_and_process_pdf(uploaded_file ,chunk_size=3000,  chunk_overlap=300,
    table_flavor='lattice')
        
        
        st.write("🧠 Tạo embeddings...")
        embeddings = generate_embeddings(chunks, api_key)
        
        st.write("🚀 Upload dữ liệu lên Qdrant...")
        metadata_list = [{"source": uploaded_file.name, "chunk_id": i} 
                        for i in range(len(chunks))]
        
        upload_to_qdrant(
            chunks=chunks,
            embeddings=embeddings,
            collection_name=st.session_state.collection_name,
            metadata_list=metadata_list,
            client=client,
            batch_size=100
        )
        
        status.update(label=f"✅ Đã xử lý xong {uploaded_file.name}!", state="complete", expanded=False)
        st.session_state.processed = True


if not uploaded_file:
    st.info("Vui lòng tải lên file PDF để bắt đầu")
    st.stop()


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def build_history(messages):
    """Tạo chuỗi lịch sử hội thoại cho prompt."""
    history = ""
    for msg in messages[:-1]:
        role = "Người dùng" if msg["role"] == "user" else "Trợ lý"
        history += f"{role}: {msg['content']}\n"
    return history

def show_context(reranked):
    """Hiển thị ngữ cảnh đã trích xuất với font nhỏ."""
    with st.expander("🔎 Xem ngữ cảnh đã trích xuất", expanded=False):
        for idx, (text, score) in enumerate(reranked, 1):
            st.markdown(
                f"<span style='font-size:12px'><b>Đoạn {idx} (score={score:.3f}):</b><br>{text}</span><hr>",
                unsafe_allow_html=True
            )

def handle_user_query(prompt, api_key, client, llm, collection_name):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    history = build_history(st.session_state.messages)
    
    # Bước 1: Thử trả lời bằng kiến thức của LLM
    with st.spinner("💡 Đang suy nghĩ..."):
        direct_prompt = prompt_direct_template.format(
            history=history,
            question=prompt
        )
        direct_response = llm.invoke(direct_prompt)
        direct_answer = direct_response.content

    # Kiểm tra nếu LLM không biết câu trả lời
    unknow_phrases = ["không biết", "không có thông tin", "chưa có dữ liệu"]
    if any(phrase in direct_answer.lower() for phrase in unknow_phrases):
        # Bước 2: Chỉ enhanced query khi LLM không trả lời được
        with st.spinner("🔍 Đang tìm kiếm thông tin trong tài liệu..."):
            enhanced_query = enhance_query(
                query=prompt,
                api_key=api_key,
                max_rewrites=1
            )
            enhanced_query_embedding = generate_query_embedding(enhanced_query, api_key).tolist()
            
            hits = client.search(
                collection_name=collection_name,
                query_vector=enhanced_query_embedding,
                limit=20,
                with_payload=True
            )
            candidate_texts = [hit.payload['text'] for hit in hits if hit.payload.get('text')]
            reranked = rerank_with_cross_encoder(
                query=prompt,
                candidate_texts=candidate_texts,
                k=3
            )
            context = "\n\n".join([text for text, score in reranked])
            
            # Tạo prompt với ngữ cảnh
            formatted_prompt = prompt_template.format(
                history=history,
                context=context,
                question=prompt
            )

        with st.spinner("💡 Đang tạo câu trả lời từ tài liệu..."):
            response = llm.invoke(formatted_prompt)
            answer = response.content
        
        # Hiển thị ngữ cảnh
        show_context(reranked)
    else:
        # Sử dụng câu trả lời trực tiếp từ LLM
        answer = direct_answer

    # Hiển thị câu trả lời
    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
    show_context(reranked)

if prompt := st.chat_input(f"Hỏi về nội dung {uploaded_file.name}..."):
    if not st.session_state.processed:
        st.warning("Vui lòng đợi xử lý PDF hoàn tất")
        st.stop()
    handle_user_query(
        prompt=prompt,
        api_key=api_key,
        client=client,
        llm=llm,
        collection_name=st.session_state.collection_name
    )