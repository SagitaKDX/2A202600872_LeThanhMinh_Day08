import sys
from pathlib import Path
import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI

# Tải cấu hình môi trường
load_dotenv()

# Thêm thư mục gốc vào sys.path để python nhận dạng package 'src'
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.task9_retrieval_pipeline import retrieve
from src.task10_generation import reorder_for_llm, format_context, SYSTEM_PROMPT, TEMPERATURE, TOP_P

# Thiết lập cấu hình trang với phong cách chuyên nghiệp
st.set_page_config(
    page_title="Thẩm Phán Số - Trợ Lý Pháp Luật RAG",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS cho phong cách giao diện tối hiện đại, cao cấp
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #f1f5f9;
    }
    
    .app-title {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8 0%, #a78bfa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        margin-bottom: 0.1rem;
        text-align: center;
    }
    
    .app-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        color: #94a3b8;
        text-align: center;
        margin-bottom: 1.8rem;
    }

    section[data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid #334155;
    }

    .source-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s;
    }
    
    .source-card:hover {
        transform: translateY(-2px);
        border-color: #38bdf8;
    }

    .source-header {
        font-size: 0.9rem;
        font-weight: bold;
        color: #38bdf8;
        margin-bottom: 5px;
    }

    .source-body {
        font-size: 0.85rem;
        color: #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# Khởi tạo OpenAI Client
api_key = os.getenv("OPENAI_API_KEY", "")
openai_client = None
if api_key and "sk-xxx" not in api_key:
    openai_client = OpenAI(api_key=api_key)


def rewrite_query_with_history(query: str, history: list) -> str:
    """
    Sử dụng LLM để viết lại câu hỏi dựa trên lịch sử trò chuyện.
    Giúp tìm kiếm ngữ nghĩa chính xác hơn với các câu hỏi tiếp nối (follow-up).
    """
    if not history or not openai_client:
        return query

    # Tạo prompt hướng dẫn viết lại câu hỏi
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Given the chat history and a follow-up question, rewrite it as a standalone search query in Vietnamese. Do not add any conversational text, only return the rewritten query."},
    ]
    # Thêm tối đa 4 lượt hội thoại gần nhất để tránh tràn ngữ cảnh
    for msg in history[-4:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    
    messages.append({"role": "user", "content": f"Rewrite this follow-up question into a standalone query: {query}"})

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.1
        )
        rewritten = response.choices[0].message.content.strip()
        # Nếu LLM không thể viết lại, trả về câu hỏi gốc
        return rewritten if rewritten else query
    except Exception:
        return query


# Giao diện chính Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/scale-of-justice.png", width=80)
    st.markdown("### ⚖️ Thẩm Phán Số V2")
    st.markdown("Trợ lý pháp luật phòng chống ma túy và tra cứu tin tức xã hội thông minh.")
    st.markdown("---")
    
    # Cấu hình tham số tìm kiếm
    st.markdown("#### ⚙️ Cấu hình Hệ thống")
    score_threshold = st.slider("Ngưỡng điểm chất lượng (Threshold)", 0.1, 0.9, 0.3, 0.05)
    top_k = st.slider("Số lượng tài liệu tham chiếu (Top K)", 1, 10, 5)
    use_reranking = st.toggle("Sử dụng Reranking (Cross-Encoder)", value=True)
    
    st.markdown("---")
    if st.button("🧹 Xóa lịch sử trò chuyện"):
        st.session_state.messages = []
        st.rerun()

# Khởi tạo session state cho lịch sử chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Tiêu đề giao diện
st.markdown("<div class='app-title'>⚖️ THẨM PHÁN SỐ</div>", unsafe_allow_html=True)
st.markdown("<div class='app-subtitle'>Hệ thống RAG Tra cứu Luật Phòng chống Ma túy và Tin tức Nghệ sĩ</div>", unsafe_allow_html=True)

# Hiển thị lịch sử trò chuyện
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Nếu có nguồn tài liệu đi kèm thì hiển thị
        if "sources" in msg and msg["sources"]:
            with st.expander("🔍 Xem các tài liệu tham chiếu đã dùng"):
                cols = st.columns(min(len(msg["sources"]), 3))
                for idx, src in enumerate(msg["sources"]):
                    col = cols[idx % 3]
                    source_name = src.get("metadata", {}).get("source", "Nguồn tham khảo")
                    doc_type = src.get("metadata", {}).get("doc_type", "Chưa rõ")
                    content_preview = src.get("content", "")
                    
                    col.markdown(f"""
                    <div class='source-card'>
                        <div class='source-header'>📄 {source_name} [{doc_type.upper()}]</div>
                        <div class='source-body'>{content_preview[:200]}...</div>
                    </div>
                    """, unsafe_allow_html=True)

# Nhận tin nhắn mới từ người dùng
if prompt := st.chat_input("Hãy đặt câu hỏi về luật ma túy hoặc thông tin nghệ sĩ tại đây..."):
    # 1. Hiển thị câu hỏi của người dùng
    st.chat_message("user").markdown(prompt)
    
    # 2. Xử lý câu trả lời từ hệ thống RAG
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # Tạo trạng thái loading đẹp mắt các bước xử lý
        with st.status("Đang truy xuất và phân tích dữ liệu...", expanded=True) as status_box:
            # Lịch sử hội thoại dạng OpenAI format
            chat_history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
            
            # Bước A: Viết lại câu hỏi nếu là câu hỏi tiếp nối
            st.write("🔄 Đang phân tích ngữ cảnh hội thoại...")
            rewritten_query = rewrite_query_with_history(prompt, chat_history)
            if rewritten_query != prompt:
                st.write(f"👉 Câu hỏi tìm kiếm độc lập: *\"{rewritten_query}\"*")
            
            # Bước B: Gọi Pipeline truy xuất (Task 9)
            st.write("🔍 Đang tìm kiếm cơ sở dữ liệu (Hybrid Search)...")
            chunks = retrieve(
                query=rewritten_query, 
                top_k=top_k, 
                score_threshold=score_threshold, 
                use_reranking=use_reranking
            )
            st.write(f"✓ Đã tìm thấy {len(chunks)} phân đoạn tài liệu phù hợp.")
            
            # Xác định nguồn gốc retrieval
            retrieval_source = chunks[0].get("source", "hybrid") if chunks else "none"
            st.write(f"📥 Phương thức truy xuất chính: **{retrieval_source.upper()}**")
            
            # Bước C: Sắp xếp lại tài liệu để tránh lost in the middle
            st.write("📑 Sắp xếp lại thứ tự tài liệu tối ưu cho LLM...")
            reordered_chunks = reorder_for_llm(chunks)
            context = format_context(reordered_chunks)
            
            status_box.update(label="Truy xuất hoàn tất! Đang sinh câu trả lời...", state="complete", expanded=False)

        # Bước D: Gọi LLM sinh câu trả lời có trích dẫn (Task 10)
        user_message = f"Ngữ cảnh tham chiếu:\n{context}\n\n---\n\nCâu hỏi: {prompt}"
        
        try:
            # Tạo hiệu ứng gõ chữ (streaming)
            full_response = ""
            response_stream = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                temperature=TEMPERATURE,
                top_p=TOP_P,
                stream=True
            )
            for chunk in response_stream:
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            
            # Hiển thị trực quan tài liệu tham chiếu dưới câu trả lời
            if chunks:
                with st.expander("🔍 Xem các tài liệu tham chiếu đã dùng"):
                    cols = st.columns(min(len(chunks), 3))
                    for idx, src in enumerate(chunks):
                        col = cols[idx % 3]
                        source_name = src.get("metadata", {}).get("source", "Nguồn tham khảo")
                        doc_type = src.get("metadata", {}).get("doc_type", "Chưa rõ")
                        content_preview = src.get("content", "")
                        
                        col.markdown(f"""
                        <div class='source-card'>
                            <div class='source-header'>📄 {source_name} [{doc_type.upper()}]</div>
                            <div class='source-body'>{content_preview[:200]}...</div>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Lưu lại vào lịch sử session state
            st.session_state.messages.append({
                "role": "user", 
                "content": prompt
            })
            st.session_state.messages.append({
                "role": "assistant", 
                "content": full_response,
                "sources": chunks
            })
            
        except Exception as e:
            st.error(f"❌ Có lỗi xảy ra trong quá trình gọi mô hình sinh câu trả lời: {e}")
