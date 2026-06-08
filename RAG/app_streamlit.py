import sys
from pathlib import Path
import streamlit as st
import os
import importlib
from dotenv import load_dotenv
from openai import OpenAI

# Tải cấu hình môi trường
load_dotenv()

# Thêm thư mục gốc vào sys.path để python nhận dạng package 'src'
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Reload các module cục bộ để tránh lỗi lưu cache của Streamlit (khi fileWatcherType = "none")
import RAG.rag_pipeline
import RAG.styles
importlib.reload(RAG.rag_pipeline)
importlib.reload(RAG.styles)

from RAG.rag_pipeline import retrieve, reorder_for_llm, format_context, SYSTEM_PROMPT, TEMPERATURE, TOP_P
from RAG.styles import apply_custom_css, render_status_dashboard, render_source_card_html

# Thiết lập cấu hình trang với phong cách chuyên nghiệp
st.set_page_config(
    page_title="Thẩm Phán Số - Trợ Lý RAG Cao Cấp",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Áp dụng giao diện tùy biến tối cao cấp từ file styles.py độc lập
apply_custom_css()

# Khởi tạo OpenAI Client
api_key = os.getenv("OPENAI_API_KEY", "")
openai_client = None
if api_key and "sk-xxx" not in api_key:
    openai_client = OpenAI(api_key=api_key)


def rewrite_query_with_history(query: str, history: list, model: str = "gpt-4o-mini") -> str:
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
            model=model,
            messages=messages,
            temperature=0.1
        )
        rewritten = response.choices[0].message.content.strip()
        return rewritten if rewritten else query
    except Exception:
        return query


def export_chat_history() -> str:
    """
    Xuất toàn bộ lịch sử trò chuyện hiện tại dưới định dạng Markdown chất lượng cao.
    """
    if not st.session_state.messages:
        return ""
    
    md_content = "# ⚖️ Lịch sử Trò chuyện - Thẩm Phán Số RAG Chatbot\n"
    md_content += "Hệ thống RAG Tra cứu Luật Phòng chống Ma túy và Tin tức Xã hội\n"
    md_content += "------------------------------------------------------------\n\n"
    
    for msg in st.session_state.messages:
        role = "👤 Người dùng" if msg["role"] == "user" else "🤖 Trợ lý Pháp luật"
        md_content += f"### {role}\n{msg['content']}\n\n"
        if "sources" in msg and msg["sources"]:
            md_content += "**Tài liệu tham khảo đã dùng:**\n"
            for idx, src in enumerate(msg["sources"], 1):
                source_name = src.get("metadata", {}).get("source", "Nguồn")
                doc_type = src.get("metadata", {}).get("doc_type", src.get("metadata", {}).get("type", "Chưa rõ")).upper()
                md_content += f"- **[{doc_type}]** {source_name}\n"
            md_content += "\n"
        md_content += "---\n\n"
    return md_content


# Giao diện chính Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/scale-of-justice.png", width=80)
    st.markdown("### ⚖️ Thẩm Phán Số V2")
    st.markdown("Trợ lý pháp luật phòng chống ma túy và tra cứu tin tức xã hội thông minh.")
    st.markdown("---")
    
    # Bảng chỉ số hệ thống trực quan (Imported từ styles.py)
    st.markdown("#### 📡 Trạng thái Hệ thống")
    st.markdown(render_status_dashboard(), unsafe_allow_html=True)

    # Cấu hình bộ lọc & tham số tìm kiếm
    st.markdown("#### ⚙️ Cấu hình RAG")
    
    # A. Bộ lọc nguồn tài liệu
    doc_filter_label = st.selectbox(
        "📁 Bộ lọc nguồn tài liệu",
        ["Tất cả tài liệu", "Chỉ Luật Ma Túy", "Chỉ Tin Tức Nghệ Sĩ"]
    )
    doc_filter_map = {
        "Tất cả tài liệu": "all",
        "Chỉ Luật Ma Túy": "legal",
        "Chỉ Tin Tức Nghệ Sĩ": "news"
    }
    doc_type_filter_val = doc_filter_map[doc_filter_label]

    # B. Lựa chọn mô hình LLM
    selected_llm = st.selectbox(
        "🧠 Mô hình ngôn ngữ (LLM)",
        ["gpt-4o-mini", "gpt-4o"]
    )

    # C. Bật/tắt so sánh Rerank
    show_rerank_inspector = st.toggle("🔬 Hiển thị so sánh Rerank", value=True)

    st.markdown("---")
    
    # D. Các tham số truy xuất bổ sung
    score_threshold = st.slider("Ngưỡng điểm chất lượng (Threshold)", 0.1, 0.9, 0.3, 0.05)
    top_k = st.slider("Số lượng tài liệu tham chiếu (Top K)", 1, 10, 5)
    use_reranking = st.toggle("Sử dụng Reranking", value=True)
    
    st.markdown("---")
    
    # Nút xóa lịch sử chat
    if st.button("🧹 Xóa lịch sử trò chuyện", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # Nút tải lịch sử chat
    if "messages" in st.session_state and st.session_state.messages:
        chat_markdown = export_chat_history()
        st.download_button(
            label="💾 Tải lịch sử trò chuyện (.md)",
            data=chat_markdown,
            file_name="tham_phan_so_chat_history.md",
            mime="text/markdown",
            use_container_width=True
        )

# Khởi tạo session state cho lịch sử chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Tiêu đề giao diện chính
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
                    doc_type = src.get("metadata", {}).get("doc_type", src.get("metadata", {}).get("type", "Chưa rõ"))
                    content_preview = src.get("content", "")
                    
                    # Vẽ card tài liệu thông qua template được định nghĩa trong styles.py
                    col.markdown(render_source_card_html(source_name, doc_type, content_preview), unsafe_allow_html=True)

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
            rewritten_query = rewrite_query_with_history(prompt, chat_history, model=selected_llm)
            if rewritten_query != prompt:
                st.write(f"👉 Câu hỏi tìm kiếm độc lập: *\"{rewritten_query}\"*")
            
            # Bước B: Gọi Pipeline truy xuất cục bộ (bỏ qua PageIndex và sử dụng mxbai-xs)
            st.write("🔍 Đang tìm kiếm cơ sở dữ liệu (Hybrid Search)...")
            if show_rerank_inspector:
                res_data = retrieve(
                    query=rewritten_query, 
                    top_k=top_k, 
                    score_threshold=score_threshold, 
                    use_reranking=use_reranking,
                    doc_type_filter=doc_type_filter_val,
                    return_comparison=True
                )
                chunks = res_data["final"]
                before_chunks = res_data["before_rerank"]
            else:
                chunks = retrieve(
                    query=rewritten_query, 
                    top_k=top_k, 
                    score_threshold=score_threshold, 
                    use_reranking=use_reranking,
                    doc_type_filter=doc_type_filter_val,
                    return_comparison=False
                )
                chunks = chunks if isinstance(chunks, list) else []
                before_chunks = []
                
            st.write(f"✓ Đã tìm thấy {len(chunks)} phân đoạn tài liệu phù hợp (Đã áp dụng bộ lọc: **{doc_filter_label}**).")
            
            # Bước C: Sắp xếp lại tài liệu để tránh lost in the middle
            st.write("📑 Sắp xếp lại thứ tự tài liệu tối ưu cho LLM...")
            reordered_chunks = reorder_for_llm(chunks)
            context = format_context(reordered_chunks)
            
            status_box.update(label="Truy xuất hoàn tất! Đang sinh câu trả lời...", state="complete", expanded=False)

        # Bước D: Gọi LLM sinh câu trả lời có trích dẫn
        user_message = f"Ngữ cảnh tham chiếu:\n{context}\n\n---\n\nCâu hỏi: {prompt}"
        
        try:
            # Tạo hiệu ứng gõ chữ (streaming)
            full_response = ""
            response_stream = openai_client.chat.completions.create(
                model=selected_llm,
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
            
            # 🔬 Hiển thị so sánh thay đổi thứ hạng sau Reranking
            if show_rerank_inspector and before_chunks and chunks:
                with st.expander("🔬 So sánh thay đổi thứ hạng sau Reranking (Cross-Encoder Cục bộ)"):
                    st.markdown("""
                    Bảng dưới đây so sánh sự thay đổi thứ hạng của các phân đoạn tài liệu trước và sau khi được tính toán chấm điểm lại bằng mô hình **`mixedbread-ai/mxbai-rerank-xsmall-v1`**:
                    """)
                    
                    comp_data = []
                    # Tạo bản đồ ánh xạ thứ hạng kết quả cuối cùng
                    final_rank_map = {item["content"]: idx + 1 for idx, item in enumerate(chunks)}
                    
                    for idx, item in enumerate(before_chunks, 1):
                        content_preview = item["content"][:80].replace("\n", " ") + "..."
                        source_name = item.get("metadata", {}).get("source", "Nguồn")
                        final_rank = final_rank_map.get(item["content"], "Bị loại (> Top K)")
                        
                        rank_change = ""
                        if isinstance(final_rank, int):
                            diff = idx - final_rank
                            if diff > 0:
                                rank_change = f"⬆️ Tăng {diff} bậc"
                            elif diff < 0:
                                rank_change = f"⬇️ Giảm {abs(diff)} bậc"
                            else:
                                rank_change = "➡️ Giữ nguyên"
                        else:
                            rank_change = "❌ Không lọt Top K"
                            
                        comp_data.append({
                            "Thứ hạng ban đầu (RRF)": f"#{idx}",
                            "Nguồn tài liệu": source_name,
                            "Nội dung tóm tắt": content_preview,
                            "Thứ hạng sau Rerank": f"#{final_rank}" if isinstance(final_rank, int) else final_rank,
                            "Thay đổi": rank_change
                        })
                    st.table(comp_data)
            
            # Hiển thị trực quan tài liệu tham chiếu dưới câu trả lời
            if chunks:
                with st.expander("🔍 Xem các tài liệu tham chiếu đã dùng"):
                    cols = st.columns(min(len(chunks), 3))
                    for idx, src in enumerate(chunks):
                        col = cols[idx % 3]
                        source_name = src.get("metadata", {}).get("source", "Nguồn tham khảo")
                        doc_type = src.get("metadata", {}).get("doc_type", src.get("metadata", {}).get("type", "Chưa rõ"))
                        content_preview = src.get("content", "")
                        
                        # Vẽ card tài liệu thông qua template được định nghĩa trong styles.py
                        col.markdown(render_source_card_html(source_name, doc_type, content_preview), unsafe_allow_html=True)
            
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
            
            # Thực hiện cập nhật lại giao diện để nút "Tải lịch sử" xuất hiện ngay lập tức
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Có lỗi xảy ra trong quá trình gọi mô hình sinh câu trả lời: {e}")
