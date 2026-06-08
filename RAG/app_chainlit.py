import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import chainlit as cl

# Tải cấu hình môi trường
load_dotenv()

# Thêm thư mục gốc vào sys.path để python nhận dạng package 'src'
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.task9_retrieval_pipeline import retrieve
from src.task10_generation import reorder_for_llm, format_context, SYSTEM_PROMPT, TEMPERATURE, TOP_P

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
        return rewritten if rewritten else query
    except Exception:
        return query


@cl.on_chat_start
async def start_chat():
    """
    Kích hoạt khi cuộc trò chuyện bắt đầu. Khởi tạo lịch sử và hiển thị lời chào.
    """
    # Khởi tạo lịch sử tin nhắn
    cl.user_session.set("messages", [])
    
    # Hiển thị thông điệp chào mừng đẹp mắt
    welcome_message = """⚖️ **Chào mừng bạn đến với hệ thống Thẩm Phán Số V2!**

Tôi là trợ lý trí tuệ nhân tạo chuyên tra cứu **Luật Phòng chống Ma túy Việt Nam** và các tin tức sự kiện liên quan đến xã hội.

**Bạn có thể hỏi tôi các nội dung như:**
* *Quy trình cai nghiện bắt buộc theo luật 2021 diễn ra như thế nào?*
* *Hình phạt cho tội tàng trữ trái phép chất ma túy là gì?*
* *Có thông tin gì về việc ca sĩ Chi Dân bị bắt gần đây không?*

*Hệ thống được thiết kế với cơ chế Hybrid Search, Reranking chéo và cơ chế fallback tự động thông minh.*"""
    
    await cl.Message(content=welcome_message).send()


@cl.on_message
async def handle_message(message: cl.Message):
    """
    Xử lý tin nhắn gửi tới từ người dùng. Chạy pipeline RAG và hiển thị các bước trung gian.
    """
    # Lấy lịch sử hội thoại
    chat_history = cl.user_session.get("messages")
    
    # Lấy cấu hình ngưỡng điểm số và số tài liệu (hoặc mặc định)
    score_threshold = 0.3
    top_k = 5
    
    # ==========================================
    # Bước 1: Phân tích ngữ cảnh & viết lại câu hỏi
    # ==========================================
    async with cl.Step(name="🔄 Phân tích ngữ cảnh câu hỏi...") as step:
        step.input = message.content
        rewritten_query = rewrite_query_with_history(message.content, chat_history)
        if rewritten_query != message.content:
            step.output = f"Câu hỏi gốc: \"{message.content}\"\n👉 Câu hỏi viết lại: \"{rewritten_query}\""
        else:
            step.output = "Câu hỏi độc lập, không cần viết lại."

    # ==========================================
    # Bước 2: Tìm kiếm dữ liệu song song (Hybrid)
    # ==========================================
    chunks = []
    async with cl.Step(name="🔍 Truy xuất dữ liệu (Semantic & Lexical Search)...") as step:
        step.input = rewritten_query
        try:
            # Gọi trực tiếp retrieve từ Task 9
            chunks = retrieve(
                query=rewritten_query,
                top_k=top_k,
                score_threshold=score_threshold,
                use_reranking=True
            )
            retrieval_source = chunks[0].get("source", "hybrid") if chunks else "none"
            step.output = f"Tìm thấy {len(chunks)} phân đoạn thích hợp từ nguồn: **{retrieval_source.upper()}**."
        except Exception as e:
            step.output = f"Gặp lỗi khi tìm kiếm: {e}"
            chunks = []

    # ==========================================
    # Bước 3: Sắp xếp lại tài liệu chống Lost in the middle
    # ==========================================
    context = ""
    async with cl.Step(name="📑 Sắp xếp tài liệu (Reordering)...") as step:
        if chunks:
            reordered = reorder_for_llm(chunks)
            context = format_context(reordered)
            step.output = f"Đã định dạng và sắp xếp lại {len(reordered)} tài liệu (quan trọng nhất ở đầu và cuối)."
        else:
            step.output = "Không có tài liệu tham chiếu nào được tìm thấy."

    # ==========================================
    # Bước 4: Gọi LLM sinh câu trả lời (Streaming)
    # ==========================================
    user_message = f"Ngữ cảnh tham chiếu:\n{context}\n\n---\n\nCâu hỏi: {message.content}"
    
    # Khởi tạo tin nhắn phản hồi trống
    response_msg = cl.Message(content="")
    await response_msg.send()

    try:
        # Sử dụng OpenAI stream API để in chữ dần
        stream = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            temperature=TEMPERATURE,
            top_p=TOP_P,
            stream=True
        )
        
        full_response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                full_response += token
                await response_msg.stream_token(token)
        
        # Tạo danh sách phần tử text cho tài liệu tham chiếu (sử dụng cl.Text để hiển thị đẹp ở panel bên cạnh)
        text_elements = []
        if chunks:
            for idx, src in enumerate(chunks, 1):
                source_name = src.get("metadata", {}).get("source", "Nguồn")
                doc_type = src.get("metadata", {}).get("doc_type", "Chưa rõ")
                content = src.get("content", "")
                
                text_elements.append(
                    cl.Text(
                        name=f"📄 Tài liệu {idx} ({source_name} - {doc_type})",
                        content=content,
                        display="side"
                    )
                )
        
        # Cập nhật tin nhắn phản hồi với các tài liệu đính kèm
        response_msg.elements = text_elements
        await response_msg.update()
        
        # Lưu vào lịch sử hội thoại trong Session
        chat_history.append({"role": "user", "content": message.content})
        chat_history.append({"role": "assistant", "content": full_response})
        cl.user_session.set("messages", chat_history)

    except Exception as e:
        await cl.Message(content=f"❌ Có lỗi xảy ra trong quá trình sinh câu trả lời: {e}").send()
