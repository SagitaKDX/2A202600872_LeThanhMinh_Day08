"""
Task 10 — Generation Có Citation.

Hướng dẫn:
    1. Chọn top_k, top_p phù hợp (giải thích lý do)
    2. Sắp xếp lại chunks sau reranking để tránh "lost in the middle"
    3. Inject context vào prompt
    4. Yêu cầu LLM trả lời có citation
    5. Nếu không đủ evidence → "I cannot verify this information"
"""

import os
from dotenv import load_dotenv
import sys
from pathlib import Path

load_dotenv()

# Thêm thư mục gốc vào sys.path để python nhận dạng package 'src'
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.task9_retrieval_pipeline import retrieve
except ImportError:
    from task9_retrieval_pipeline import retrieve


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn
# =============================================================================

# top_k: Số chunks đưa vào context
# Chọn 5 vì: Đủ thông tin làm minh chứng mà không làm ngữ cảnh quá dài gây hiện tượng "lost in the middle"
TOP_K = 5

# top_p (nucleus sampling): Xác suất tích luỹ cho token generation
# Chọn 0.9 vì: Tạo sự đa dạng ngôn từ hợp lý nhưng không vượt quá tầm kiểm soát hoặc ngẫu nhiên quá mức
TOP_P = 0.9

# temperature: Độ ngẫu nhiên của câu trả lời
# Chọn 0.3 vì: Hệ thống RAG cần sự chính xác tuyệt đối theo tài liệu gốc (factual), hạn chế tối đa sự sáng tạo
TEMPERATURE = 0.3


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """Answer the following question comprehensively in Vietnamese.
For every statement of fact or claim, immediately insert a citation in brackets
linking to the specific source (e.g., [Luật Phòng chống ma tuý 2021, Điều 3]
or [VnExpress, 2024]).

If the information is not explicitly stated in the provided context or knowledge
base, state 'Tôi không thể xác minh thông tin này từ nguồn hiện có' rather than
guessing.

Rules:
- Only use information from the provided context
- Every factual claim MUST have a citation
- If context is insufficient, say so clearly
- Structure your answer with clear paragraphs"""


# =============================================================================
# DOCUMENT REORDERING (tránh lost in the middle)
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp lại các phân đoạn để tránh hiệu ứng "lost in the middle" (thất lạc thông tin ở giữa).

    Mô hình ngôn ngữ lớn (LLM) có xu hướng ghi nhớ tốt thông tin nằm ở ĐẦU và CUỐI của prompt ngữ cảnh,
    và dễ bỏ sót thông tin ở GIỮA.
    Chiến lược: Đặt phân đoạn quan trọng nhất ở đầu, phân đoạn quan trọng thứ hai ở cuối, 
    và phân đoạn ít quan trọng nhất ở chính giữa.

    Thứ tự đầu vào (theo độ tương đồng):  [1, 2, 3, 4, 5]
    Thứ tự đầu ra sau sắp xếp:             [1, 3, 5, 4, 2]

    Tham số:
        chunks: Danh sách các phân đoạn đã được sắp xếp giảm dần theo điểm số độ tương đồng.

    Trả về:
        Danh sách các phân đoạn đã được sắp xếp lại thứ tự tối ưu.
    """
    if len(chunks) <= 2:
        return chunks

    reordered = [None] * len(chunks)
    left = 0
    right = len(chunks) - 1

    for i, chunk in enumerate(chunks):
        if i % 2 == 0:
            reordered[left] = chunk
            left += 1
        else:
            reordered[right] = chunk
            right -= 1

    return reordered


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

def format_context(chunks: list[dict]) -> str:
    """
    Định dạng danh sách các phân đoạn thành một chuỗi văn bản ngữ cảnh duy nhất.
    Mỗi phân đoạn sẽ đi kèm thông tin nguồn rõ ràng để mô hình LLM có căn cứ thực hiện trích dẫn (cite).

    Tham số:
        chunks: Danh sách các phân đoạn cần định dạng.

    Trả về:
        Chuỗi ngữ cảnh đã định dạng.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("metadata", {}).get("source", f"Nguồn_{i}")
        doc_type = chunk.get("metadata", {}).get("doc_type", "Chưa rõ")
        context_parts.append(
            f"[Tài liệu {i} | Nguồn: {source} | Loại: {doc_type}]\n"
            f"{chunk['content']}\n"
        )
    return "\n---\n".join(context_parts)


# =============================================================================
# GENERATION
# =============================================================================

def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """
    Quy trình RAG hoàn chỉnh từ đầu đến cuối (End-to-End) có trích dẫn nguồn.

    Quy trình:
        1. Tìm kiếm và truy xuất các phân đoạn liên quan (Retrieve)
        2. Sắp xếp lại thứ tự phân đoạn để tối ưu hóa sự tập trung của LLM (Reorder)
        3. Định dạng ngữ cảnh kèm theo nhãn nguồn rõ ràng (Format Context)
        4. Xây dựng prompt tích hợp ngữ cảnh và câu hỏi của người dùng (Build Prompt)
        5. Gọi API của mô hình ngôn ngữ lớn (Call LLM)
        6. Trả về câu trả lời hoàn thiện cùng thông tin các nguồn tài liệu đã sử dụng

    Tham số:
        query: Câu hỏi của người dùng.
        top_k: Số lượng tài liệu tham chiếu tối đa.

    Trả về:
        Từ điển chứa:
        {
            'answer': str,           # Câu trả lời hoàn chỉnh kèm trích dẫn nguồn
            'sources': list[dict],   # Danh sách các phân đoạn tài liệu đã sử dụng làm căn cứ
            'retrieval_source': str  # Nguồn truy xuất chính ('hybrid' hoặc 'pageindex')
        }
    """
    # Bước 1: Tìm kiếm các tài liệu liên quan
    chunks = retrieve(query, top_k=top_k)

    # Bước 2: Sắp xếp lại tài liệu để tránh hiện tượng lost in the middle
    reordered = reorder_for_llm(chunks)

    # Bước 3: Định dạng ngữ cảnh cho mô hình ngôn ngữ
    context = format_context(reordered)

    # Bước 4: Xây dựng nội dung tin nhắn gửi tới LLM
    user_message = f"Ngữ cảnh tham chiếu:\n{context}\n\n---\n\nCâu hỏi: {query}"

    # Bước 5: Gọi OpenAI API (sử dụng gpt-4o-mini để tối ưu hóa tốc độ và chi phí)
    from openai import OpenAI
    
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or "sk-xxx" in api_key or not api_key.strip():
        raise ValueError("Thiếu hoặc khóa OPENAI_API_KEY không hợp lệ trong tệp .env. Vui lòng cấu hình OpenAI API Key.")

    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )

    answer = response.choices[0].message.content

    # Bước 6: Lấy thông tin nguồn dữ liệu chính
    retrieval_source = "none"
    if chunks:
        retrieval_source = chunks[0].get("source", "hybrid")

    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": retrieval_source
    }


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý theo pháp luật Việt Nam?",
        "Những nghệ sĩ nào đã bị bắt vì liên quan tới ma tuý?",
        "Quy trình cai nghiện bắt buộc theo Luật Phòng chống ma tuý 2021?",
    ]

    for q in test_queries:
        print(f"\n{'='*70}")
        print(f"Hỏi: {q}")
        print("=" * 70)
        try:
            result = generate_with_citation(q)
            print(f"\nTrả lời: {result['answer']}")
            print(f"\n[Nguồn: {len(result['sources'])} phân đoạn tham chiếu | Phương thức: {result['retrieval_source']}]")
        except Exception as e:
            print(f"❌ Có lỗi xảy ra trong quá trình sinh câu trả lời: {e}")
