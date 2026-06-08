import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Tải cấu hình môi trường
load_dotenv()

# Thêm thư mục gốc vào sys.path để python nhận dạng package 'src'
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.task5_semantic_search import semantic_search
from src.task6_lexical_search import lexical_search
from src.task7_reranking import rerank_rrf

# Cấu hình tham số cho LLM sinh câu trả lời
TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3

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

def rerank_local_cross_encoder(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """
    Rerank các ứng viên sử dụng duy nhất mô hình CrossEncoder cục bộ siêu nhẹ
    (mixedbread-ai/mxbai-rerank-xsmall-v1) trên CPU/GPU để hoạt động hoàn toàn cục bộ.
    """
    if not candidates:
        return []
    try:
        from sentence_transformers import CrossEncoder
        print("Reranking locally using CrossEncoder (mixedbread-ai/mxbai-rerank-xsmall-v1)...")
        model = CrossEncoder("mixedbread-ai/mxbai-rerank-xsmall-v1")
        pairs = [[query, c["content"]] for c in candidates]
        scores = model.predict(pairs)
        
        reranked_candidates = []
        for c, score in zip(candidates, scores):
            reranked_candidates.append({**c, "score": float(score)})
            
        reranked_candidates = sorted(reranked_candidates, key=lambda x: x["score"], reverse=True)
        return reranked_candidates[:top_k]
    except Exception as e:
        print(f"Local CrossEncoder failed: {e}. Falling back to basic score-based sort...")
        sorted_candidates = sorted(candidates, key=lambda x: x.get("score", 0.0), reverse=True)
        return sorted_candidates[:top_k]

def retrieve(
    query: str,
    top_k: int = TOP_K,
    score_threshold: float = 0.3,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Quy trình truy xuất dữ liệu cục bộ (Local Retrieval Pipeline).
    Kết hợp Semantic Search và Lexical Search (Hybrid Search), sau đó Rerank bằng
    CrossEncoder cục bộ. Bỏ qua hoàn toàn cơ chế Fallback PageIndex Cloud.
    """
    # 1. Tìm kiếm ngữ nghĩa (Semantic Search)
    try:
        dense_results = semantic_search(query, top_k=top_k * 2)
    except Exception as e:
        print(f"Cảnh báo: Lỗi khi tìm kiếm ngữ nghĩa: {e}")
        dense_results = []

    # 2. Tìm kiếm từ khóa (Lexical Search)
    try:
        sparse_results = lexical_search(query, top_k=top_k * 2)
    except Exception as e:
        print(f"Cảnh báo: Lỗi khi tìm kiếm từ khóa: {e}")
        sparse_results = []

    # 3. Trộn kết quả dùng Reciprocal Rank Fusion (RRF)
    merged = rerank_rrf([dense_results, sparse_results], top_k=top_k * 2)
    for item in merged:
        item["source"] = "hybrid"

    # 4. Reranking cục bộ (không dùng Jina API)
    if use_reranking and merged:
        final_results = rerank_local_cross_encoder(query, merged, top_k=top_k)
    else:
        final_results = merged[:top_k]

    return final_results

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp lại các phân đoạn để tránh hiệu ứng "lost in the middle" (thành phần ở giữa bị lãng quên).
    Đưa các phân đoạn có độ tương quan cao lên đầu và cuối ngữ cảnh.
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

def format_context(chunks: list[dict]) -> str:
    """
    Định dạng danh sách các phân đoạn thành một chuỗi văn bản ngữ cảnh duy nhất có kèm nhãn nguồn rõ ràng.
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
