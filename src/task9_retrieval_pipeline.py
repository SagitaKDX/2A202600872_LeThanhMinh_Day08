"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Kết hợp semantic search + lexical search + reranking + PageIndex fallback
thành một pipeline thống nhất.

Logic:
    1. Chạy semantic_search + lexical_search song song
    2. Merge kết quả (RRF hoặc weighted fusion)
    3. Rerank
    4. Nếu top result score < threshold → fallback sang PageIndex
    5. Return top_k results
"""

import sys
from pathlib import Path

# Thêm thư mục gốc vào sys.path để python nhận dạng package 'src'
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.task5_semantic_search import semantic_search
    from src.task6_lexical_search import lexical_search
    from src.task7_reranking import rerank, rerank_rrf
    from src.task8_pageindex_vectorless import pageindex_search
except ImportError:
    from task5_semantic_search import semantic_search
    from task6_lexical_search import lexical_search
    from task7_reranking import rerank, rerank_rrf
    from task8_pageindex_vectorless import pageindex_search


# =============================================================================
# CONFIGURATION
# =============================================================================

SCORE_THRESHOLD = 0.3   # Nếu best score < threshold → fallback PageIndex
DEFAULT_TOP_K = 5
RERANK_METHOD = "cross_encoder"  # "cross_encoder" | "mmr" | "rrf"


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Quy trình truy xuất dữ liệu hoàn chỉnh kết hợp (Hybrid Retrieval Pipeline) với cơ chế fallback.

    Quy trình:
        Truy vấn
          ├→ Tìm kiếm Ngữ nghĩa (Semantic Search) → results_dense
          ├→ Tìm kiếm Từ khóa (Lexical Search)  → results_sparse
          │
          ├→ Gộp kết quả dùng thuật toán RRF (Reciprocal Rank Fusion) → merged_results
          ├→ Xếp hạng lại dùng Cross-Encoder (Rerank) → reranked_results
          │
          └→ Nếu điểm số tốt nhất < ngưỡng tối thiểu (score_threshold):
                └→ Tự động kích hoạt PageIndex Vectorless → fallback_results

    Tham số:
        query: Câu truy vấn dưới dạng văn bản.
        top_k: Số lượng kết quả cần trả về.
        score_threshold: Ngưỡng điểm tối thiểu để chấp nhận kết quả tìm kiếm kết hợp.
        use_reranking: Cho phép hoặc không cho phép áp dụng bộ xếp hạng lại (reranking).

    Trả về:
        Danh sách các kết quả dạng:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    # Bước 1: Chạy song song/tuần tự cả tìm kiếm ngữ nghĩa và tìm kiếm từ khóa
    try:
        dense_results = semantic_search(query, top_k=top_k * 2)
    except Exception as e:
        print(f"Cảnh báo: Lỗi khi tìm kiếm ngữ nghĩa: {e}")
        dense_results = []

    try:
        sparse_results = lexical_search(query, top_k=top_k * 2)
    except Exception as e:
        print(f"Cảnh báo: Lỗi khi tìm kiếm từ khóa: {e}")
        sparse_results = []

    # Bước 2: Gộp hai nguồn kết quả lại bằng thuật toán RRF (Reciprocal Rank Fusion)
    merged = rerank_rrf([dense_results, sparse_results], top_k=top_k * 2)
    for item in merged:
        item["source"] = "hybrid"

    # Bước 3: Thực hiện chấm điểm xếp hạng lại (Reranking)
    if use_reranking and merged:
        try:
            final_results = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
        except Exception as e:
            print(f"Cảnh báo: Tiến trình reranking gặp lỗi: {e}. Chuyển sang dùng kết quả thô sau gộp.")
            final_results = merged[:top_k]
    else:
        final_results = merged[:top_k]

    # Bước 4: Kiểm tra ngưỡng điểm số chất lượng để quyết định kích hoạt chế độ dự phòng PageIndex
    best_score = final_results[0]["score"] if final_results else 0.0
    if not final_results or best_score < score_threshold:
        score_val = f"{best_score:.3f}" if final_results else "0.000"
        print(f"  ⚠ Điểm số tìm kiếm kết hợp ({score_val}) nhỏ hơn ngưỡng tối thiểu ({score_threshold}). "
              f"Kích hoạt cơ chế tìm kiếm dự phòng bằng PageIndex...")
        try:
            fallback = pageindex_search(query, top_k=top_k)
            if fallback:
                for item in fallback:
                    item["source"] = "pageindex"
                return fallback
        except Exception as e:
            print(f"Cảnh báo: Tiến trình truy vấn PageIndex gặp lỗi: {e}")

    # Trả về kết quả tìm kiếm kết hợp hybrid nếu không kích hoạt hoặc PageIndex không có dữ liệu
    return final_results[:top_k]


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý",
        "Nghệ sĩ nào bị bắt vì sử dụng ma tuý năm 2024",
        "Luật phòng chống ma tuý 2021 quy định gì về cai nghiện",
    ]

    for q in test_queries:
        print(f"\nTruy vấn: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.3f}] [{r['source']}] {r['content'][:80]}...")
