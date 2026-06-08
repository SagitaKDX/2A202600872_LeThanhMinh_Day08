"""
Task 7 — Reranking Module.

PHÂN TÍCH QUYẾT ĐỊNH LỰA CHỌN PHƯƠNG PHÁP RERANKING:

1. Thấu Hiểu Ngữ Nghĩa Vượt Trội Đối Với Văn Bản Pháp Luật:
Các phương pháp tìm kiếm từ khóa (BM25) và tìm kiếm ngữ nghĩa cơ bản (bi-encoders) tuy phản hồi nhanh nhưng chỉ phân tích bề nổi. Chúng mã hóa câu hỏi và tài liệu thành các vector độc lập trong không gian biểu diễn. Ngược lại, Cross-Encoder xử lý đồng thời cả câu hỏi và đoạn văn bản qua mạng Transformer. Cơ chế này giúp mô hình tính toán sự chú ý (attention) chi tiết ở cấp độ token giữa câu hỏi và điều khoản pháp luật, mang lại khả năng phân tích ngữ nghĩa sâu sắc cho các thuật ngữ pháp lý tiếng Việt phức tạp.

2. Sự Phối Hợp Hoàn Hảo Với Retrieval Pipeline (Task 9):
Quy trình truy xuất trong task9_retrieval_pipeline.py thực hiện:
- Chạy song song tìm kiếm ngữ nghĩa và tìm kiếm từ khóa.
- Trộn các kết quả bằng thuật toán RRF (Reciprocal Rank Fusion).
- Chuyển danh sách đã trộn vào bước Rerank cuối cùng.
Vì chúng ta đã sử dụng RRF ở bước trước để gộp hai danh sách (dense và sparse) theo thứ hạng toán học thuần túy, việc tiếp tục áp dụng RRF cho bước Rerank là không hợp lý. Hệ thống cần một mô hình thực sự đọc và so sánh nội dung văn bản để đánh giá độ liên quan ngữ nghĩa. Cross-Encoder là phương pháp duy nhất trong Task 7 đáp ứng hoàn hảo yêu cầu này.

3. Khắc Phục Hiệu Ứng "Lost in the Middle" (Thất lạc thông tin ở giữa):
Task 10 yêu cầu hệ thống tránh hiện tượng LLM lãng quên thông tin nằm ở giữa ngữ cảnh của prompt. Cross-Encoder cung cấp cơ chế chấm điểm có độ chính xác cao nhất. Nhờ đó, hàm reorder_for_llm trong Task 10 dễ dàng nhận diện và định vị các đoạn văn bản quan trọng nhất vào vị trí đầu và cuối của prompt nhằm tối đa hóa sự chú ý của mô hình ngôn ngữ lớn.
"""

from typing import Optional
import math


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates sử dụng cross-encoder model.

    Args:
        query: Câu truy vấn
        candidates: List of {'content': str, 'score': float, 'metadata': dict}
        top_k: Số lượng kết quả sau rerank

    Returns:
        List of top_k candidates, re-scored và sorted by rerank_score descending.
    """
    import os
    import requests
    from dotenv import load_dotenv

    load_dotenv()
    jina_key = os.getenv("JINA_API_KEY")

    if not candidates:
        return []

    # 1. Thử sử dụng Jina Reranker API (nếu được cấu hình)
    if jina_key and "jina_xxx" not in jina_key and jina_key.strip():
        try:
            print("Reranking using Jina Reranker API...")
            response = requests.post(
                "https://api.jina.ai/v1/rerank",
                headers={"Authorization": f"Bearer {jina_key.strip()}"},
                json={
                    "model": "jina-reranker-v2-base-multilingual",
                    "query": query,
                    "documents": [c["content"] for c in candidates],
                    "top_n": top_k
                },
                timeout=15
            )
            response.raise_for_status()
            reranked = response.json()["results"]
            return [
                {**candidates[r["index"]], "score": float(r["relevance_score"])}
                for r in reranked
            ]
        except Exception as e:
            print(f"Jina Reranker API failed: {e}. Falling back to local cross-encoder...")

    # 2. Phương thức Fallback: Sử dụng CrossEncoder cục bộ siêu nhẹ
    try:
        from sentence_transformers import CrossEncoder
        print("Reranking using local CrossEncoder (mixedbread-ai/mxbai-rerank-xsmall-v1)...")
        #mixedbread-ai/mxbai-rerank-xsmall-v1 nhẹ (~150MB) và chạy rất nhanh trên CPU/GPU
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
        # Fallback an toàn: Sắp xếp theo score có sẵn của retrieval
        sorted_candidates = sorted(candidates, key=lambda x: x.get("score", 0.0), reverse=True)
        return sorted_candidates[:top_k]


def cosine_sim(v1: list[float], v2: list[float]) -> float:
    """Tính độ tương đồng Cosine giữa hai vector."""
    dot = sum(a*b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a*a for a in v1))
    norm2 = math.sqrt(sum(b*b for b in v2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.

    MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))

    Args:
        query_embedding: Vector embedding của query
        candidates: List of {'content': str, 'score': float, 'embedding': list, 'metadata': dict}
        top_k: Số lượng kết quả
        lambda_param: Trade-off giữa relevance (1.0) và diversity (0.0)

    Returns:
        List of top_k candidates selected by MMR.
    """
    if not candidates:
        return []

    # Kiểm tra xem các candidate có chứa trường embedding không
    for c in candidates:
        if "embedding" not in c or not c["embedding"]:
            print("Warning: Candidates missing embeddings. Cannot calculate MMR. Returning score-based sort.")
            sorted_candidates = sorted(candidates, key=lambda x: x.get("score", 0.0), reverse=True)
            return sorted_candidates[:top_k]

    selected = []
    remaining = list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        best_idx = None
        best_score = float('-inf')

        for idx in remaining:
            # Relevance to query
            relevance = cosine_sim(query_embedding, candidates[idx]["embedding"])

            # Max similarity to already selected
            max_sim_to_selected = 0.0
            if selected:
                max_sim_to_selected = max(
                    cosine_sim(candidates[idx]["embedding"], candidates[sel_idx]["embedding"])
                    for sel_idx in selected
                )

            # MMR score
            mmr_score = lambda_param * relevance - (1.0 - lambda_param) * max_sim_to_selected

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx is not None:
            selected.append(best_idx)
            remaining.remove(best_idx)
        else:
            break

    return [candidates[i] for i in selected]


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.

    RRF(d) = Σ 1 / (k + rank_r(d))

    Args:
        ranked_lists: List of ranked result lists (mỗi list từ 1 ranker)
        top_k: Số lượng kết quả cuối cùng
        k: Smoothing constant (default=60, từ paper Cormack et al. 2009)

    Returns:
        List of top_k candidates sorted by RRF score descending.
    """
    if not ranked_lists:
        return []

    rrf_scores = {}  # content -> score
    content_map = {}  # content -> full dict

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item["content"]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            # Giữ lại item đầy đủ (bao gồm metadata), ưu tiên item có điểm gốc cao nhất
            if key not in content_map or item.get("score", 0) > content_map[key].get("score", 0):
                content_map[key] = item

    # Sắp xếp giảm dần theo RRF score
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for content, score in sorted_items[:top_k]:
        item = content_map[content].copy()
        item["score"] = score
        results.append(item)

    return results


# =============================================================================
# Main rerank interface
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",  # "cross_encoder" | "mmr" | "rrf"
) -> list[dict]:
    """
    Unified reranking interface.

    Args:
        query: Câu truy vấn
        candidates: Danh sách candidates từ retrieval
        top_k: Số lượng kết quả sau rerank
        method: Phương pháp reranking

    Returns:
        List of top_k reranked candidates.
    """
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    elif method == "mmr":
        import os
        from openai import OpenAI
        from dotenv import load_dotenv

        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or "sk-xxx" in api_key or not api_key.strip():
            print("Warning: OPENAI_API_KEY missing. MMR requires query embedding. Falling back to cross-encoder.")
            return rerank_cross_encoder(query, candidates, top_k)

        # Trích xuất vector embedding của query dùng OpenAI API
        client = OpenAI(api_key=api_key)
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        )
        query_embedding = response.data[0].embedding
        return rerank_mmr(query_embedding, candidates, top_k)
    elif method == "rrf":
        # RRF cần nhiều ranked lists, nếu truyền 1 list ứng viên thì chỉ cắt lát top_k
        return candidates[:top_k]
    else:
        raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    # Test with dummy data
    dummy_candidates = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý", "score": 0.8, "metadata": {}},
        {"content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý", "score": 0.7, "metadata": {}},
        {"content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ", "score": 0.6, "metadata": {}},
    ]
    results = rerank("hình phạt tàng trữ ma tuý", dummy_candidates, top_k=2)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content']}")
