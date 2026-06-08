# RAG Evaluation Results

## Framework sử dụng

> **DeepEval** kết hợp với **Custom LLM Judge** (sử dụng GPT-4o-mini làm quan tòa đánh giá) để đánh giá offline tốc độ cao và đảm bảo tính độc lập ổn định cho pipeline.

---

## Overall Scores

| Metric | Config A (hybrid + rerank) | Config B (dense-only) | Δ |
|--------|---------------------------|----------------------|---|
| Faithfulness | 0.9100 | 0.8200 | +0.0900 |
| Answer Relevance | 0.8150 | 0.7650 | +0.0500 |
| Context Recall | 0.7900 | 0.7000 | +0.0900 |
| Context Precision | 0.7700 | 0.6900 | +0.0800 |
| **Average** | **0.8213** | **0.7438** | **+0.0775** |

---

## A/B Comparison Analysis

**Config A (hybrid + rerank):**
- Sử dụng tìm kiếm Hybrid (kết hợp Dense search với Vector embeddings `text-embedding-3-small` và Sparse search BM25), trộn kết quả qua thuật toán RRF (Reciprocal Rank Fusion). Sau đó sử dụng mô hình Cross-Encoder cục bộ siêu nhẹ `mixedbread-ai/mxbai-rerank-xsmall-v1` để xếp hạng lại độ liên quan của các phân đoạn tài liệu trước khi gửi vào prompt của LLM.

**Config B (dense-only):**
- Sử dụng tìm kiếm Hybrid (kết hợp Dense + BM25) trộn bằng RRF trực tiếp, lấy top_k tài liệu hàng đầu mà không qua bước xếp hạng lại bằng Cross-Encoder.

**Kết luận:**
- **Config A mang lại kết quả tốt hơn đáng kể ở tất cả các chỉ số (đặc biệt là Context Precision và Faithfulness)**. Nhờ có Cross-Encoder reranking, các phân đoạn tài liệu chứa nội dung trả lời chính xác nhất được đẩy lên đầu và lọc bớt nhiễu, giúp LLM nhận diện thông tin dễ dàng hơn, tránh hiện tượng "lost in the middle" và giảm thiểu tối đa hiện tượng ảo giác (hallucination), làm tăng điểm Faithfulness.

---

## Worst Performers (Bottom 3)

| # | Question | Faithfulness | Relevance | Recall | Failure Stage | Root Cause |
|---|----------|-------------|-----------|--------|---------------|------------|
| 1 | Tội sản xuất trái phép chất ma túy theo Điều 248 Bộ luật Hình sự có khung hình phạt cao nhất là gì? | 0.00 | 0.00 | 0.00 | Retrieval Stage | Các tài liệu liên quan không được thu thập đủ từ cơ sở dữ liệu vector Weaviate (do thiếu từ khóa đặc trưng hoặc embedding khoảng cách xa). |
| 2 | Đường dây ma túy liên quan đến người mẫu Andrea Aybar và ca sĩ Chi Dân bắt nguồn từ vụ án nào? | 1.00 | 0.00 | 0.00 | Retrieval Stage | Các tài liệu liên quan không được thu thập đủ từ cơ sở dữ liệu vector Weaviate (do thiếu từ khóa đặc trưng hoặc embedding khoảng cách xa). |
| 3 | Ca sĩ Miu Lê và những người liên quan bị bắt quả tang sử dụng ma túy ở địa điểm nào? | 0.80 | 0.70 | 0.50 | Retrieval Stage | Các tài liệu liên quan không được thu thập đủ từ cơ sở dữ liệu vector Weaviate (do thiếu từ khóa đặc trưng hoặc embedding khoảng cách xa). |

---

## Recommendations

### Cải tiến 1
**Action:**  
- Cải thiện tham số `score_threshold` hoặc tăng `top_k` ở bước retrieval thô trước khi Rerank để tránh bỏ sót thông tin quan trọng đối với các câu hỏi phức tạp (tăng Context Recall).

### Cải tiến 2
**Action:**  
- Tối ưu cấu trúc phân mảnh (Chunking Strategy): bổ sung thêm metadata phân cấp rõ ràng hơn cho các Nghị định chi tiết để khi tìm kiếm có thể lấy được toàn bộ ngữ cảnh liên đới của Chương/Điều.

### Cải tiến 3
**Action:**  
- Tinh chỉnh `SYSTEM_PROMPT` của Generator để quy định nghiêm ngặt hơn nữa việc KHÔNG tự ý suy diễn hoặc thêm bớt thông tin ngoài tài liệu tham chiếu, nhằm đẩy điểm số Faithfulness lên tối đa 1.0.
