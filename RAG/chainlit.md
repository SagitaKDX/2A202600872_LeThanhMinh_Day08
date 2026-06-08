# ⚖️ Trợ Lý Pháp Luật - Thẩm Phán Số V2

Chào mừng bạn đến với **Thẩm Phán Số**, hệ thống hỗ trợ tra cứu văn bản pháp luật phòng chống ma túy và theo dõi tin tức nghệ sĩ Việt Nam liên quan đến các chất cấm.

Hệ thống được phát triển tích hợp các công nghệ RAG tiên tiến:
1. **Tìm kiếm kết hợp (Hybrid Search):** Kết hợp Semantic Search (truy vấn ngữ nghĩa) và Lexical Search (BM25 từ khóa).
2. **Xếp hạng lại (Reranking):** Sử dụng Cross-Encoder (Jina Reranker API hoặc local model) để chọn lọc những thông tin chính xác nhất.
3. **Cơ chế dự phòng (Fallback):** Tự động chuyển hướng tìm kiếm không dùng vector qua PageIndex khi điểm số tương đồng dưới ngưỡng an toàn.
4. **Tránh Lost in the middle:** Tự động sắp xếp lại các phân đoạn tài liệu để đảm bảo mô hình LLM tập trung tốt nhất vào thông tin cốt lõi.

### Hướng dẫn sử dụng:
* Hãy nhập câu hỏi trực tiếp vào khung chat bên dưới.
* Xem chi tiết các bước xử lý bằng cách bấm vào các nhãn tiến trình (🔄, 🔍, 📑) trong cuộc hội thoại.
* Nhấp vào các liên kết đính kèm ở khung bên cạnh để đọc toàn văn tài liệu gốc được dùng làm căn cứ.
