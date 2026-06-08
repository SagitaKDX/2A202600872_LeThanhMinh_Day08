# ⚖️ Giao Diện Chatbot Thẩm Phán Số (Streamlit & Chainlit)

Thư mục này chứa hai giao diện người dùng (UI) hoàn chỉnh phục vụ cho việc tương tác, tra cứu và hỏi đáp luật phòng chống ma túy cùng tin tức liên quan.

---

## 🚀 Hướng Dẫn Cài Đặt

Trước khi khởi chạy giao diện, hãy đảm bảo bạn đã cài đặt đầy đủ các thư viện hỗ trợ bằng cách chạy lệnh sau trong thư mục gốc của dự án:

```bash
# Cài đặt thư viện Chainlit (nếu chưa cài đặt)
./venv/bin/pip install chainlit
```

---

## 🎨 1. Khởi Chạy Giao Diện Streamlit (Bảng điều khiển & Trò chuyện)

Giao diện **Streamlit** được thiết kế dưới dạng bảng điều khiển chuyên nghiệp, cho phép tùy chỉnh các tham số hệ thống như ngưỡng chất lượng (Score Threshold), số lượng tài liệu lấy ra (Top K), và bật/tắt mô hình Reranking.

### Cách chạy:
Chạy câu lệnh sau trong thư mục gốc dự án:
```bash
./venv/bin/streamlit run RAG/app_streamlit.py
```

Sau khi chạy, trình duyệt sẽ tự động mở giao diện tại địa chỉ: `http://localhost:8501`

---

## 💬 2. Khởi Chạy Giao Diện Chainlit (Hội thoại nâng cao với Steps)

Giao diện **Chainlit** tối ưu hóa cho trải nghiệm trò chuyện, hiển thị từng bước xử lý dữ liệu trung gian dưới dạng **Steps** (🔄 Phân tích câu hỏi → 🔍 Tìm kiếm Hybrid → 📑 Sắp xếp tài liệu) giúp tăng tính minh bạch và độ tin cậy. Tài liệu tham chiếu cũng được đính kèm ở khung (Panel) bên cạnh câu trả lời rất gọn gàng.

### Cách chạy:
Chạy câu lệnh sau trong thư mục gốc dự án:
```bash
./venv/bin/chainlit run RAG/app_chainlit.py -w
```
*(Tham số `-w` bật chế độ tự động tải lại giao diện khi có thay đổi code - auto-reload)*

Sau khi chạy, trình duyệt sẽ tự động mở giao diện tại địa chỉ: `http://localhost:8000`

---

## 💡 Các Tính Năng Đang Hoạt Động
1. **Lịch sử trò chuyện (Conversation Memory):** Hệ thống ghi nhớ các lượt chat trước để hiểu các câu hỏi tiếp nối (Ví dụ: *"Hình phạt của tội này là gì?"* sau khi hỏi về tội tàng trữ ma túy).
2. **Steps trực quan:** Hiển thị chi tiết cách hệ thống xử lý dữ liệu dưới nền.
3. **Citations & Sources:** Trích dẫn nguồn chi tiết từng phần trả lời và hiển thị tệp tin đính kèm để kiểm chứng.
