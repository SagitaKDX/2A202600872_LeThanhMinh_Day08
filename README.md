# Ngày 8 — RAG Pipeline v2

**Chương 2 | Ngày 8 trong 15**

---

## Mục Tiêu

Xây dựng một RAG pipeline thực tế, end-to-end, từ thu thập dữ liệu pháp luật và báo chí về ma tuý → xử lý → indexing → retrieval (hybrid + vectorless fallback) → generation có citation.

---

## Chủ Đề Dữ Liệu

**Pháp luật Việt Nam về ma tuý và các chất cấm** + **Các bài báo về nghệ sĩ liên quan tới ma tuý**

---

## Cấu Trúc Thư Mục

```
day_08_rag_pipeline_v2/
├── README.md
├── data/
│   ├── landing/          ← Task 1 & 2: raw files (PDF, DOCX, HTML)
│   └── standardized/     ← Task 3: converted markdown files
├── src/
│   ├── __init__.py
│   ├── task1_collect_legal_docs.py
│   ├── task2_crawl_news.py
│   ├── task3_convert_markdown.py
│   ├── task4_chunking_indexing.py
│   ├── task5_semantic_search.py
│   ├── task6_lexical_search.py
│   ├── task7_reranking.py
│   ├── task8_pageindex_vectorless.py
│   ├── task9_retrieval_pipeline.py
│   └── task10_generation.py
├── notebooks/
│   └── demo.ipynb         ← Notebook demo cho buổi trình bày
├── group_project/
│   └── README.md          ← Hướng dẫn bài tập nhóm
├── requirements.txt
└── .env.example
```

---

## Nhiệm Vụ Chi Tiết

### Task 1 — Thu Thập Văn Bản Pháp Luật (Cá nhân)

Tìm và tải về **tối thiểu 3 văn bản pháp luật** dạng PDF/DOCX về ma tuý và các chất cấm. Lưu vào `data/landing/`.

**Gợi ý nguồn:**
- Luật Phòng, chống ma tuý 2021 (Luật số 73/2021/QH15)
- Nghị định 105/2021/NĐ-CP hướng dẫn thi hành Luật Phòng chống ma tuý
- Bộ luật Hình sự 2015 (sửa đổi 2017) — Chương XX: Các tội phạm về ma tuý
- Thông tư liên tịch về danh mục chất ma tuý và tiền chất

**Yêu cầu:**
- Lưu file gốc (PDF/DOCX) vào `data/landing/legal/`
- Đặt tên file rõ ràng: `luat-phong-chong-ma-tuy-2021.pdf`, `nghi-dinh-105-2021.pdf`, ...

---

### Task 2 — Crawl Bài Báo (Cá nhân)

Crawl **tối thiểu 5 bài báo** về các nghệ sĩ Việt Nam liên quan tới ma tuý.

**Thư viện khuyến nghị:** [Crawl4AI](https://github.com/unclecode/crawl4ai)

**Yêu cầu:**
- Lưu output vào `data/landing/news/`
- Mỗi bài báo lưu thành 1 file (JSON hoặc HTML)
- Ghi rõ metadata: URL gốc, ngày crawl, tiêu đề bài báo

**Code mẫu (Crawl4AI):**
```python
from crawl4ai import AsyncWebCrawler

async def crawl_article(url: str, output_dir: str):
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)
        # Lưu result.markdown vào file
        ...
```

---

### Task 3 — Convert Sang Markdown (Cá nhân)

Sử dụng [MarkItDown](https://github.com/microsoft/markitdown) của Microsoft để convert toàn bộ file trong `data/landing/` thành Markdown.

**Cài đặt:**
```bash
pip install markitdown
```

**Code mẫu:**
```python
from markitdown import MarkItDown

md = MarkItDown()

# Convert PDF
result = md.convert("data/landing/legal/luat-phong-chong-ma-tuy-2021.pdf")
print(result.text_content)

# Convert DOCX
result = md.convert("data/landing/legal/nghi-dinh-105-2021.docx")
```

**Yêu cầu:**
- Output lưu vào `data/standardized/`
- Giữ nguyên cấu trúc thư mục con (`legal/`, `news/`)
- Mỗi file output có tên tương ứng: `luat-phong-chong-ma-tuy-2021.md`

---

### Task 4 — Chunking & Indexing (Cá nhân)

Chọn **một loại chunking strategy** và **một embedding model** để index toàn bộ markdown files vào vector store.

**Chunking — khuyến khích dùng [langchain-text-splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/):**
```bash
pip install langchain-text-splitters
```

Các loại splitter phù hợp:
- `RecursiveCharacterTextSplitter` (mặc định, an toàn)
- `MarkdownHeaderTextSplitter` (tốt cho file có heading rõ)
- `SemanticChunker` (nâng cao, dùng embedding để tách)

**Embedding model gợi ý:**
- `sentence-transformers/all-MiniLM-L6-v2` (nhẹ, nhanh)
- `BAAI/bge-m3` (multilingual, tốt cho tiếng Việt)
- OpenAI `text-embedding-3-small` (nếu có API key)

**Vector Store — khuyến cáo dùng Weaviate:**
```bash
pip install weaviate-client
```
- Weaviate hỗ trợ hybrid search (dense + BM25) built-in
- Có thể dùng Docker hoặc Weaviate Cloud
- Alternatives: ChromaDB (đơn giản), FAISS (nếu chỉ cần dense)

**Yêu cầu:**
- Ghi rõ trong code: dùng chunking nào, chunk_size bao nhiêu, overlap bao nhiêu, vì sao
- Ghi rõ embedding model nào, dimension bao nhiêu
- Index thành công toàn bộ documents

**✓ Thực tế Triển khai (Task 4):**
* **Chiến lược Phân mảnh (Chunking Strategy):** Kết hợp phân tách theo tiêu đề phân cấp Markdown (**`MarkdownHeaderTextSplitter`** chia theo `#`, `##`, `###`) làm tầng sơ cấp nhằm bảo toàn cấu trúc văn bản pháp luật (Chương/Điều/Khoản) và cấu trúc bài báo. Tiếp tục áp dụng **`RecursiveCharacterTextSplitter`** (với `chunk_size=600`, `chunk_overlap=100`) làm tầng thứ cấp để khống chế độ dài của các block ký tự vừa vặn với context limit của mô hình mà không làm mất liên kết thông tin.
* **Mô hình Nhúng (Embedding Model):** **`text-embedding-3-small`** từ OpenAI (Số chiều **1536**), nạp dữ liệu thông qua batch API 100 câu nhằm tối ưu thời gian gọi mạng.
* **Vector Database:** **`Weaviate Cloud (WCD)`** (sử dụng Client v4 mới nhất qua `connect_to_weaviate_cloud`).

---

### Task 5 — Semantic Search Module (Cá nhân)

Viết module thực hiện **semantic search** (dense retrieval) trên vector store.

**Yêu cầu:**
```python
def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}
    """
    ...
```

- Input: query string + top_k
- Output: danh sách chunks có score, sorted descending
- Phải hoạt động được với embedding model đã chọn ở Task 4

**✓ Thực tế Triển khai (Task 5):**
* Sử dụng mô hình **`text-embedding-3-small`** của OpenAI API để mã hóa câu hỏi của người dùng thành vector 1536 chiều.
* Kết nối đến Weaviate Cloud và sử dụng cú pháp tìm kiếm vector chuyên dụng **`near_vector`** của Weaviate client v4.
* Chuyển đổi khoảng cách vector sang điểm số tương đồng theo công thức `score = 1.0 - distance`, lọc các trường metadata như tiêu đề và vị trí phân mảnh (`chunk_index`), sắp xếp giảm dần và trả về kết quả khớp nhất.

---

### Task 6 — Lexical Search Module (Cá nhân)

Viết module thực hiện **lexical search**. Mặc định sử dụng **BM25**.

```bash
pip install rank-bm25
```

**Code mẫu BM25:**
```python
from rank_bm25 import BM25Okapi

# Tokenize corpus
tokenized_corpus = [doc.split() for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

# Search
tokenized_query = query.split()
scores = bm25.get_scores(tokenized_query)
```

**Yêu cầu:**
```python
def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}
    """
    ...
```

**Bonus:** Nếu dùng phương pháp khác (TF-IDF, Elasticsearch, Weaviate BM25 built-in), hãy giải thích cơ chế hoạt động trong buổi demo → **+5 điểm bonus**.

---

### Task 7 — Reranking Module (Cá nhân)

Viết module **reranking** để chấm lại độ liên quan của kết quả retrieval.

**Lựa chọn (chọn 1):**

| Phương pháp | Thư viện / Model | Đặc điểm |
|-------------|-----------------|-----------|
| Cross-encoder reranker | `jinaai/jina-reranker-v2-base-multilingual` | Multilingual, tốt cho tiếng Việt |
| Cross-encoder reranker | `Qwen/Qwen3-Reranker-0.6B` | Nhẹ, hiệu quả |
| MMR (Maximal Marginal Relevance) | Tự implement | Giảm trùng lặp, tăng diversity |
| RRF (Reciprocal Rank Fusion) | Tự implement | Gộp kết quả từ nhiều ranker |

**Code mẫu (Jina Reranker via API):**
```python
import requests

def rerank(query: str, documents: list[str], top_k: int = 5) -> list[dict]:
    response = requests.post(
        "https://api.jina.ai/v1/rerank",
        headers={"Authorization": "Bearer YOUR_API_KEY"},
        json={
            "model": "jina-reranker-v2-base-multilingual",
            "query": query,
            "documents": documents,
            "top_n": top_k
        }
    )
    return response.json()["results"]
```

**Yêu cầu:**
```python
def rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """
    Re-score and re-order candidates based on relevance to query.
    """
    ...
```

---

### Task 8 — PageIndex Vectorless RAG (Cá nhân)

Đăng ký tài khoản tại [https://pageindex.ai/](https://pageindex.ai/), sau đó sử dụng [PageIndex SDK](https://github.com/VectifyAI/PageIndex) để tạo một **vectorless RAG pipeline**.

**Cài đặt:**
```bash
pip install pageindex
```

**Tham khảo:** [https://github.com/VectifyAI/PageIndex](https://github.com/VectifyAI/PageIndex)

**Yêu cầu:**
- Upload tài liệu lên PageIndex
- Viết function query PageIndex và trả về kết quả
```python
def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval using PageIndex.
    Fallback khi hybrid search không trả về kết quả phù hợp.
    """
    ...
```

---

### Task 9 — Retrieval Pipeline Hoàn Chỉnh (Cá nhân)

Kết hợp tất cả modules thành một **retrieval pipeline** thống nhất với logic fallback:

```
Query
  │
  ├─→ Semantic Search (Task 5)  ──┐
  │                                ├─→ Merge + Rerank (Task 7) → Results
  ├─→ Lexical Search (Task 6)  ──┘
  │
  └─→ Nếu hybrid search không có kết quả đủ tốt (score < threshold)
        └─→ Fallback: PageIndex Vectorless (Task 8)
```

**Yêu cầu:**
```python
def retrieve(query: str, top_k: int = 5, score_threshold: float = 0.3) -> list[dict]:
    """
    1. Chạy semantic_search + lexical_search
    2. Merge kết quả (RRF hoặc weighted fusion)
    3. Rerank
    4. Nếu top result score < threshold → fallback PageIndex
    5. Return top_k results
    """
    ...
```

---

### Task 10 — Generation Có Citation (Cá nhân)

Sắp xếp lại context chunks sau reranking để **tránh lost in the middle**, inject vào prompt, và yêu cầu LLM trả lời có **citation**.

**Document Reordering (tránh lost in the middle):**
```python
def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks theo pattern: quan trọng nhất ở đầu và cuối,
    ít quan trọng hơn ở giữa.
    Ví dụ: [1, 3, 5, 4, 2] thay vì [1, 2, 3, 4, 5]
    """
    ...
```

**Prompt template:**
```python
SYSTEM_PROMPT = """Answer the following question comprehensively.
For every statement of fact or claim, immediately insert a citation
in brackets linking to the specific source
(e.g., [Author/Platform Name, Year]).
If the information is not explicitly stated in the provided context
or knowledge base, state 'I cannot verify this information'
rather than guessing."""

def generate_with_citation(query: str, context_chunks: list[dict]) -> str:
    """
    1. Reorder chunks để tránh lost in the middle
    2. Format context với source metadata
    3. Inject vào prompt với SYSTEM_PROMPT
    4. Gọi LLM (OpenAI, Gemini, hoặc local model)
    5. Return answer có citation
    """
    ...
```

**Yêu cầu:**
- Chọn top_k và top_p phù hợp (giải thích lý do trong code comment)
- Output phải có citation dạng `[Nguồn, Năm]`
- Nếu không đủ evidence → trả về "I cannot verify this information"

---

## Bài Tập Nhóm

> **Sau khi hoàn thành bài cá nhân**, ngồi lại với nhóm để xây dựng **1 trong 2 sản phẩm** sau:

---

### Yêu cầu 1: Sản phẩm nhóm RAG Chatbot

Xây dựng chatbot trả lời câu hỏi về pháp luật ma tuý và tin tức liên quan.

**Yêu cầu:**
- Giao diện chat (Streamlit / Gradio / Chainlit)
- Trả lời có citation (dựa trên Task 10)
- Hỗ trợ follow-up questions (conversation memory)
- Hiển thị source documents đã dùng

**Stack gợi ý:**
```
Chainlit/Streamlit → Retrieval (Task 9) → Generation (Task 10) → Display
```

---

### Yêu cầu 2: RAG Evaluation Pipeline

Sử dụng **1 trong 3 framework** sau để evaluate pipeline RAG của nhóm:

#### Framework lựa chọn

| Framework | Cài đặt | Đặc điểm |
|-----------|---------|-----------|
| [DeepEval](https://github.com/confident-ai/deepeval) | `pip install deepeval` | Nhiều metric built-in, dễ integrate với pytest |
| [RAGAS](https://github.com/explodinggradients/ragas) | `pip install ragas` | Chuẩn industry cho RAG eval, 3 trục chính |
| [TruLens](https://github.com/truera/trulens) | `pip install trulens` | Dashboard UI, feedback functions mạnh |

#### Yêu cầu Evaluation

1. **Tạo Golden Dataset** — tối thiểu 15 cặp Q&A (question, expected_answer, expected_context)
2. **Chạy evaluation** trên toàn bộ golden dataset với các metrics sau:
   - **Faithfulness** — câu trả lời có bám đúng context không?
   - **Answer Relevance** — câu trả lời có đúng câu hỏi không?
   - **Context Recall** — retriever có lấy đủ evidence không?
   - **Context Precision** — trong context lấy về, bao nhiêu % thực sự hữu ích?
3. **So sánh A/B** — chạy eval trên ít nhất 2 config khác nhau (ví dụ: có reranking vs không reranking, hoặc hybrid vs dense-only)
4. **Báo cáo** — bảng điểm + phân tích worst performers + đề xuất cải tiến

#### Code mẫu — DeepEval

```python
from deepeval import evaluate
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric,
)
from deepeval.test_case import LLMTestCase

# Tạo test cases từ golden dataset
test_cases = []
for item in golden_dataset:
    result = rag_pipeline.generate_with_citation(item["question"])
    test_case = LLMTestCase(
        input=item["question"],
        actual_output=result["answer"],
        expected_output=item["expected_answer"],
        retrieval_context=[c["content"] for c in result["sources"]],
    )
    test_cases.append(test_case)

# Chạy evaluation
metrics = [
    FaithfulnessMetric(threshold=0.7),
    AnswerRelevancyMetric(threshold=0.7),
    ContextualRecallMetric(threshold=0.7),
    ContextualPrecisionMetric(threshold=0.7),
]

results = evaluate(test_cases, metrics)
```

#### Code mẫu — RAGAS

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from datasets import Dataset

# Chuẩn bị data
eval_data = {
    "question": [],
    "answer": [],
    "contexts": [],
    "ground_truth": [],
}

for item in golden_dataset:
    result = rag_pipeline.generate_with_citation(item["question"])
    eval_data["question"].append(item["question"])
    eval_data["answer"].append(result["answer"])
    eval_data["contexts"].append([c["content"] for c in result["sources"]])
    eval_data["ground_truth"].append(item["expected_answer"])

dataset = Dataset.from_dict(eval_data)

# Chạy evaluation
result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
)
print(result.to_pandas())
```

#### Code mẫu — TruLens

```python
from trulens.apps.custom import TruCustomApp, instrument
from trulens.core import Feedback
from trulens.providers.openai import OpenAI as TruOpenAI

provider = TruOpenAI()

# Define feedback functions
f_faithfulness = Feedback(provider.groundedness_measure_with_cot_reasons).on_output()
f_relevance = Feedback(provider.relevance).on_input_output()
f_context_relevance = Feedback(provider.context_relevance).on_input()

# Wrap RAG pipeline
tru_rag = TruCustomApp(
    rag_pipeline,
    app_name="DrugLaw_RAG",
    feedbacks=[f_faithfulness, f_relevance, f_context_relevance],
)

# Run evaluation
with tru_rag as recording:
    for item in golden_dataset:
        rag_pipeline.generate_with_citation(item["question"])

# View dashboard
from trulens.dashboard import run_dashboard
run_dashboard()
```

#### Deliverable Evaluation

- [ ] File `group_project/evaluation/golden_dataset.json` — 15+ cặp Q&A
- [ ] File `group_project/evaluation/eval_pipeline.py` — script chạy evaluation
- [ ] File `group_project/evaluation/results.md` — bảng điểm + phân tích
- [ ] So sánh A/B ít nhất 2 configs

---

### Yêu Cầu Chung

1. **Tích hợp pipeline** từ bài cá nhân của các thành viên
2. **Demo hoạt động được** trong buổi trình bày (chạy local hoặc deploy)
3. **Evaluation pipeline** chạy được và có báo cáo kết quả
4. **Code push lên repository** chung của nhóm
5. **README** mô tả kiến trúc và phân công (xem `group_project/README.md`)

---

### Kiến Trúc Hệ Thống

```
[Vẽ diagram kiến trúc ở đây]
```

---

### Phân Công Công Việc

| Thành viên | MSSV | Nhiệm vụ | Trạng thái |
|-----------|------|----------|------------|
| | | | |
| | | | |
| | | | |
| | | | |

---

### Hướng Dẫn Chạy

```bash
# Cài đặt dependencies
pip install -r requirements.txt

# Chạy app
streamlit run app.py
# hoặc
chainlit run app.py
```

---

### Lưu ý

Hãy giữ lại repo này nếu như bạn học track 3 giai đoạn 2, chúng ta sẽ phát triển tiếp dự án lên knowledge graph để khắc phục các câu hỏi hóc búa khi có các câu hỏi khó.

---

## Cài Đặt Môi Trường

```bash
pip install -r requirements.txt
```

Tạo file `.env` từ `.env.example`:
```bash
cp .env.example .env
# Điền API keys vào .env
```

---

## Chấm Điểm

### Tổng Quan Phân Bổ Điểm

| Thành phần | Tỷ trọng | Mô tả |
|-----------|----------|-------|
| **Bài Cá Nhân** | **50%** | 10 tasks, chấm bằng automated tests + manual review |
| **Bài Nhóm** | **30%** | RAG Chatbot + Evaluation pipeline |
| **Bonus** | **20%** | Các tiêu chí nâng cao (xem bên dưới) |

---

### Bài Cá Nhân — 50 điểm (50%)

Chấm bằng automated test suite (`pytest tests/ -v`). Mỗi task có test riêng.

| Task | Nội dung | Điểm | Test |
|------|----------|------|------|
| 1 | Thu thập văn bản pháp luật (≥3 files tồn tại trong `data/landing/legal/`) | 3 | `test_task1_*` |
| 2 | Crawl bài báo (≥5 files tồn tại trong `data/landing/news/`) | 3 | `test_task2_*` |
| 3 | Convert markdown (files tồn tại trong `data/standardized/`) | 4 | `test_task3_*` |
| 4 | Chunking + Indexing (vector store có data) | 7 | `test_task4_*` |
| 5 | Semantic search trả về kết quả đúng format, sorted | 6 | `test_task5_*` |
| 6 | Lexical search (BM25) trả về kết quả đúng format | 6 | `test_task6_*` |
| 7 | Reranking hoạt động, output re-sorted | 6 | `test_task7_*` |
| 8 | PageIndex query trả về kết quả | 4 | `test_task8_*` |
| 9 | Retrieval pipeline + fallback logic hoạt động | 7 | `test_task9_*` |
| 10 | Generation có citation + reorder | 4 | `test_task10_*` |
| **Tổng** | | **50** | |

---

### Bài Nhóm — 30 điểm (30%)

| Tiêu chí | Điểm |
|----------|------|
| RAG Chatbot demo hoạt động được | 8 |
| Tích hợp pipeline các thành viên | 4 |
| Kiến trúc rõ ràng + README | 3 |
| Chất lượng câu trả lời (có citation, đúng nội dung) | 3 |
| **Evaluation pipeline** (DeepEval / RAGAS / TruLens) | **12** |
| — Golden dataset ≥15 Q&A pairs | 3 |
| — Chạy eval với ≥4 metrics | 4 |
| — So sánh A/B ≥2 configs + phân tích | 3 |
| — Báo cáo kết quả có phân tích worst performers | 2 |

---

### Bonus — 20 điểm (20%)

Demo hoặc đặt câu hỏi mà nhóm đang demo khiến LLM không trả lời được (mỗi câu 5 điểm)

---

### Chạy Test Chấm Điểm Bài Cá Nhân

```bash
# Chạy toàn bộ test suite
pytest tests/ -v

# Chạy từng task
pytest tests/test_individual.py::TestTask1 -v
pytest tests/test_individual.py::TestTask5 -v
```

---

## Hướng Dẫn Thời Gian

| Giai đoạn | Thời gian | Hoạt động |
|-----------|-----------|-----------|
| Task 1–3 | 0:00–0:45 | Thu thập data + convert markdown |
| Task 4–6 | 0:45–1:45 | Chunking, indexing, search modules |
| Task 7–8 | 1:45–2:15 | Reranking + PageIndex setup |
| Task 9–10 | 2:15–3:00 | Pipeline hoàn chỉnh + generation |
| Bài nhóm | Ngoài giờ | Tích hợp + build demo |

---

## Tài Liệu Tham Khảo

- [Crawl4AI](https://github.com/unclecode/crawl4ai) — Web crawling library
- [MarkItDown](https://github.com/microsoft/markitdown) — Microsoft document converter
- [LangChain Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/) — Chunking strategies
- [Weaviate](https://weaviate.io/developers/weaviate) — Vector database with hybrid search
- [rank-bm25](https://github.com/dorianbrown/rank_bm25) — BM25 implementation
- [PageIndex](https://github.com/VectifyAI/PageIndex) — Vectorless RAG
- [Jina Reranker](https://jina.ai/reranker/) — Cross-encoder reranking API
- Liu et al. (2023), *Lost in the Middle: How Language Models Use Long Contexts*

---

## Tư Duy Phát Triển & Lựa Chọn Giải Pháp (Task 2 - Task 7)

Tài liệu này trình bày chi tiết tư duy thiết kế hệ thống và lý do lựa chọn các phương pháp/công nghệ cụ thể từ Task 2 đến Task 7 trong hệ thống RAG (Retrieval-Augmented Generation) phục vụ tra cứu văn bản pháp luật và tin tức ma túy.

### Task 2: Thu thập Tin tức (News Crawler)

* **Giải pháp lựa chọn:** Sử dụng song song **Crawl4AI** (chính) và **Requests + BeautifulSoup + Markdownify** (dự phòng).
* **Tư duy phát triển:** 
  Khi thu thập tin tức từ các báo điện tử (như VnExpress), hệ thống thường đối mặt với cấu trúc trang phức tạp và các cơ chế chống cào dữ liệu. Tôi thiết lập cơ chế hai lớp:
  * **Crawl4AI** đảm nhận nhiệm vụ chính vì đây là thư viện hiện đại được thiết kế riêng cho việc chuẩn bị dữ liệu LLM. Công cụ này tự động bóc tách và làm sạch các thẻ HTML thừa để trả về định dạng Markdown sạch sẽ.
  * **Requests + BeautifulSoup + Markdownify** đóng vai trò là lưới bảo hiểm dự phòng. Nếu Crawl4AI bị lỗi hoặc môi trường thiếu thư viện headless browser, hệ thống sẽ thực hiện yêu cầu HTTP thông thường, tìm đúng khung nội dung (`article`) và tự động chuyển sang Markdown. Cơ chế dự phòng này giúp pipeline thu thập dữ liệu luôn vận hành ổn định mà không bị gián đoạn.

---

### Task 3: Chuyển đổi Định dạng Markdown (Markdown Converter)

* **Giải pháp lựa chọn:** Sử dụng thư viện **MarkItDown** của Microsoft.
* **Tư duy phát triển:**
  Dữ liệu pháp luật được lưu trữ dưới nhiều định dạng thô khác nhau (PDF, DOCX, JSON). Để xây dựng một hệ thống RAG hiệu quả, chúng ta cần chuẩn hóa toàn bộ các tài liệu này về một định dạng thống nhất: **Markdown**.
  * Tôi chọn **MarkItDown** vì thư viện này có khả năng phân tích và trích xuất cấu trúc văn bản rất tốt từ các định dạng phức tạp (như PDF và DOCX) sang văn bản thuần mà không làm mất thông tin.
  * Việc sử dụng **Markdown** làm định dạng chuẩn hóa giúp bảo toàn các cấu trúc phân cấp quan trọng (Chương, Điều, Khoản thông qua các ký tự `#`, `##`, `###`). Cấu trúc này là chìa khóa để bộ phân mảnh dữ liệu (splitter) ở bước sau hoạt động chính xác hơn so với văn bản thô không định dạng.

---

### Task 4: Phân đoạn & Lập chỉ mục (Chunking & Indexing)

* **Giải pháp lựa chọn:**
  * **Phân đoạn hỗn hợp:** Kết hợp bộ tách tiêu đề Markdown (`MarkdownHeaderTextSplitter`) và bộ tách ký tự đệ quy (`RecursiveCharacterTextSplitter`).
  * **Mô hình nhúng:** `text-embedding-3-small` (1536 chiều) của OpenAI qua API.
  * **Cơ sở dữ liệu Vector:** Weaviate Cloud (WCD).
* **Tư duy phát triển:**
  * **Tại sao cần Phân đoạn hỗn hợp?** Văn bản pháp luật Việt Nam có cấu trúc phân cấp cực kỳ nghiêm ngặt (Chương -> Điều -> Khoản). Nếu chỉ chia cắt ngẫu nhiên theo số ký tự, chúng ta sẽ làm mất liên kết thông tin (ví dụ: Khoản 2 của Điều 248 sẽ bị tách rời khỏi tiêu đề "Tội tàng trữ trái phép chất ma tuý"). Bằng cách dùng `MarkdownHeaderTextSplitter` trước, tôi giữ lại thông tin tiêu đề gốc trong siêu dữ liệu (metadata) của từng đoạn. Sau đó, tôi áp dụng `RecursiveCharacterTextSplitter` với kích thước đoạn `600` ký tự và độ trùng lặp `100` ký tự để khống chế độ dài, giúp các đoạn thông tin vừa vặn với giới hạn ngữ cảnh của LLM và giữ được tính liên kết giữa các phân đoạn cạnh nhau.
  * **Tại sao chọn text-embedding-3-small?** Thay vì tải các mô hình cục bộ nặng nề (như `bge-m3` nặng hơn 2GB) gây tốn RAM và làm chậm máy, tôi sử dụng mô hình qua API của OpenAI. Mô hình này rất nhẹ, chi phí thấp, hỗ trợ đa ngôn ngữ xuất sắc và trả về vector 1536 chiều chất lượng cao, giúp tăng độ chính xác khi đối chiếu ngữ nghĩa.
  * **Tại sao chọn Weaviate Cloud?** Weaviate là cơ sở dữ liệu vector chuẩn công nghiệp, hỗ trợ tìm kiếm kết hợp (Hybrid Search) tích hợp sẵn. Tôi chọn phiên bản đám mây để đơn giản hóa việc triển khai và đảm bảo hệ thống có thể mở rộng dễ dàng mà không phụ thuộc vào tài nguyên phần cứng local.

---

### Task 5: Tìm kiếm Ngữ nghĩa (Semantic Search)

* **Giải pháp lựa chọn:** Tìm kiếm tương đồng vector (`near_vector`) trên Weaviate.
* **Tư duy phát triển:**
  Người dùng thường không nhớ chính xác từng từ ngữ chuyên môn pháp lý khi đặt câu hỏi (ví dụ: họ gõ "chơi thuốc lắc bị phạt thế nào" thay vì "tội sử dụng trái phép chất ma tuý").
  * Tìm kiếm ngữ nghĩa giải quyết vấn đề này bằng cách chuyển đổi câu hỏi của người dùng thành vector biểu diễn không gian thông qua OpenAI API.
  * Sau đó, Weaviate sẽ thực hiện tính toán khoảng cách vector (Cosine distance) giữa truy vấn và toàn bộ các phân đoạn đã được lưu trữ. Những phân đoạn có ý nghĩa tương đồng nhất sẽ được trả về đầu tiên kể cả khi chúng không trùng khớp bất kỳ từ khóa thô nào với câu hỏi.

---

### Task 6: Tìm kiếm Từ khóa (Lexical Search)

* **Giải pháp lựa chọn:** Thuật toán **BM25** (thư viện `rank-bm25`).
* **Tư duy phát triển:**
  Mặc dù tìm kiếm ngữ nghĩa rất thông minh, nhưng nó lại có điểm yếu là đôi khi bỏ sót các chi tiết chính xác hoặc số hiệu pháp lý (như số hiệu điều luật "Điều 248", tên chất cấm cụ thể như "Heroine", "Methamphetamine").
  * Tôi chọn **BM25** để bù đắp cho điểm yếu trên. Thuật toán này chấm điểm dựa trên tần suất xuất hiện của từ khóa chính xác trong đoạn văn và mức độ đặc trưng của từ khóa đó trên toàn bộ ngữ liệu.
  * Sự kết hợp giữa Tìm kiếm ngữ nghĩa (Task 5) và Tìm kiếm từ khóa (Task 6) tạo tiền đề cho việc xây dựng cơ chế Tìm kiếm kết hợp (Hybrid Search) ở các bước sau, tận dụng ưu điểm của cả hai thế giới: hiểu ý nghĩa câu hỏi của Semantic Search và tìm chính xác từ khóa của Lexical Search.

---

### Task 7: Chấm điểm Xếp hạng lại (Reranking)

* **Giải pháp lựa chọn:** Mô hình Cross-Encoder cục bộ siêu nhẹ **`mixedbread-ai/mxbai-rerank-xsmall-v1`** (tích hợp dự phòng qua Jina Reranker API).
* **Tư duy phát triển:**
  Khi kết hợp kết quả từ Semantic Search và Lexical Search, chúng ta có một danh sách thô gồm nhiều phân đoạn tiềm năng. Tuy nhiên, các mô hình nhúng (Bi-Encoders) chấm điểm dựa trên việc so sánh độc lập hai vector câu hỏi và tài liệu, dẫn đến việc thiếu phân tích sâu sắc ở cấp độ từ ngữ.
  * Tôi chọn sử dụng mô hình **Cross-Encoder** làm lớp lọc cuối cùng. Thay vì tính toán độc lập, Cross-Encoder đưa cả câu hỏi và phân đoạn tài liệu vào mô hình cùng một lúc để tính toán mức độ chú ý chéo (cross-attention) giữa các từ. Việc này giống như việc cho một chuyên gia đọc lại kỹ lưỡng cả câu hỏi và đoạn văn để chấm điểm độ liên quan.
  * Tôi lựa chọn mô hình **`mxbai-rerank-xsmall-v1`** vì nó siêu nhẹ (~150MB), chạy cực nhanh trên CPU cục bộ mà vẫn mang lại hiệu suất xếp hạng lại xuất sắc cho tiếng Việt. Bên cạnh đó, tôi cấu hình thêm Jina Reranker API để làm phương án dự phòng hiệu năng cao khi cần thiết.
  * Bước này đảm bảo loại bỏ các kết quả nhiễu, chỉ giữ lại những điều khoản pháp luật thực sự trả lời được câu hỏi của người dùng để đưa vào LLM ở Task 10, trực tiếp giải quyết vấn đề trích xuất sai ngữ cảnh.

