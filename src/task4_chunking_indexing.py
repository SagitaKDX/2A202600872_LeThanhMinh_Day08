"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy (giải thích lý do)
    3. Chọn 1 embedding model (giải thích lý do)
    4. Index vào vector store (Weaviate khuyến cáo)

Chunking options (langchain-text-splitters):
    - RecursiveCharacterTextSplitter: an toàn, phổ biến
    - MarkdownHeaderTextSplitter: tốt cho file có heading
    - SemanticChunker: dùng embedding để tách (nâng cao)

Embedding model options:
    - sentence-transformers/all-MiniLM-L6-v2 (384 dim, nhẹ)
    - BAAI/bge-m3 (1024 dim, multilingual, tốt cho tiếng Việt)
    - OpenAI text-embedding-3-small (1536 dim, API)

Vector store options:
    - Weaviate (khuyến cáo: hỗ trợ hybrid search built-in)
    - ChromaDB (đơn giản, local)
    - FAISS (chỉ dense search)

Cài đặt:
    pip install langchain-text-splitters sentence-transformers weaviate-client
"""

from pathlib import Path

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"



# TODO: Chọn chunking strategy và giải thích vì sao
CHUNK_SIZE = 600        # Kích thước 500-800 phù hợp với giới hạn ngữ cảnh của các mô hình LLM và embedding.
CHUNK_OVERLAP = 100     # Chồng lấn 50-100 ký tự giúp giữ lại ngữ cảnh liên kết giữa các đoạn.
CHUNKING_METHOD = "markdown_header_and_recursive"  # Sử dụng MarkdownHeaderTextSplitter để bảo toàn cấu trúc văn bản pháp luật trước, sau đó dùng RecursiveCharacterTextSplitter để khống chế kích thước tối đa.

# TODO: Chọn embedding model và giải thích
EMBEDDING_MODEL = "text-embedding-3-small"  # Model embedding từ OpenAI, hiệu năng cao và gọn nhẹ thông qua API.
EMBEDDING_DIM = 1536

# TODO: Chọn vector store
VECTOR_STORE = "weaviate"  # Hỗ trợ hybrid search (sparse + dense) mặc định cực mạnh.

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []
    if not STANDARDIZED_DIR.exists():
        return documents
    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        if md_file.name.startswith("."):
            continue
        content = md_file.read_text(encoding="utf-8")
        # doc_type là thư mục cha chứa file (legal hoặc news)
        doc_type = "legal" if "legal" in str(md_file.parent) else "news"
        documents.append({
            "content": content,
            "metadata": {"source": md_file.name, "type": doc_type}
        })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents sử dụng kết hợp MarkdownHeaderTextSplitter và RecursiveCharacterTextSplitter.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

    headers_to_split_on = [
        ("#", "Header_1"),
        ("##", "Header_2"),
        ("###", "Header_3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = []
    for doc in documents:
        # Bước 1: Chia nhỏ theo cấu trúc markdown headers
        md_header_splits = markdown_splitter.split_text(doc["content"])

        # Bước 2: Với mỗi header split, tiếp tục chia nhỏ bằng Recursive nếu vượt quá kích thước
        for split in md_header_splits:
            sub_splits = recursive_splitter.split_text(split.page_content)
            
            # Metadata cơ bản từ document gốc
            base_meta = {
                "source": doc["metadata"]["source"],
                "doc_type": doc["metadata"]["type"]
            }
            # Bổ sung các headers tìm thấy từ MarkdownHeaderTextSplitter
            for k, v in split.metadata.items():
                base_meta[k] = str(v)

            for sub_split in sub_splits:
                chunks.append({
                    "content": sub_split,
                    "metadata": base_meta.copy()
                })

    # Đánh chỉ mục chunk_index
    for idx, chunk in enumerate(chunks):
        chunk["metadata"]["chunk_index"] = idx

    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng model đã chọn.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    import os
    from openai import OpenAI
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or "sk-xxx" in api_key or not api_key.strip():
        raise ValueError("Missing or invalid OPENAI_API_KEY in .env. Please configure your OpenAI API Key before running.")

    print(f"Loading embedding model: {EMBEDDING_MODEL} via OpenAI API...")
    client = OpenAI(api_key=api_key)
    texts = [c["content"] for c in chunks]
    print(f"Encoding {len(texts)} chunks via API...")
    
    # Send in batches of 100 to avoid request size limits
    batch_size = 100
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch_texts
        )
        all_embeddings.extend([data.embedding for data in response.data])

    for chunk, emb in zip(chunks, all_embeddings):
        chunk["embedding"] = emb
    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào vector store Weaviate.
    """
    import weaviate
    from weaviate.classes.config import Configure, Property, DataType
    import os
    from dotenv import load_dotenv

    load_dotenv()

    weaviate_url = os.getenv("WEAVIATE_URL")
    weaviate_api_key = os.getenv("WEAVIATE_API_KEY")

    client = None
    # Nếu có thông số Weaviate Cloud thì dùng Cloud, ngược lại dùng local/embedded
    if weaviate_url and weaviate_api_key and "xxx" not in weaviate_url:
        url = weaviate_url.strip('"\'')
        if not url.startswith("http://") and not url.startswith("https://"):
            url = f"https://{url}"
        print(f"Connecting to Weaviate Cloud cluster at {url}...")
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=url,
            auth_credentials=weaviate.auth.AuthApiKey(weaviate_api_key.strip('"\''))
        )
    else:
        try:
            print("Connecting to local Weaviate (port 8080)...")
            client = weaviate.connect_to_local()
            if not client.is_ready():
                raise ConnectionError("Local Weaviate not responding")
        except Exception as e:
            print(f"Local Weaviate not running: {e}. Falling back to embedded Weaviate...")
            client = weaviate.connect_to_embedded()

    try:
        collection_name = "DrugLawDocs"

        # Xoá collection nếu đã tồn tại để tránh nạp trùng dữ liệu
        if client.collections.exists(collection_name):
            print(f"Recreating collection '{collection_name}'...")
            client.collections.delete(collection_name)

        properties = [
            Property(name="content", data_type=DataType.TEXT),
            Property(name="source", data_type=DataType.TEXT),
            Property(name="doc_type", data_type=DataType.TEXT),
            Property(name="chunk_index", data_type=DataType.INT),
            Property(name="header_1", data_type=DataType.TEXT),
            Property(name="header_2", data_type=DataType.TEXT),
            Property(name="header_3", data_type=DataType.TEXT),
        ]

        # Tạo collection mới (vectorizer_config=none vì chúng ta tự cung cấp embedding)
        collection = client.collections.create(
            name=collection_name,
            vectorizer_config=Configure.Vectorizer.none(),
            properties=properties
        )

        print(f"Inserting {len(chunks)} chunks into Weaviate...")
        with collection.batch.dynamic() as batch:
            for chunk in chunks:
                properties_data = {
                    "content": chunk["content"],
                    "source": chunk["metadata"].get("source", ""),
                    "doc_type": chunk["metadata"].get("doc_type", ""),
                    "chunk_index": int(chunk["metadata"].get("chunk_index", 0)),
                    "header_1": chunk["metadata"].get("Header_1", ""),
                    "header_2": chunk["metadata"].get("Header_2", ""),
                    "header_3": chunk["metadata"].get("Header_3", ""),
                }
                batch.add_object(
                    properties=properties_data,
                    vector=chunk["embedding"]
                )
        print("✓ All chunks successfully indexed into Weaviate.")
    finally:
        client.close()


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("✓ Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("✓ Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()
