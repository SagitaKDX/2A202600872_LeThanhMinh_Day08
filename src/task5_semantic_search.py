"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    import os
    import weaviate
    from weaviate.classes.query import MetadataQuery
    from openai import OpenAI
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or "sk-xxx" in api_key or not api_key.strip():
        raise ValueError("Missing or invalid OPENAI_API_KEY in .env. Please configure your OpenAI API Key before running.")

    # 1. Embed query
    print(f"Embedding query: '{query}' via OpenAI API...")
    openai_client = OpenAI(api_key=api_key)
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    query_embedding = response.data[0].embedding

    # 2. Connect to Weaviate
    weaviate_url = os.getenv("WEAVIATE_URL")
    weaviate_api_key = os.getenv("WEAVIATE_API_KEY")

    client = None
    if weaviate_url and weaviate_api_key and "xxx" not in weaviate_url:
        url = weaviate_url.strip('"\'')
        if not url.startswith("http://") and not url.startswith("https://"):
            url = f"https://{url}"
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=url,
            auth_credentials=weaviate.auth.AuthApiKey(weaviate_api_key.strip('"\''))
        )
    else:
        try:
            client = weaviate.connect_to_local()
            if not client.is_ready():
                raise ConnectionError("Local Weaviate not responding")
        except Exception as e:
            print(f"Local Weaviate not available: {e}. Falling back to embedded Weaviate...")
            client = weaviate.connect_to_embedded()

    try:
        collection_name = "DrugLawDocs"
        if not client.collections.exists(collection_name):
            print(f"Collection '{collection_name}' does not exist.")
            return []

        collection = client.collections.get(collection_name)

        # Query vector similarity
        results = collection.query.near_vector(
            near_vector=query_embedding,
            limit=top_k,
            return_metadata=MetadataQuery(distance=True)
        )

        output = []
        for obj in results.objects:
            meta = {
                "source": obj.properties.get("source", ""),
                "doc_type": obj.properties.get("doc_type", ""),
                "chunk_index": int(obj.properties.get("chunk_index", 0)) if obj.properties.get("chunk_index") is not None else 0
            }
            # Phục hồi headers nếu có
            for header in ["header_1", "header_2", "header_3"]:
                val = obj.properties.get(header)
                if val:
                    meta[header.replace("header_", "Header_")] = val

            # Cosine similarity score xấp xỉ = 1 - distance
            distance = obj.metadata.distance if obj.metadata.distance is not None else 1.0
            score = 1.0 - distance

            output.append({
                "content": obj.properties.get("content", ""),
                "score": float(score),
                "metadata": meta
            })

        # Sắp xếp giảm dần theo điểm số (Weaviate near_vector tự động sắp xếp giảm dần theo độ tương đồng)
        output = sorted(output, key=lambda x: x["score"], reverse=True)
        return output
    finally:
        client.close()


if __name__ == "__main__":
    # Test
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
