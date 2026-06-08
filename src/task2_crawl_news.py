"""
Task 2 — Crawl bài báo về nghệ sĩ liên quan tới ma tuý.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài báo từ các trang tin tức Việt Nam.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# TODO: Điền danh sách URL bài báo cần crawl
ARTICLE_URLS = [
    "https://vnexpress.net/dem-su-dung-ma-tuy-cuong-loan-cua-ca-si-chau-viet-cuong-3863999.html",
    "https://vnexpress.net/ca-si-miu-le-chua-bi-khoi-to-trong-vu-an-dung-ma-tuy-o-dao-cat-ba-5074052.html",
    "https://vnexpress.net/ca-si-long-nhat-son-ngoc-minh-bi-bat-vi-lien-quan-ma-tuy-5060857.html",
    "https://vnexpress.net/nguoi-mau-andrea-aybar-cung-tro-ly-lam-tiec-ma-tuy-trong-can-ho-cao-cap-5059429.html",
    "https://vnexpress.net/anh-em-ca-si-chi-dan-ru-nhieu-nguoi-choi-ma-tuy-nhu-the-nao-4929804.html"
]


async def crawl_article(url: str) -> dict:
    """
    Crawl một bài báo và trả về dict chứa metadata + content.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str (ISO format),
            "content_markdown": str
        }
    """
    # Thử crawl bằng crawl4ai trước
    try:
        print("  -> Đang thử crawl bằng crawl4ai...")
        from crawl4ai import AsyncWebCrawler
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
            if result and result.success:
                title = result.metadata.get("title") if result.metadata else None
                if isinstance(title, dict):
                    title = title.get("title")
                if not title or title == "Unknown":
                    # Thử phân tích HTML để lấy tiêu đề
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(result.html, "html.parser")
                    title_tag = soup.find("h1") or soup.find("title")
                    title = title_tag.get_text().strip() if title_tag else "Unknown"
                
                content_markdown = result.markdown or ""
                if len(content_markdown.strip()) > 100:
                    return {
                        "url": url,
                        "title": str(title),
                        "date_crawled": datetime.now().isoformat(),
                        "content_markdown": content_markdown,
                    }
    except Exception as e:
        print(f"  -> crawl4ai thất bại hoặc chưa được khởi tạo đầy đủ: {e}")

    # Phương án dự phòng: Sử dụng requests + BeautifulSoup + Markdownify
    print("  -> Đang sử dụng phương án dự phòng requests + BeautifulSoup...")
    import requests
    from bs4 import BeautifulSoup
    import markdownify

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    loop = asyncio.get_event_loop()
    try:
        response = await loop.run_in_executor(None, lambda: requests.get(url, headers=headers, timeout=15))
        response.raise_for_status()
        response.encoding = response.apparent_encoding or "utf-8"
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Trích xuất tiêu đề (áp dụng riêng cho VnExpress hoặc chung cho các trang khác)
        title = ""
        title_tag = soup.find("h1", class_="title-detail") or soup.find("h1", class_="title_detail") or soup.find("h1") or soup.find("title")
        if title_tag:
            title = title_tag.get_text().strip()
        if not title:
            title = "Unknown Title"
            
        # Loại bỏ các thẻ không cần thiết
        for element in soup(["script", "style", "iframe", "video", "audio", "noscript"]):
            element.decompose()
            
        # Trích xuất nội dung chính
        content_div = soup.find("article", class_="fck_detail") or soup.find(class_="fck_detail") or soup.find("article")
        if not content_div:
            content_div = soup.find("body")
            
        content_html = str(content_div)
        content_markdown = markdownify.markdownify(content_html, heading_style="ATX").strip()
        
        return {
            "url": url,
            "title": title,
            "date_crawled": datetime.now().isoformat(),
            "content_markdown": content_markdown,
        }
    except Exception as e:
        print(f"  -> Phương án dự phòng cũng thất bại: {e}")
        # Trả về cấu trúc tối thiểu để tránh làm sập pipeline
        return {
            "url": url,
            "title": "Failed to crawl",
            "date_crawled": datetime.now().isoformat(),
            "content_markdown": f"Failed to crawl content from {url} due to error: {e}",
        }


async def crawl_all():
    """Crawl toàn bộ bài báo trong ARTICLE_URLS."""
    setup_directory()

    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        article = await crawl_article(url)

        # Lưu file JSON
        filename = f"article_{i:02d}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2))
        print(f"  ✓ Saved: {filepath}")


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("⚠ Hãy điền ARTICLE_URLS trước khi chạy!")
        print("Gợi ý: tìm bài báo trên VnExpress, Tuổi Trẻ, Thanh Niên, ...")
    else:
        asyncio.run(crawl_all())
