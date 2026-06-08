import streamlit as st

# Custom CSS cho phong cách giao diện tối hiện đại, Glassmorphism và Google Fonts
CUSTOM_CSS = """
<style>
    /* Nhập Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@300;400;600&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #090d16 0%, #15102a 100%);
        color: #f1f5f9;
        font-family: 'Outfit', sans-serif;
    }
    
    .app-title {
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8 0%, #a78bfa 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.2rem;
        margin-bottom: 0.1rem;
        text-align: center;
        text-shadow: 0 10px 30px rgba(167, 139, 250, 0.15);
    }
    
    .app-subtitle {
        font-family: 'Outfit', sans-serif;
        font-weight: 400;
        font-size: 1.15rem;
        color: #94a3b8;
        text-align: center;
        margin-bottom: 2.2rem;
    }

    section[data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.95) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(10px);
    }

    /* Thẻ tài liệu tham khảo với phong cách Glassmorphism */
    .source-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .source-card:hover {
        transform: translateY(-4px) scale(1.01);
        border-color: rgba(56, 189, 248, 0.5);
        box-shadow: 0 12px 40px 0 rgba(56, 189, 248, 0.15);
        background: rgba(255, 255, 255, 0.05);
    }

    .source-header {
        font-size: 0.95rem;
        font-weight: 600;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    .source-title {
        color: #f8fafc;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    .source-body {
        font-size: 0.88rem;
        color: #cbd5e1;
        line-height: 1.55;
    }

    /* Badges nhãn tài liệu */
    .badge {
        padding: 2px 8px;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    
    .badge-legal {
        background: rgba(56, 189, 248, 0.12);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.25);
    }
    
    .badge-news {
        background: rgba(236, 72, 153, 0.12);
        color: #ec4899;
        border: 1px solid rgba(236, 72, 153, 0.25);
    }

    /* Bảng trạng thái Sidebar */
    .status-dashboard {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 12px;
        margin-bottom: 20px;
    }
    
    .status-item {
        display: flex;
        justify-content: space-between;
        font-size: 0.8rem;
        margin-bottom: 8px;
        color: #94a3b8;
    }
    
    .status-item:last-child {
        margin-bottom: 0;
    }
    
    .status-value {
        font-weight: 600;
        color: #38bdf8;
    }
    
    .status-online {
        color: #10b981;
        font-weight: 600;
    }
</style>
"""

def apply_custom_css():
    """
    Tiêm toàn bộ các thiết lập CSS tùy chỉnh vào trang Streamlit.
    """
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_status_dashboard(db_status: str = "Online", reranker: str = "Local (mxbai-xs)", fallback: str = "Bypassed"):
    """
    Trả về chuỗi HTML của bảng chỉ báo trạng thái hệ thống trên Sidebar.
    """
    status_color = "#10b981" if db_status == "Online" else "#ef4444"
    return f"""
    <div class="status-dashboard">
        <div class="status-item">
            <span>CSDL Weaviate:</span>
            <span style="color: {status_color}; font-weight: 600;">● {db_status}</span>
        </div>
        <div class="status-item">
            <span>Reranker:</span>
            <span class="status-value">{reranker}</span>
        </div>
        <div class="status-item">
            <span>PageIndex Fallback:</span>
            <span class="status-value" style="color: #94a3b8;">{fallback}</span>
        </div>
    </div>
    """


def render_source_card_html(source_name: str, doc_type: str, content_preview: str) -> str:
    """
    Trả về mã HTML cho một thẻ tài liệu tham chiếu với phong cách Glassmorphism.
    """
    badge_class = "badge-legal" if doc_type == "legal" else "badge-news"
    # Giới hạn nội dung hiển thị tối đa
    preview = content_preview[:200] + "..." if len(content_preview) > 200 else content_preview
    
    return f"""
    <div class='source-card'>
        <div class='source-header'>
            <span class='source-title'>📄 {source_name}</span>
            <span class='badge {badge_class}'>{doc_type}</span>
        </div>
        <div class='source-body'>{preview}</div>
    </div>
    """
