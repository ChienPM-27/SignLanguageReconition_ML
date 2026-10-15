"""Streamlit Web Application for Isolated Sign Language Recognition (ISLR).

Run with:
    streamlit run app.py
"""
from __future__ import annotations

import os
from pathlib import Path
import tempfile
import time

import pandas as pd
import streamlit as st

# Set page config FIRST before any other streamlit call
st.set_page_config(
    page_title="ISLR - Sign Language Recognition",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.inference.predictor import ISLRPredictor

# Custom CSS for modern styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F3F4F6;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #3B82F6;
        margin-bottom: 10px;
    }
    .prediction-box {
        background: #EFF6FF;
        border: 2px solid #3B82F6;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-top: 15px;
    }
    .prediction-title {
        font-size: 1.8rem;
        font-weight: 800;
        color: #1D4ED8;
    }
    .confidence-badge {
        font-size: 1.2rem;
        color: #059669;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Đang nạp mô hình PyTorch GRU...")
def get_predictor() -> ISLRPredictor:
    """Initialize and cache predictor instance."""
    return ISLRPredictor()


def main():
    st.markdown('<div class="main-title">🤟 Isolated Sign Language Recognition (ISLR)</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Hệ thống nhận dạng 20 từ Ngôn ngữ Ký hiệu Mỹ (ASL) '
        'dựa trên MediaPipe Hands và Mô hình Deep Learning GRU (Top-1: <b>81.07%</b> | Top-3: <b>95.38%</b>)</div>',
        unsafe_allow_html=True,
    )

    # Initialize predictor
    try:
        predictor = get_predictor()
    except Exception as e:
        st.error(f"Lỗi nạp mô hình: {e}. Vui lòng kiểm tra lại file checkpoint best_model.pt.")
        return

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Cấu hình Hệ thống")
        st.success("✅ Mô hình: GRU Classifier (Đã nạp)")
        st.info(f"**Số lượng nhãn**: {predictor.num_classes} từ vựng\n\n"
                f"**Checkpoint**: `{predictor.model_path.name}`\n\n"
                f"**Độ dài chuẩn hóa**: $T=32$ frames ($D=143$)")

        st.markdown("---")
        st.subheader("💡 Mẹo ký hiệu chính xác")
        st.markdown(
            "- Giữ tay trong khung hình của camera.\n"
            "- Ký dứt khoát trong vòng 1-2 giây.\n"
            "- Ánh sáng rõ ràng để MediaPipe bắt đủ 21 khớp ngón tay."
        )

        st.markdown("---")
        st.caption("Dự án Sign Language Recognition ML - Phase 3 UI Demo")

    # Tabs
    tab1, tab2, tab3 = st.tabs([
        "📹 Nhận diện từ Video File",
        "📖 Từ điển 20 Ký hiệu",
        "📊 Báo cáo & Đánh giá Hiệu năng",
    ])

    # TAB 1: Video Recognition
    with tab1:
        st.subheader("Nhận diện cử chỉ ký hiệu từ Video")
        
        sample_dir = Path("data/sample_videos")
        has_samples = sample_dir.is_dir() and any(sample_dir.glob("*.mp4"))

        input_mode = st.radio(
            "Nguồn dữ liệu video:",
            ["🎯 Chọn video mẫu có sẵn (20 từ ASL)", "📁 Tải lên video của bạn (.mp4, .mov, .avi)"],
            horizontal=True,
        )

        video_path = None
        is_temp = False

        gloss_info = {
            "awake": ("🌅 awake", "Tỉnh táo / Thức giấc"),
            "bird": ("🐦 bird", "Con chim"),
            "brown": ("🟤 brown", "Màu nâu"),
            "bye": ("👋 bye", "Tạm biệt"),
            "cow": ("🐮 cow", "Con bò"),
            "doll": ("🪆 doll", "Búp bê"),
            "donkey": ("🫏 donkey", "Con lừa"),
            "drink": ("🥤 drink", "Uống nước"),
            "duck": ("🦆 duck", "Con vịt"),
            "fireman": ("🧑‍🚒 fireman", "Lính cứu hỏa"),
            "lips": ("👄 lips", "Đôi môi"),
            "listen": ("👂 listen", "Lắng nghe"),
            "mouse": ("🐭 mouse", "Con chuột"),
            "nuts": ("🥜 nuts", "Hạt đậu phộng"),
            "pretend": ("🎭 pretend", "Giả vờ"),
            "shhh": ("🤫 shhh", "Yên lặng"),
            "sleepy": ("🥱 sleepy", "Buồn ngủ"),
            "uncle": ("🧔 uncle", "Người chú / Bác"),
            "wake": ("⏰ wake", "Thức dậy"),
            "who": ("❓ who", "Ai / Người nào"),
        }

        if input_mode == "🎯 Chọn video mẫu có sẵn (20 từ ASL)":
            if has_samples:
                available_samples = sorted(list(sample_dir.glob("*.mp4")), key=lambda p: p.stem)
                sample_options = {}
                for p in available_samples:
                    tag, meaning = gloss_info.get(p.stem, (f"✨ {p.stem}", ""))
                    label = f"{tag} — {meaning}" if meaning else tag
                    sample_options[label] = p

                selected_label = st.selectbox(
                    "Chọn video mẫu từ 20 từ vựng mục tiêu:",
                    options=list(sample_options.keys()),
                    index=0,
                )
                video_path = str(sample_options[selected_label])
                is_temp = False
            else:
                st.warning("Chưa tìm thấy thư mục video mẫu `data/sample_videos`. Vui lòng chạy `python scripts/download_sample_videos.py` để tải về.")
        else:
            uploaded_file = st.file_uploader(
                "Kéo thả hoặc chọn file video từ máy tính:",
                type=["mp4", "mov", "avi", "webm"],
            )
            if uploaded_file is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    video_path = tmp_file.name
                    is_temp = True

        if video_path is not None:
            col_left, col_right = st.columns([1, 1])

            with col_left:
                st.video(video_path)
                st.caption(f"Đường dẫn file: `{Path(video_path).name}`")

            with col_right:
                if st.button("🚀 Bắt đầu Nhận diện", type="primary", use_container_width=True):
                    with st.spinner("Đang trích xuất MediaPipe landmarks và suy luận qua GRU..."):
                        t_start = time.time()
                        try:
                            result = predictor.predict_video(video_path, top_k=5)
                            elapsed = time.time() - t_start

                            st.markdown(
                                f"""
                                <div class="prediction-box">
                                    <div style="font-size: 0.95rem; color: #6B7280; text-transform: uppercase; letter-spacing: 1px;">Từ ký hiệu dự đoán (Top-1)</div>
                                    <div class="prediction-title">"{result.predicted_gloss.upper()}"</div>
                                    <div class="confidence-badge">Độ tin cậy: {result.confidence * 100:.1f}%</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                            st.markdown("#### 🎯 Xác suất Top-5 Dự đoán")
                            top5_df = pd.DataFrame(
                                result.top_k,
                                columns=["Từ ký hiệu (Gloss)", "Xác suất"],
                            )
                            top5_df["Xác suất (%)"] = (top5_df["Xác suất"] * 100).round(2)

                            st.bar_chart(
                                top5_df.set_index("Từ ký hiệu (Gloss)")["Xác suất (%)"],
                                color="#2563EB",
                            )

                            st.markdown("#### ⏱️ Thông số Kỹ thuật")
                            st.markdown(
                                f"""
                                - **Thời gian suy luận**: `{elapsed:.3f} giây`
                                - **Tổng số khung hình video**: `{result.num_frames} frames`
                                - **Tỷ lệ nhận diện bàn tay**: `{result.hand_detection_rate * 100:.1f}%`
                                """
                            )

                        except Exception as err:
                            st.error(f"Lỗi trong quá trình xử lý video: {err}")
                        finally:
                            if is_temp and os.path.exists(video_path):
                                os.unlink(video_path)

    # TAB 2: Sign Dictionary
    with tab2:
        st.subheader("Từ điển 20 Ký hiệu Mục tiêu (Target Vocabulary)")
        st.write("Danh sách 20 từ được chọn lọc bài bản từ cuộc thi Google ISLR dựa trên phân bổ người ký độc lập:")

        glosses_data = []
        gloss_meanings = {
            "awake": "Tỉnh táo / Thức giấc",
            "bird": "Con chim",
            "brown": "Màu nâu",
            "bye": "Tạm biệt",
            "cow": "Con bò",
            "doll": "Búp bê",
            "donkey": "Con lừa",
            "drink": "Uống nước",
            "duck": "Con vịt",
            "fireman": "Lính cứu hỏa",
            "lips": "Đôi môi",
            "listen": "Lắng nghe",
            "mouse": "Con chuột",
            "nuts": "Hạt đậu phộng",
            "pretend": "Giả vờ",
            "shhh": "Yên lặng / Giữ im lặng",
            "sleepy": "Buồn ngủ",
            "uncle": "Người chú / Bác",
            "wake": "Thức dậy",
            "who": "Ai / Người nào",
        }

        for label in sorted(predictor.label_to_gloss.keys()):
            gloss = predictor.label_to_gloss[label]
            glosses_data.append({
                "Mã (ID)": label,
                "Từ tiếng Anh (Gloss)": gloss.upper(),
                "Ý nghĩa tiếng Việt": gloss_meanings.get(gloss, "Chưa cập nhật"),
            })

        st.dataframe(pd.DataFrame(glosses_data), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("🎥 Xem Video Thị phạm từng Ký hiệu")
        col_sel, col_vid = st.columns([1, 1])
        with col_sel:
            preview_gloss = st.selectbox(
                "Chọn từ muốn xem video mẫu:",
                options=[g["Từ tiếng Anh (Gloss)"].lower() for g in glosses_data],
            )
            meaning_str = gloss_meanings.get(preview_gloss, "")
            st.info(f"**Cử chỉ**: `{preview_gloss.upper()}`\n\n**Ý nghĩa**: {meaning_str}")
        with col_vid:
            vid_p = Path(f"data/sample_videos/{preview_gloss}.mp4")
            if vid_p.is_file():
                st.video(str(vid_p))
            else:
                st.info("Chưa có video minh họa cho từ này.")

    # TAB 3: Model Performance
    with tab3:
        st.subheader("Báo cáo Hiệu năng Huấn luyện (Phase 2)")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Top-1 Accuracy", "81.07%", "+7.36% vs RF")
        col2.metric("Top-3 Accuracy", "95.38%", "+2.39% vs RF")
        col3.metric("Macro-F1 Score", "0.8028", "+0.0757 vs RF")
        col4.metric("Kích thước Mô hình", "828 KB", "Ultra-lightweight")

        st.markdown("---")
        st.subheader("Biểu đồ Huấn luyện & Ma trận Nhầm lẫn")

        plot_paths = [
            Path("C:/Users/Lionel Pham/.gemini/antigravity/brain/acec4975-4800-4634-a223-9ad69412cd2c/learning_curves_seed42.png"),
            Path("outputs/Kaggle_test/islr_phase2_artifacts_15.10.2026/outputs/runs/20261015_032408_gru_baseline/seed_42/learning_curves.png"),
        ]
        cm_paths = [
            Path("C:/Users/Lionel Pham/.gemini/antigravity/brain/acec4975-4800-4634-a223-9ad69412cd2c/confusion_matrix_seed42.png"),
            Path("outputs/Kaggle_test/islr_phase2_artifacts_15.10.2026/outputs/runs/20261015_032408_gru_baseline/seed_42/confusion_matrix.png"),
        ]

        col_img1, col_img2 = st.columns(2)
        with col_img1:
            st.markdown("#### Learning Curves (Seed 42)")
            for p in plot_paths:
                if p.is_file():
                    st.image(str(p), use_container_width=True)
                    break

        with col_img2:
            st.markdown("#### Confusion Matrix (Seed 42)")
            for p in cm_paths:
                if p.is_file():
                    st.image(str(p), use_container_width=True)
                    break


if __name__ == "__main__":
    main()
