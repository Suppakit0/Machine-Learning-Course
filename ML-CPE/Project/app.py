
import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
from pathlib import Path

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="EcoSort AI | Smart Waste Classification",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

MODEL_PATH = "models/waste_cnn_model.keras"
CLASS_PATH = "models/class_names.json"
IMAGE_SIZE = (180, 180)

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Prompt', sans-serif;
}

.stApp {
    background: linear-gradient(180deg, #f0fdf4 0%, #f8fafc 45%, #ffffff 100%);
}

.block-container {
    max-width: 1150px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

#MainMenu, footer, header {
    visibility: hidden;
}

.hero {
    background: linear-gradient(120deg, #064e3b, #059669);
    padding: 35px;
    border-radius: 24px;
    color: white;
    margin-bottom: 28px;
    box-shadow: 0 12px 30px rgba(5, 150, 105, 0.18);
}

.hero h1 {
    font-size: 36px;
    font-weight: 700;
    margin: 0;
    color: white;
}

.hero p {
    font-size: 16px;
    color: #d1fae5;
    margin-top: 10px;
}

.hero-tag {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.25);
    padding: 6px 14px;
    border-radius: 30px;
    font-size: 13px;
    margin-bottom: 15px;
}

.section-title {
    font-size: 21px;
    font-weight: 600;
    color: #064e3b;
    margin-bottom: 12px;
}

.panel {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 20px;
    padding: 22px;
    box-shadow: 0 5px 18px rgba(15, 23, 42, 0.04);
    height: 100%;
}

.result-card {
    background: linear-gradient(135deg, #ecfdf5, #d1fae5);
    border: 1px solid #a7f3d0;
    border-radius: 20px;
    padding: 24px;
    text-align: center;
}

.result-label {
    color: #047857;
    font-size: 14px;
    font-weight: 500;
}

.result-name {
    color: #064e3b;
    font-size: 32px;
    font-weight: 700;
    margin: 8px 0;
}

.result-en {
    color: #64748b;
    font-size: 14px;
}

.confidence {
    color: #059669;
    font-size: 38px;
    font-weight: 700;
    margin-top: 12px;
}

.info-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 17px;
    margin-top: 14px;
}

.info-title {
    color: #065f46;
    font-weight: 600;
    margin-bottom: 6px;
}

.info-text {
    color: #475569;
    font-size: 14px;
    line-height: 1.7;
}

.upload-hint {
    color: #64748b;
    font-size: 13px;
    text-align: center;
    margin-top: 8px;
}

div[data-testid="stFileUploader"] {
    background: #f8fafc;
    border: 2px dashed #86efac;
    border-radius: 16px;
    padding: 15px;
}

.stButton button {
    background: #059669;
    color: white;
    border: none;
    border-radius: 12px;
    padding: 10px 20px;
}

.stButton button:hover {
    background: #047857;
    color: white;
}

.footer {
    text-align: center;
    color: #94a3b8;
    font-size: 12px;
    padding-top: 35px;
}
</style>
""", unsafe_allow_html=True)


# -----------------------------
# CLASS INFORMATION
# -----------------------------
CLASS_INFO = {
    "cardboard": {
        "th": "กระดาษลัง",
        "advice": "พับให้แบนและแยกออกจากขยะเปียก เพื่อให้นำไปรีไซเคิลได้ง่าย"
    },
    "glass": {
        "th": "แก้ว",
        "advice": "แยกขวดแก้วออกจากขยะประเภทอื่น และระมัดระวังการแตก"
    },
    "metal": {
        "th": "โลหะ",
        "advice": "เทของเหลวออก ทำความสะอาด และแยกกระป๋องหรือโลหะเพื่อรีไซเคิล"
    },
    "paper": {
        "th": "กระดาษ",
        "advice": "เก็บกระดาษให้แห้ง ไม่ปนเปื้อนอาหาร และแยกไว้สำหรับรีไซเคิล"
    },
    "plastic": {
        "th": "พลาสติก",
        "advice": "เทของเหลวออกและแยกตามข้อกำหนดในพื้นที่"
    },
    "trash": {
        "th": "ขยะทั่วไป",
        "advice": "ทิ้งลงถังขยะทั่วไป และตรวจสอบว่าไม่สามารถนำกลับมาใช้หรือรีไซเคิลได้"
    }
}


# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def load_class_names():
    with open(CLASS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        # รองรับ {"0": "cardboard", "1": "glass", ...}
        if all(str(k).isdigit() for k in data.keys()):
            return [data[k] for k in sorted(data, key=lambda x: int(x))]

        # รองรับ {"cardboard": 0, "glass": 1, ...}
        if all(isinstance(v, int) for v in data.values()):
            return [
                k for k, v in sorted(data.items(), key=lambda item: item[1])
            ]

    raise ValueError("รูปแบบ class_names.json ไม่ถูกต้อง")


def get_class_info(class_name):
    key = str(class_name).lower().strip()
    return CLASS_INFO.get(
        key,
        {"th": str(class_name).capitalize(),
         "advice": "โปรดตรวจสอบประเภทขยะก่อนนำไปทิ้ง"}
    )


# -----------------------------
# HEADER
# -----------------------------
st.markdown("""
<div class="hero">
    <div class="hero-tag">♻️ AI POWERED WASTE CLASSIFICATION</div>
    <h1>EcoSort AI</h1>
    <p>ระบบจำแนกประเภทขยะอัจฉริยะด้วยปัญญาประดิษฐ์</p>
    <p>อัปโหลดรูปภาพ แล้วให้ AI ช่วยวิเคราะห์ประเภทขยะของคุณ</p>
</div>
""", unsafe_allow_html=True)


# -----------------------------
# LOAD MODEL AND CLASSES
# -----------------------------
try:
    model = load_model()
    class_names = load_class_names()
except Exception as e:
    st.error(f"ไม่สามารถโหลดโมเดลหรือข้อมูลคลาสได้: {e}")
    st.stop()


# -----------------------------
# UPLOAD SECTION
# -----------------------------
st.markdown('<div class="section-title">📸 วิเคราะห์ประเภทขยะ</div>',
            unsafe_allow_html=True)

left, right = st.columns([1, 1.15], gap="large")

with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("### 📤 อัปโหลดรูปภาพ")
    st.write("เลือกรูปภาพขยะที่ต้องการให้ AI วิเคราะห์")

    uploaded_file = st.file_uploader(
        "เลือกไฟล์รูปภาพ",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    st.markdown(
        '<div class="upload-hint">รองรับไฟล์ JPG, JPEG และ PNG</div>',
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("### 🖼️ ภาพตัวอย่าง")

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_container_width=True)
        st.caption(f"ชื่อไฟล์: {uploaded_file.name}")
    else:
        st.markdown("""
        <div style="
            height: 240px;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-direction: column;
            background: #f8fafc;
            border: 2px dashed #cbd5e1;
            border-radius: 15px;
            color: #94a3b8;">
            <div style="font-size: 55px;">🌱</div>
            <div>รูปภาพของคุณจะแสดงที่นี่</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# -----------------------------
# PREDICTION
# -----------------------------
if uploaded_file is not None:

    st.markdown("---")

    try:
        resized_image = image.resize(IMAGE_SIZE)
        img_array = np.array(resized_image, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array, verbose=0)[0]

        if len(predictions) != len(class_names):
            st.error("จำนวนคลาสของโมเดลไม่ตรงกับ class_names.json")
            st.stop()

        predicted_index = int(np.argmax(predictions))
        predicted_class = class_names[predicted_index]
        confidence = float(predictions[predicted_index]) * 100

        info = get_class_info(predicted_class)

        # RESULT
        st.markdown(
            '<div class="section-title">✨ ผลการวิเคราะห์</div>',
            unsafe_allow_html=True
        )

        result_col, advice_col = st.columns([1, 1], gap="large")

        with result_col:
            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">ประเภทขยะที่ตรวจพบ</div>
                <div class="result-name">{info["th"]}</div>
                <div class="result-en">{str(predicted_class).capitalize()}</div>
                <div class="confidence">{confidence:.2f}%</div>
                <div class="result-label">ระดับความมั่นใจของ AI</div>
            </div>
            """, unsafe_allow_html=True)

        with advice_col:
            st.markdown(f"""
            <div class="info-card" style="height: 100%;">
                <div class="info-title">💡 คำแนะนำในการจัดการขยะ</div>
                <div class="info-text">{info["advice"]}</div>
                <hr style="border: 0; border-top: 1px solid #e2e8f0; margin: 18px 0;">
                <div class="info-title">🤖 เกี่ยวกับผลการวิเคราะห์</div>
                <div class="info-text">
                    AI เลือกประเภทที่มีคะแนนความน่าจะเป็นสูงที่สุด
                    ผลลัพธ์เป็นการคาดการณ์จากภาพ ไม่ใช่การยืนยันชนิดวัสดุ
                </div>
            </div>
            """, unsafe_allow_html=True)

        # CONFIDENCE WARNING
        if confidence < 60:
            st.warning(
                "ความมั่นใจค่อนข้างต่ำ แนะนำให้ถ่ายภาพใหม่ "
                "โดยให้เห็นวัตถุชัดเจนและมีแสงสว่างเพียงพอ"
            )

        # PROBABILITY
        st.markdown("---")
        st.markdown(
            '<div class="section-title">📊 คะแนนความน่าจะเป็นของแต่ละประเภท</div>',
            unsafe_allow_html=True
        )

        sorted_indices = np.argsort(predictions)[::-1]

        for idx in sorted_indices:
            class_name = class_names[int(idx)]
            class_info = get_class_info(class_name)
            score = float(predictions[idx]) * 100

            col_name, col_score = st.columns([4, 1])

            with col_name:
                st.markdown(
                    f"**{class_info['th']}** "
                    f"<span style='color:#94a3b8;'>({str(class_name).capitalize()})</span>",
                    unsafe_allow_html=True
                )
                st.progress(min(max(score / 100, 0.0), 1.0))

            with col_score:
                st.markdown(
                    f"<div style='text-align:right; color:#059669; "
                    f"font-weight:700; padding-top:3px;'>{score:.2f}%</div>",
                    unsafe_allow_html=True
                )

        st.caption(
            "หมายเหตุ: คะแนนความน่าจะเป็นเป็นค่าที่โมเดลคำนวณ "
            "ไม่ใช่ค่าความแม่นยำของโมเดลโดยรวม"
        )

    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการวิเคราะห์ภาพ: {e}")


# -----------------------------
# FOOTER
# -----------------------------
st.markdown("""
<div class="footer">
    ♻️ EcoSort AI | Smart Waste Classification<br>
    Powered by TensorFlow & Convolutional Neural Network
</div>
""", unsafe_allow_html=True)
