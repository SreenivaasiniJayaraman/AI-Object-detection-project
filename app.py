import streamlit as st
import cv2
from ultralytics import YOLO

st.set_page_config(page_title="AI Object Detection", layout="wide")

st.title("🧠 Real-Time Object Detection using YOLOv8")

# Initialize session state
if "run" not in st.session_state:
    st.session_state.run = False

# Sidebar
st.sidebar.header("Controls")

if st.sidebar.button("▶ Start Detection"):
    st.session_state.run = True

if st.sidebar.button("⏹ Stop Detection"):
    st.session_state.run = False

confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)

FRAME_WINDOW = st.image([])

model = YOLO("yolov8n.pt")

camera = cv2.VideoCapture(0)

if st.session_state.run:
    while st.session_state.run:
        ret, frame = camera.read()
        if not ret:
            st.error("Webcam error")
            break

        results = model(frame, conf=confidence)
        annotated_frame = results[0].plot()

        FRAME_WINDOW.image(annotated_frame, channels="BGR")

camera.release()