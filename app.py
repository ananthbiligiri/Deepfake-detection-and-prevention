import streamlit as st
import tempfile
import cv2
import os
from model import VideoDeepfakeDetector

st.set_page_config(layout="wide")
st.title("🎬 Deepfake Video Detector")
st.write("Upload a video to detect deepfakes, watermark it, and optionally receive an email alert.")

email = st.text_input("Alert Email", "user@example.com")
enable_email = st.checkbox("Enable Email Alerts", True)
uploaded_file = st.file_uploader("Upload Video", type=["mp4", "mov"])

model_path = 'C:\\deepfake.com\\video_deepfake_detector.h5'
detector = VideoDeepfakeDetector(model_path)

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    st_frame = st.empty()
    cap = cv2.VideoCapture(video_path)
    output_frames = []

    prediction = detector.predict_video(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640, 360))
        if prediction > 0.5:
            frame = detector.add_watermark(frame)
            output_frames.append(frame)
        st_frame.image(frame, channels="BGR", use_container_width=True)

    cap.release()

    st.write(f"Prediction: **{'FAKE' if prediction > 0.5 else 'REAL'}** (Confidence: {abs(prediction - 0.5) * 200:.1f}%)")

    if prediction > 0.5 and output_frames:
        output_path = "watermarked_output.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 30, (640, 360))
        for f in output_frames:
            out.write(f)
        out.release()

        st.video(output_path)

        if enable_email:
            with st.spinner("Sending email alert..."):
                try:
                    detector.send_alert(email, output_path)
                    st.success("Email alert sent successfully!")
                except Exception as e:
                    st.error(f"Failed to send email: {e}")

        os.remove(output_path)

    os.remove(video_path)
