import os
import cv2
import numpy as np
from keras.models import load_model
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication

class VideoDeepfakeDetector:
    def __init__(self, model_path='C:\\deepfake.com\\video_deepfake_detector.h5'):
        if not os.path.exists(model_path):
            raise FileNotFoundError("Trained model not found.")
        self.model = load_model(model_path)

    def predict_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (112, 112))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame.astype("float32") / 255.0)

        cap.release()

        if len(frames) < 16:
            return 0.0

        sequences = [frames[i:i+16] for i in range(0, len(frames) - 15, 16)]
        y_pred = self.model.predict(np.array(sequences))
        return float(np.mean(y_pred))

    def add_watermark(self, frame, text='DEEPFAKE DETECTED'):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.putText(overlay, text, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        return overlay

    def send_alert(self, to_email, video_path):
        from_email = 'your_email@example.com'
        password = 'your_password'

        msg = MIMEMultipart()
        msg['From'] = from_email
        msg['To'] = to_email
        msg['Subject'] = '⚠️ Deepfake Alert: Fake Video Detected'

        msg.attach(MIMEText("A fake video was detected. See the attached file.", 'plain'))
        with open(video_path, 'rb') as f:
            part = MIMEApplication(f.read(), Name=os.path.basename(video_path))
        part['Content-Disposition'] = f'attachment; filename="{os.path.basename(video_path)}"'
        msg.attach(part)

        try:
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(from_email, password)
            server.send_message(msg)
            server.quit()
        except Exception as e:
            raise RuntimeError(f"Failed to send email: {e}")
