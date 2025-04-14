# 🎭 Deepfake Video Detector with Email Alerts and Watermarking

This project detects deepfake videos using a custom-trained 3D CNN model. It includes a user-friendly Streamlit app that allows you to upload a video, get real-time predictions, watermark the video if it's fake, and even send an email alert with the watermarked video attached.

---

## 🚀 Features

- 🧠 Deepfake detection using a 3D CNN model trained on video clips.
- 🎬 Streamlit-based web interface for easy video upload and analysis.
- 🏷️ Adds a **"DEEPFAKE DETECTED"** watermark to fake videos.
- 📩 Sends email alerts with the watermarked video attached.
- 📊 Displays prediction result with confidence level.

---

## 📦 Requirements

Install all necessary dependencies:

```bash
pip install tensorflow opencv-python scikit-learn streamlit keras numpy
```
##FileStructure
deepfake-detector/
├── app.py                      # Streamlit frontend
├── model.py                    # Deepfake detection class (training, prediction, email)
├── video_deepfake_detector.h5 # Trained deepfake detection model
├── dataset/
│   ├── real/                   # Real training videos
│   └── fake/                   # Fake training videos
└── README.md
