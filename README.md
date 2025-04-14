# 🎭 Deepfake Video Detector with Watermarking

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
## 🧠 How It Works
- A 3D CNN processes 16-frame video clips resized to 112x112.
- The model predicts whether the clip is real or fake.
- Predictions across all clips in a video are averaged.
- If fake, the app:
- Watermarks the video.
- Sends an email alert with the watermarked video (if enabled).
- Real-time results and confidence scores are shown in the app.

   Run the Streamlit App
```bash
streamlit run app.py
```
## Upload and Analyze
- Upload a .mp4 or .mov video.
- The app will process and display prediction.
- If a deepfake is detected:
- A watermark will be added to the video.
- An email will be sent (if enabled).
## 🛠️ Customization
- Watermark text: Change in add_watermark() function in model.py.
- Video source: Modify cv2.VideoCapture() to use webcam or another file.
- Model path: Replace model_path in app.py and model.py with your own .h5 model file.
## 🖼️ Example Output
Here’s how the watermarked result looks when a deepfake is detected:
![Image](https://github.com/user-attachments/assets/063b5f23-03bb-410d-9b1b-f0f617563130)
![Image](https://github.com/user-attachments/assets/833f0435-abea-4f50-b0bf-0d472b3ba439)

## 📬 Feedback & Contributions
- Feel free to open issues, submit pull requests, or fork the repo to improve it. Let’s fight deepfakes together!



  
