import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

class VideoDeepfakeDetector:
    def __init__(self, model_path='C:\\deepfake.com\\video_deepfake_detector.h5'):
        self.clip_length = 16
        if os.path.exists(model_path):
            self.model = load_model(model_path)
        else:
            self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Conv3D(32, kernel_size=(3, 3, 3), activation='relu', input_shape=(self.clip_length, 112, 112, 3)))
        model.add(MaxPooling3D(pool_size=(2, 2, 2)))
        model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
        return model

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        clips = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (112, 112))
            clips.append(frame)

            if len(clips) == self.clip_length:
                yield np.array(clips)
                clips = []

        cap.release()

    def predict_video(self, video_path):
        predictions = []
        for clip in self.process_video(video_path):
            clip = np.expand_dims(clip, axis=0) / 255.0
            pred = self.model.predict(clip)[0][0]
            predictions.append(pred)
        return np.mean(predictions)

    def add_watermark(self, frame, text="FAKE"):
        h, w = frame.shape[:2]
        cv2.putText(frame, text, (w//4, h//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 10)
        return frame

    def train(self, data_dir, model_save_path):
        X = []
        y = []

        for label in ['real', 'fake']:
            class_path = os.path.join(data_dir, label)
            label_value = 0 if label == 'real' else 1
            for file in os.listdir(class_path):
                file_path = os.path.join(class_path, file)
                for clip in self.process_video(file_path):
                    if clip.shape == (self.clip_length, 112, 112, 3):
                        X.append(clip)
                        y.append(label_value)

        X = np.array(X) / 255.0
        y = np.array(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=25, batch_size=2)
        self.model.save(model_save_path)

# Run this block separately to trigger training
if __name__ == "__main__":
    detector = VideoDeepfakeDetector(model_path='')  # Empty so it builds new
    dataset_dir = "C:\\deepfake.com\\dataset\\videos"
    save_path = "C:\\deepfake.com\\video_deepfake_detector.h5"
    detector.train(dataset_dir, save_path)