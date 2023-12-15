import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# Path ke model yang sudah dilatih
model_path = 'model_save/model.h5'

# Inisialisasi MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

mp_drawing = mp.solutions.drawing_utils

# Load pre-trained TensorFlow model
model = tf.keras.models.load_model(model_path)

# Mapping indeks kelas ke label
poses_mapping = {0: 'Downdog', 1: 'Goddess', 2: 'Plank', 3: 'Tree', 4: 'Warrior2'}

# Inisialisasi Webcam
cap = cv2.VideoCapture(0)  # Angka 0 menunjukkan penggunaan kamera default

# Ambang batas keyakinan (confidence)
confidence_threshold = 0.95  # Sesuaikan sesuai kebutuhan

# Fungsi untuk mengupdate tampilan GUI dengan frame terbaru
def update_frame():
    ret, frame = cap.read()

    if not ret:
        print("Gagal mendapatkan frame dari webcam.")
        return
    
    # Anti mirror kamera
    frame = cv2.flip(frame, 1)

    # Proses frame menggunakan MediaPipe Pose
    with mp_pose.Pose() as pose_tracker:
        result = pose_tracker.process(image=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        pose_landmarks = result.pose_landmarks

    # Check if pose landmarks are detected with sufficient confidence
    if pose_landmarks and result.pose_world_landmarks:
        landmarks = [[landmark.x, landmark.y, landmark.z] for landmark in pose_landmarks.landmark]
        frame_height, frame_width = frame.shape[:2]

        # Normalisasi koordinat landmark
        landmarks *= np.array([frame_height, frame_height, frame_width])

        # Bundel landmark menjadi array 1D dan ubah tipe data menjadi float32
        pose_landmarks = np.around(landmarks, 5).flatten().astype(np.float32)

        # Lakukan prediksi dengan model yang sudah dilatih
        prediction = model.predict(np.array([pose_landmarks]))

        # Pengecekan confidence threshold
        if np.max(prediction[0]) > confidence_threshold:
            print("Raw Prediction:", prediction)
            print("Argmax Prediction:", np.argmax(prediction[0]))

            # Mendapatkan indeks kelas hasil prediksi
            predicted_class_index = np.argmax(prediction)

            # Mendapatkan label hasil prediksi dari dictionary poses_mapping
            predicted_pose_label = poses_mapping.get(predicted_class_index, '')

            # Tampilkan hasil prediksi pada GUI
            predicted_pose_var.set(f"Predicted Pose: {predicted_pose_label}")
        else:
            # Jika confidence kurang dari ambang, set label sebagai
            predicted_pose_var.set("Predicted Pose: Tidak Diketahui")
    else:
        # Pose not detected, set the label sebagai
        predicted_pose_var.set("Tidak Ada Landmark")

    # Render landmark pose pada frame
    mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Konversi frame OpenCV menjadi format yang dapat ditampilkan di Tkinter
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = ImageTk.PhotoImage(image=img)

    # Update label gambar pada GUI
    video_label.img = img
    video_label.config(image=img)

    # Schedule fungsi update_frame setiap 10 milidetik
    root.after(10, update_frame)

# Fungsi untuk mengatur latar belakang GUI dengan foto
def set_background(image_path):
    img = Image.open(image_path)
    img = img.resize((root.winfo_screenwidth(), root.winfo_screenheight()))

    img = ImageTk.PhotoImage(img)

    background_label.img = img
    background_label.config(image=img)
    background_label.place(relx=0, rely=0, relwidth=1, relheight=1)

# Inisialisasi GUI
root = tk.Tk()
root.title("Deteksi Pose Yoga")
root.iconbitmap('logo.ico')

# Label untuk latar belakang
background_label = ttk.Label(root)
background_label.pack()

# Label untuk menampilkan teks "Deteksi Pose Yoga" diatas frame kamera
title_label = ttk.Label(root, text="Deteksi Pose Yoga", font=("Helvetica", 18, "bold"))
title_label.pack(pady=30)

# Label untuk menampilkan video dari webcam
video_label = ttk.Label(root)
video_label.pack()

# Label untuk menampilkan hasil prediksi
predicted_pose_var = tk.StringVar()
predicted_pose_label = ttk.Label(root, textvariable=predicted_pose_var, font=("Helvetica", 16))
predicted_pose_label.pack(pady=30)

# Button untuk keluar dari aplikasi
exit_button = ttk.Button(root, text="Tutup Kamera", command=root.destroy)
exit_button.pack(pady=30)

# Panggil fungsi set_background dengan path foto yang diinginkan
background_image_path = 'bg_yoga.jpg'
set_background(background_image_path)

# Mulai fungsi update_frame untuk menampilkan video secara real-time
update_frame()

# Jalankan GUI
root.mainloop()

# Tutup webcam saat aplikasi ditutup
cap.release()
cv2.destroyAllWindows()