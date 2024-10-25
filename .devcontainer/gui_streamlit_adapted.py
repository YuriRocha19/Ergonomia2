
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time
from pathlib import Path

# Configuração do MediaPipe
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5)

# Função para processar imagem com OpenCV e MediaPipe
def process_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(frame_rgb)

    # Visualizar as landmarks
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    return frame

st.title("Ergonomia com OpenCV e MediaPipe")

# Carregar imagem
uploaded_file = st.file_uploader("Escolha uma imagem", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Ler imagem
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)

    # Processar imagem
    processed_frame = process_frame(frame)

    # Exibir imagem processada
    st.image(processed_frame, channels="BGR", caption="Imagem processada")

st.write("Para mais informações, visite o [LinkedIn do autor](https://www.linkedin.com/in/yuri-rudimar/)")
