# -*- coding: utf-8 -*-
"""
Advanced Audio Deepfake Detection App - Streamlit
"""

import streamlit as st
import numpy as np
import librosa
import pickle
import os
import matplotlib.pyplot as plt

# ✅ Must be at the top
st.set_page_config(page_title="🎙️ Audio Deepfake Detector", page_icon="🎧", layout="wide")

# ---- Sidebar ----
with st.sidebar:
    st.header("📌 How it Works")
    st.markdown("""
    1. Upload an audio file (`.wav`, `.mp3`, or `.m4a`).
    2. We extract features using **MFCC**.
    3. Features are scaled using a saved `scaler.pkl`.
    4. The model (`trial.pkl`) predicts **Real** or **Fake**.
    """)
    st.markdown("---")
    #st.caption("🔐 Ensure `scaler.pkl` and `trial.pkl` are in the same folder as this script.")

# ---- Feature Extraction ----
def extract_features(audio_path, n_mfcc=13):
    y, sr = librosa.load(audio_path, duration=5, sr=16000)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfccs.T, axis=0), y, sr

# ---- Audio Visualizer ----
def plot_waveform(y, sr):
    fig, ax = plt.subplots(figsize=(6, 2))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title("Audio Waveform", fontsize=10)
    ax.set_yticks([])
    ax.set_xticks([])
    st.pyplot(fig)

# ---- Main App ----
def main():
    st.title("🔍 Advanced Audio Deepfake Detection")
    st.markdown("Upload an audio file to classify it as **Real** 🟢 or **Fake** 🔴 using a pre-trained ML model.")
    st.markdown("---")

    uploaded_audio = st.file_uploader("🎼 Upload Audio", type=["wav", "mp3", "m4a"])
    if uploaded_audio:
        # Left Column: Audio
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("▶️ Preview")
            st.audio(uploaded_audio)

        # Save to temp file
        temp_audio_path = "temp_audio.wav"
        with open(temp_audio_path, "wb") as f:
            f.write(uploaded_audio.getbuffer())

        with st.spinner("🧠 Extracting features and predicting..."):
            try:
                features, y, sr = extract_features(temp_audio_path)
                features = features.reshape(1, -1)

                if np.isnan(features).any():
                    st.error("🚫 Feature extraction failed due to NaN values.")
                    return

                # Load scaler
                scaler_path = "scaler.pkl"
                if not os.path.exists(scaler_path):
                    st.error("⚠️ Missing `scaler.pkl` file.")
                    return
                scaler = pickle.load(open(scaler_path, "rb"))
                features = scaler.transform(features)

                # Load model
                model_path = "audio_deepfake_model.pkl"
                if not os.path.exists(model_path):
                    st.error("⚠️ Missing `audio_deepfake_model.pkl` model file.")
                    return
                model = pickle.load(open(model_path, "rb"))

                # Predict
                prediction = model.predict(features)
                result = "🟢 Real" if prediction[0] == 0 else "🔴 Fake"
                proba = model.predict_proba(features)[0] if hasattr(model, "predict_proba") else None

                # Show waveform
                with col2:
                    st.subheader("📊 Waveform")
                    plot_waveform(y, sr)

                # Styled result
                with st.container():
                    st.markdown("---")
                    st.markdown("## 🧪 Prediction Result")
                    st.success(f"The uploaded audio is classified as: **{result}**")

                    if proba is not None:
                        st.metric("Confidence (Fake)", f"{proba[1]*100:.2f}%")
                        st.metric("Confidence (Real)", f"{proba[0]*100:.2f}%")

            except Exception as e:
                st.error(f"💥 Error during prediction: `{e}`")

    st.markdown("---")
    st.caption("Made with ❤️by Bhavesh")

if __name__ == "__main__":
    main()
