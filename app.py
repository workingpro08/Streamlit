import streamlit as st
import whisper
import librosa
import soundfile as sf
import tempfile
import os
from transcribe import transcribe_audio_file

st.title("🎤 Speech-to-Text Transcription (Librosa)")

# Upload audio file
uploaded_file = st.file_uploader("Upload Audio (MP3/WAV)", type=["mp3", "wav"])

# Real-time transcription option
if st.button("🎤 Start Real-Time Transcription"):
    st.warning("Real-time transcription requires running the script locally with microphone access")

# Process uploaded file
if uploaded_file:
    st.audio(uploaded_file)
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # Transcribe
    if st.button("Transcribe"):
        with st.spinner("Transcribing..."):
            transcription = transcribe_audio_file(tmp_path)
            st.success("Transcription Complete!")
            st.text_area("Transcription", transcription, height=200)

    # Clean up
    os.unlink(tmp_path)