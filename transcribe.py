import whisper
import noisereduce as nr
import numpy as np
import librosa
import soundfile as sf
import os

def transcribe_audio_file(file_path, noise_reduction=True):
    # Load Whisper model
    model = whisper.load_model("base")
    
    # Load audio with librosa (automatically resamples to 16kHz for Whisper)
    try:
        audio, sr = librosa.load(file_path, sr=16000)  # Force 16kHz sample rate
    except Exception as e:
        return f"Error loading audio: {str(e)}"

    # Noise reduction
    if noise_reduction:
        audio = nr.reduce_noise(y=audio, sr=sr)

    # Save processed audio temporarily (Whisper works better with files)
    temp_file = "temp_clean.wav"
    sf.write(temp_file, audio, sr)

    # Transcribe with Whisper
    try:
        result = model.transcribe(temp_file)
        os.remove(temp_file)  # Clean up temporary file
    except Exception as e:
        return f"Transcription error: {str(e)}"

    # Format output with timestamps
    transcription = ""
    for segment in result["segments"]:
        start = segment["start"]
        end = segment["end"]
        text = segment["text"]
        transcription += f"[{start:.2f}s - {end:.2f}s] {text}\n"

    return transcription

def real_time_transcription():
    import sounddevice as sd
    import queue
    import threading
    
    model = whisper.load_model("base")
    audio_queue = queue.Queue()
    SAMPLE_RATE = 16000
    
    def callback(indata, frames, time, status):
        audio_queue.put(indata.copy())
    
    print("🎤 Recording... Press Ctrl+C to stop.")
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        callback=callback,
        dtype='float32'
    )
    
    try:
        with stream:
            while True:
                audio_data = audio_queue.get()
                audio_np = np.array(audio_data).flatten()
                
                # Save to temp file for Whisper
                temp_file = "temp_realtime.wav"
                sf.write(temp_file, audio_np, SAMPLE_RATE)
                
                result = model.transcribe(temp_file)
                os.remove(temp_file)
                print(f"[Real-Time] {result['text']}")
    except KeyboardInterrupt:
        print("\n⏹️ Stopped recording.")

if __name__ == "__main__":
    # File-based test
    file_path = "Testvoice.mp3"
    if os.path.exists(file_path):
        print("File transcription:")
        print(transcribe_audio_file(file_path))
    else:
        print(f"Test file {file_path} not found")
    
    # Uncomment for real-time test
    # real_time_transcription()