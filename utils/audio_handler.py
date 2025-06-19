# audio_handler.py
import os
import tempfile
import whisper
import torch

# Load Whisper model
device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = whisper.load_model("base", device=device)

def transcribe_audio(file):
    """
    Transcribe audio file using Whisper.
    
    Args:
        file: Uploaded audio file (e.g. from Streamlit or file path)
    
    Returns:
        str: Transcribed text from the audio
    """
    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
        temp_audio.write(file.read())
        temp_audio_path = temp_audio.name

    # Run Whisper transcription
    result = whisper_model.transcribe(temp_audio_path)

    # Clean up temp file
    os.remove(temp_audio_path)

    return result['text']
