import os
import tempfile
import whisper
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = whisper.load_model("base", device=device)

def transcribe_audio(file):
  
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
        temp_audio.write(file.read())
        temp_audio_path = temp_audio.name

    result = whisper_model.transcribe(temp_audio_path)

 
    os.remove(temp_audio_path)

    return result['text']
