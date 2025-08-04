# app/stt_elevenlabs.py

import requests
import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variables
ELEVENLABS_API_KEY = os.getenv("XI_API_KEY")

def transcribe_audio(audio_path: str, language: str = "en") -> str:
    """
    Send the audio file at audio_path to ElevenLabs STT and return the transcript.
    Optionally forces the language via `language_code`.
    """
    url = "https://api.elevenlabs.io/v1/speech-to-text"
    headers = {"xi-api-key": ELEVENLABS_API_KEY}

    data = {
        "model_id": "scribe_v1",        # required
        "language_code": language       # ISO-639-1 or ISO-639-3, e.g. "en" or "eng" :contentReference[oaicite:0]{index=0}
    }
    
    try:
        with open(audio_path, "rb") as audio_file:
            files = {"file": audio_file}
            resp = requests.post(url, headers=headers, data=data, files=files)
            
        if resp.status_code == 200:
            return resp.json().get("text", "")
        else:
            print("[ERROR] ElevenLabs STT:", resp.status_code, resp.text)
            return "Transcription failed."
    except Exception as e:
        print(f"[ERROR] Exception during transcription: {str(e)}")
        return "Transcription failed due to error."

# Allow running directly from command line
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python stt_elevenlabs.py <audio_file_path> [language]")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    language = sys.argv[2] if len(sys.argv) > 2 else "en"
    
    result = transcribe_audio(audio_path, language)
    print(result)  # Print to stdout for the Node.js process to capture
