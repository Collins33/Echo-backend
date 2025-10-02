from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from functools import lru_cache
from fastapi.responses import FileResponse
from pydantic import BaseModel
from app.services.whisper_asr import WhisperASR
from app import config
import boto3
import base64

import os
import io

app = FastAPI(title="Echo Backend")
asr_model = WhisperASR()

MAX_FILE_SIZE_MB = 50

def get_settings():
    return config.Settings()
class Text(BaseModel):
    content: str
    output_format: str

@app.get("/")
def root():
    return {"message": "Echo backend is running"}

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)) :
    temp_path = f"temp_{file.filename}"

    # read the file
    file_bytes = await file.read()

    if len(file_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large")

    # throw error if the file is empty
    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Invalid or empty audio file")

    with open(temp_path, "wb") as f:
        f.write(await file.read())

    try:
        try:
            transcription = asr_model.transcribe(temp_path)
        except RuntimeError as e:
            raise HTTPException(status_code=415, detail="Unsupported or unreadable audio format")
        return {"transcription": transcription}

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/text-to-speech")
def generate_tts(text: Text):
    # --- Edge Case 1: Empty text ---
    if not text.content.strip():
        raise HTTPException(status_code=400, detail="Text input cannot be empty")

    # --- Edge Case 2: Unsupported output format ---
    supported_formats = {"mp3", "ogg_vorbis", "pcm"}
    if text.output_format not in supported_formats:
        raise HTTPException(status_code=400, detail="Unsupported output format")

    try:
        client = boto3.client(
            'polly',
            aws_access_key_id=get_settings().AWS_AK,
            aws_secret_access_key=get_settings().AWS_SAK,
            region_name='us-east-1'
        )
        result = client.synthesize_speech(
            Text=text.content,
            OutputFormat=text.output_format,
            VoiceId='Brian'
        )
        audio = result['AudioStream'].read()
        encoded_audio = base64.b64encode(audio).decode('utf-8')

        return {"message": "Audio conversion complete", "data": {
            "text": text.content,
            "output_format": text.output_format,
            "audio": encoded_audio
        }}

    # --- Edge Case 3: Polly service failure ---
    except Exception as e:
        raise HTTPException(status_code=500, detail="Text-to-speech service unavailable")


