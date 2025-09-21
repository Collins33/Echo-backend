from fastapi import FastAPI, UploadFile, File, Form
from app.services.whisper_asr import WhisperASR
import os

app = FastAPI(title="Echo Backend")
asr_model = WhisperASR()


@app.get("/")
def root():
    return {"message": "Echo backend is running"}


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)) :
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    try:
        transcription = asr_model.transcribe(temp_path)
        return {"transcription": transcription}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
