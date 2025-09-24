from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel
from app.services.text_to_speech import TextToSpeech


from app.services.whisper_asr import WhisperASR

import os
import io

app = FastAPI(title="Echo Backend")
asr_model = WhisperASR()

class TTSRequest(BaseModel):
    text: str

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

@app.post("/tts")
def generate_tts(request: TTSRequest):
    tts = TextToSpeech(request.text)
    mp3_path = tts.speak()
    return FileResponse(
        mp3_path,
        media_type="audio/mpeg",
        filename="speech.mp3"
    )


