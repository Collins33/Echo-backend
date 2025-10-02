import os
import io
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_main_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Echo backend is running"}


def test_transcribe_audio(mocker):
    # mock the ASR models transcribe method so that we do no load whisper
    mock_transcribe = mocker.patch("app.main.asr_model.transcribe", return_value="Hello World")
    # Simulate uploading an audio file
    dummy_audio = io.BytesIO(b"fake-audio-bytes")
    response = client.post(
        "/transcribe",
        files={"file": ("test.wav", dummy_audio, "audio/wav")}
    )

    # Assertions
    assert response.status_code == 200
    assert response.json() == {"transcription": "Hello World"}
    mock_transcribe.assert_called_once()


def test_empty_file_rejected():
    empty_audio = io.BytesIO(b"")  # no content
    response = client.post(
        "/transcribe",
        files={"file": ("empty.wav", empty_audio, "audio/wav")}
    )
    assert response.status_code == 400
    assert "Invalid or empty audio file" in response.json()["detail"]

def test_unsupported_file_format():
    text_file = io.BytesIO(b"Not really audio")
    response = client.post(
        "/transcribe",
        files={"file": ("bad.txt", text_file, "text/plain")}
    )
    assert response.status_code == 415
    assert "Unsupported or unreadable audio format" in response.json()["detail"]

def test_file_too_large():
    # Files that are too large could cause timeouts
    big_file = io.BytesIO(b"0" * (60 * 1024 * 1024))  # 60 MB fake file
    response = client.post(
        "/transcribe",
        files={"file": ("big.wav", big_file, "audio/wav")}
    )
    assert response.status_code == 413
    assert "File too large" in response.json()["detail"]
