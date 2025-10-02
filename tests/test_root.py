import os
import io
import base64
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


# =====================AWS POLLY TESTS ============================

def test_generate_tts(mocker):
    # --- Step 1: Mock AWS Polly client ---
    fake_audio = b"fake-binary-audio" # Simulate AWS Polly audio stream with a short byte string
    mock_client = mocker.Mock() # create a mock client that can behave how we want it(it can pretend to be aws polly)
    # AWS Polly's synthesize_speech returns a dict with AudioStream
    # io.BytesIO(fake_audio) makes the fake_audio look like a real file stream
    mock_client.synthesize_speech.return_value = {
        "AudioStream": io.BytesIO(fake_audio)
    }
    # temp replace boto3.client function with our fake one
    mocker.patch("app.main.boto3.client", return_value=mock_client)

    # Patch get_settings so AWS credentials don't matter
    fake_settings = mocker.Mock()
    fake_settings.AWS_AK = "fake-ak"
    fake_settings.AWS_SAK = "fake-sak"
    mocker.patch("app.main.get_settings", return_value=fake_settings)

    # --- Step 2: Call the endpoint ---
    payload = {"content": "Hello world", "output_format": "mp3"}
    response = client.post("/text-to-speech", json=payload)

    # --- Step 3: Assertions ---
    assert response.status_code == 200
    data = response.json()

    # Check message and input reflection
    assert data["message"] == "Audio conversion complete"
    assert data["data"]["text"] == "Hello world"
    assert data["data"]["output_format"] == "mp3"

    # Check base64 encoding correctness
    expected_base64 = base64.b64encode(fake_audio).decode("utf-8")
    assert data["data"]["audio"] == expected_base64


def test_empty_text_rejected():
    payload = {"content": "", "output_format": "mp3"}
    response = client.post("/text-to-speech", json=payload)
    assert response.status_code == 400
    assert response.json()["detail"] == "Text input cannot be empty"


def test_unsupported_format_rejected():
    payload = {"content": "Hello", "output_format": "wavz"}
    response = client.post("/text-to-speech", json=payload)
    assert response.status_code == 400
    assert response.json()["detail"] == "Unsupported output format"

def test_polly_failure_handled(mocker):
    mock_client = mocker.Mock()
    mock_client.synthesize_speech.side_effect = Exception("AWS Error")
    mocker.patch("app.main.boto3.client", return_value=mock_client)

    payload = {"content": "Hello", "output_format": "mp3"}
    response = client.post("/text-to-speech", json=payload)
    assert response.status_code == 500
    assert response.json()["detail"] == "Text-to-speech service unavailable"
