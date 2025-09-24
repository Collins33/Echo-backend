from openai import OpenAI
import os
from dotenv import load_dotenv
import base64
import streamlit as st
load_dotenv()
api_key = os.getenv("openai_api_key")

client = OpenAI(api_key=api_key)

class TextToSpeech:
    def __init__(self, input_text):

        self.text = input_text

    def speak(self):
        mp3_file_path = "temp_audio_play.mp3"
        with client.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice="nova",
            input=self.text,
            instructions="Speak in a cheerful and positive tone."
        ) as response:
            response.stream_to_file(mp3_file_path)
        return mp3_file_path
