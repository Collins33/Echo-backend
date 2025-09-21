import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration

class WhisperASR:
    def __init__(self, model_size="small"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = WhisperProcessor.from_pretrained(f"openai/whisper-{model_size}")
        self.model = WhisperForConditionalGeneration.from_pretrained(f"openai/whisper-{model_size}").to(self.device)

    def transcribe(self, audio_path):
        waveform, sample_rate = torchaudio.load(audio_path) # get the original sample rate
        if sample_rate != 16000: # expects 16khz audio
            # resample it if it is not 16khz
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)

        # convert raw waveform into the model's expected input features
        inputs = self.processor(
            waveform.squeeze(),
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features.to(self.device)

        predicted_ids = self.model.generate(inputs) # generate predicted token IDs, generate() runs decoding

        # convert token IDs back to text
        # skip special tokens, removes model specific tokens like <pad>
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        return transcription
