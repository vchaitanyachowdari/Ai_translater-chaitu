import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from gtts import gTTS
import soundfile as sf
from pathlib import Path

class AudioProcessor:
    def __init__(self, config):
        self.config = config
        # Initialize speech recognition model for Hindi
        self.asr_processor = Wav2Vec2Processor.from_pretrained("Harveenchadha/wav2vec2-large-xlsr-hindi")
        self.asr_model = Wav2Vec2ForCTC.from_pretrained("Harveenchadha/wav2vec2-large-xlsr-hindi")
        
    def speech_to_text(self, audio_path: str) -> str:
        """Convert Hindi speech to text"""
        # Load and preprocess audio
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        
        # Process through ASR model
        inputs = self.asr_processor(waveform[0], sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            logits = self.asr_model(inputs.input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.asr_processor.decode(predicted_ids[0])
        
        return transcription
    
    def text_to_speech(self, text: str, output_path: str, lang: str = 'hi') -> str:
        """Convert text to speech in specified language"""
        tts = gTTS(text=text, lang=lang)
        tts.save(output_path)
        return output_path 