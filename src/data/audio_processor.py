import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from gtts import gTTS
import soundfile as sf
from pathlib import Path
import os
from config.model_config import ModelConfig

class AudioProcessor:
    def __init__(self, config: ModelConfig):
        self.config = config
        # Example for future API integrations
        self.google_api_key = os.getenv("GOOGLE_API_KEY", "")
        # Initialize speech recognition model for Hindi
        self.asr_processor = Wav2Vec2Processor.from_pretrained(config.ASR_MODEL)
        self.asr_model = Wav2Vec2ForCTC.from_pretrained(config.ASR_MODEL)
        self.device = torch.device(config.DEVICE)
        self.asr_model.to(self.device)
        
    def speech_to_text(self, audio_path: str) -> str:
        """Convert Hindi speech to text"""
        try:
            # Load and preprocess audio
            waveform, sample_rate = torchaudio.load(audio_path)
            if sample_rate != self.config.SAMPLE_RATE:
                waveform = torchaudio.functional.resample(waveform, sample_rate, self.config.SAMPLE_RATE)
            
            # Process through ASR model
            inputs = self.asr_processor(
                waveform[0], 
                sampling_rate=self.config.SAMPLE_RATE, 
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                logits = self.asr_model(inputs.input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.asr_processor.decode(predicted_ids[0])
            
            return transcription
            
        except Exception as e:
            print(f"Error processing audio: {e}")
            return ""
    
    def text_to_speech(self, text: str, output_path: str, lang: str = 'hi') -> str:
        """Convert text to speech in specified language"""
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            tts = gTTS(text=text, lang=lang)
            tts.save(str(output_path))
            
            return str(output_path)
            
        except Exception as e:
            print(f"Error generating speech: {e}")
            return "" 