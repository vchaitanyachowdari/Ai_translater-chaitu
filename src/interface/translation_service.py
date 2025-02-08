from pathlib import Path
from typing import Optional, Union
from dataclasses import dataclass
from config.model_config import ModelConfig
from src.models.multilingual_translator import MultilingualTranslator
from src.data.audio_processor import AudioProcessor

@dataclass
class TranslationRequest:
    source: str  # Path to audio file or text content
    source_lang: str
    target_lang: str
    input_type: str = "text"  # "text" or "audio"
    output_type: str = "text"  # "text" or "audio"

class TranslationService:
    def __init__(self, config, translator: MultilingualTranslator, audio_processor: AudioProcessor):
        self.config = config
        self.translator = translator
        self.audio_processor = audio_processor
        self.output_dir = Path(config.OUTPUT_DIR)
        self.output_dir.mkdir(exist_ok=True)
    
    def process(self, request: TranslationRequest) -> Union[str, Path]:
        """Process translation request with any combination of text/audio input/output"""
        
        # Handle input
        if request.input_type == "audio":
            # Convert audio to text in source language
            source_text = self.audio_processor.speech_to_text(request.source)
        else:
            source_text = request.source
        
        # Perform translation based on source and target languages
        if request.source_lang == "en" and request.target_lang == "es":
            translated_text = self.translator.translate_to_spanish(source_text)
        elif request.source_lang == "en" and request.target_lang == "hi":
            translated_text = self.translator.translate_to_hindi(source_text)
        elif request.source_lang == "es" and request.target_lang == "en":
            translated_text = self.translator.translate_from_spanish(source_text)
        elif request.source_lang == "hi" and request.target_lang == "en":
            translated_text = self.translator.translate_from_hindi(source_text)
        else:
            translated_text = self.translator.translate(source_text, request.source_lang, request.target_lang)
        
        # Handle output
        if request.output_type == "audio":
            output_path = self.output_dir / f"output_{request.target_lang}.mp3"
            self.audio_processor.text_to_speech(
                translated_text,
                str(output_path),
                request.target_lang
            )
            return output_path
        else:
            return translated_text 