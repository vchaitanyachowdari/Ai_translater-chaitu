import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from src.interface import TranslationService, TranslationRequest
from src.models import MultilingualTranslator
from src.data import AudioProcessor
from config.model_config import ModelConfig

def main():
    # Initialize components
    config = ModelConfig()
    translator = MultilingualTranslator(config)
    audio_processor = AudioProcessor(config)
    service = TranslationService(config, translator, audio_processor)
    
    # Example text translation (English to Spanish)
    request_es = TranslationRequest(
        source="Hello, how are you?",
        source_lang="en",
        target_lang="es",
        input_type="text",
        output_type="text"
    )
    result_es = service.process(request_es)
    print(f"Translation (English to Spanish): {result_es}")
    
    # Example text translation (English to Hindi)
    request_hi = TranslationRequest(
        source="Hello, how are you?",
        source_lang="en",
        target_lang="hi",
        input_type="text",
        output_type="text"
    )
    result_hi = service.process(request_hi)
    print(f"Translation (English to Hindi): {result_hi}")
    
    # Example audio to text translation (Hindi audio to English text)
    audio_request = TranslationRequest(
        source="path/to/hindi_audio.wav",
        source_lang="hi",
        target_lang="en",
        input_type="audio",
        output_type="text"
    )
    audio_result = service.process(audio_request)
    print(f"Audio translation (Hindi to English): {audio_result}")

if __name__ == "__main__":
    main() 