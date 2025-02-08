from src.model.multilingual_translator import MultilingualTranslator
from src.data_processing.audio_processor import AudioProcessor
from src.interface.translation_service import TranslationService, TranslationRequest
from config import ProjectConfig

def main():
    # Initialize components
    config = ProjectConfig()
    translator = MultilingualTranslator(config)
    audio_processor = AudioProcessor(config)
    service = TranslationService(config, translator, audio_processor)
    
    # Example usage
    
    # 1. Text translation (Hindi to English)
    text_request = TranslationRequest(
        source="नमस्ते, आप कैसे हैं?",
        source_lang="hi",
        target_lang="en",
        input_type="text",
        output_type="text"
    )
    text_result = service.process(text_request)
    print(f"Text translation: {text_result}")
    
    # 2. Audio to text translation (Hindi audio to English text)
    audio_request = TranslationRequest(
        source="path/to/hindi_audio.wav",
        source_lang="hi",
        target_lang="en",
        input_type="audio",
        output_type="text"
    )
    audio_result = service.process(audio_request)
    print(f"Audio translation: {audio_result}")
    
    # 3. Text to audio translation (English text to Hindi audio)
    speech_request = TranslationRequest(
        source="Hello, how are you?",
        source_lang="en",
        target_lang="hi",
        input_type="text",
        output_type="audio"
    )
    speech_result = service.process(speech_request)
    print(f"Audio output saved to: {speech_result}")

    # Create request
    request = TranslationRequest(
        source="your_input",
        source_lang="hi",
        target_lang="en",
        input_type="audio",  # or "text"
        output_type="text"   # or "audio"
    )

    # Get translation
    result = service.process(request)

if __name__ == "__main__":
    main()
