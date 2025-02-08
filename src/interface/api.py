from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import Optional
from .translation_service import TranslationService, TranslationRequest
from ..utils.logger import setup_logger

app = FastAPI()
logger = setup_logger("translation_api", log_file="logs/api.log")

class TranslationRequestModel(BaseModel):
    source: str
    source_lang: str
    target_lang: str
    input_type: str = "text"  # "text" or "audio"
    output_type: str = "text"  # "text" or "audio"

class TranslationResponseModel(BaseModel):
    translation: Optional[str] = None
    audio_path: Optional[str] = None

class FeedbackModel(BaseModel):
    user_id: str
    feedback: str

# Initialize the translation service
def create_translation_service():
    from config.model_config import ModelConfig
    from src.models.multilingual_translator import MultilingualTranslator
    from src.data.audio_processor import AudioProcessor

    config = ModelConfig()
    translator = MultilingualTranslator(config)
    audio_processor = AudioProcessor(config)
    return TranslationService(config, translator, audio_processor)

translation_service = create_translation_service()

@app.post("/translate", response_model=TranslationResponseModel)
async def translate(request: TranslationRequestModel):
    """Translate text or audio based on the request."""
    try:
        translation_request = TranslationRequest(
            source=request.source,
            source_lang=request.source_lang,
            target_lang=request.target_lang,
            input_type=request.input_type,
            output_type=request.output_type
        )
        
        result = translation_service.process(translation_request)
        
        if request.output_type == "audio":
            return TranslationResponseModel(audio_path=str(result))
        else:
            return TranslationResponseModel(translation=result)
    
    except Exception as e:
        logger.error(f"Error during translation: {str(e)}")
        raise HTTPException(status_code=500, detail="Translation failed")

@app.post("/upload-audio", response_model=TranslationResponseModel)
async def upload_audio(file: UploadFile = File(...), source_lang: str = "hi", target_lang: str = "en"):
    """Upload audio file for translation."""
    try:
        # Save the uploaded file temporarily
        temp_file_path = f"temp/{file.filename}"
        with open(temp_file_path, "wb") as f:
            f.write(await file.read())
        
        translation_request = TranslationRequest(
            source=temp_file_path,
            source_lang=source_lang,
            target_lang=target_lang,
            input_type="audio",
            output_type="text"
        )
        
        result = translation_service.process(translation_request)
        return TranslationResponseModel(translation=result)
    
    except Exception as e:
        logger.error(f"Error during audio upload: {str(e)}")
        raise HTTPException(status_code=500, detail="Audio processing failed")

@app.post("/feedback")
async def submit_feedback(feedback: FeedbackModel):
    """Submit user feedback."""
    logger.info(f"Feedback from {feedback.user_id}: {feedback.feedback}")
    # Store feedback in a database or file for analysis
    return {"message": "Feedback received"} 