class TranslationError(Exception):
    """Base exception for translation errors"""
    pass

class ModelError(TranslationError):
    """Raised when there's an error with the model"""
    pass

class DataError(TranslationError):
    """Raised when there's an error with the data"""
    pass

class AudioError(TranslationError):
    """Raised when there's an error processing audio"""
    pass 