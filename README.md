# AI Translator 🌐🎙️

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A production-ready multilingual translation system supporting **English ↔ Spanish ↔ Hindi** with audio capabilities, built with Hugging Face Transformers and FastAPI.

![AI Translator Architecture](https://via.placeholder.com/800x400.png?text=AI+Translator+Architecture)

## Features ✨

- **Multilingual Translation**
  - Text translation between English, Spanish, and Hindi
  - Audio-to-text and text-to-audio conversion
- **Advanced NLP Models**
  - M2M100 for machine translation
  - Wav2Vec2 for speech recognition
- **Production Features**
  - FastAPI REST endpoints
  - Docker support
  - Monitoring and logging
- **Developer Friendly**
  - Type hints throughout
  - Pre-commit hooks
  - CI/CD ready

## Installation 🛠️

```bash
# Clone repository
git clone [https://github.com/vchaitanyachowdari/Ai_translater.git](https://github.com/vchaitanyachowdari/Ai_translater.git)
cd Ai_translater

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.template .env
# Edit .env with your credentials
```

## Usage 🚀

### Basic Translation

```python
from src.interface import TranslationService, TranslationRequest
from config.model_config import ModelConfig

config = ModelConfig()
service = TranslationService(config)

# English to Spanish
request = TranslationRequest(
    source="Hello world",
    source_lang="en",
    target_lang="es"
)
print(service.process(request))  # Output: "Hola mundo"
```

### Audio Translation

```python
# Hindi audio to English text
audio_request = TranslationRequest(
    source="path/to/hindi_audio.wav",
    source_lang="hi",
    target_lang="en",
    input_type="audio"
)
print(service.process(audio_request)) # Output: Translated text
```

## API Documentation 📚

### Endpoints

| Endpoint        | Method | Description                 | Parameters     |
|-----------------|--------|-----------------------------|----------------|
| `/translate`    | POST   | Text/audio translation      | JSON body      |
| `/upload-audio` | POST   | Audio file translation      | Multipart form |
| `/docs`         | GET    | Interactive API docs        | -              |

### Example Request

```bash
curl -X POST "http://localhost:8000/translate" \
-H "Content-Type: application/json" \
-d '{
"source": "Good morning",
"source_lang": "en",
"target_lang": "es"
}'
```

## Configuration ⚙️

Create `.env` file from template:

```bash
cp .env.template .env
```

Key environment variables:

```ini
HF_TOKEN="your_huggingface_token"  # Required
APP_ENV="development"  # development/production
LOG_LEVEL="INFO" # DEBUG/INFO/WARNING/ERROR
MAX_SEQ_LENGTH=128  # Model sequence length
AUDIO_SAMPLE_RATE=16000  # Audio processing rate
```

## Deployment 🛢💀

### Local Deployment

```bash
uvicorn src.interface.api:app --reload
```

### Production Deployment

```bash
gunicorn -w 4 -k uvicorn.workers.UvicornWorker run:app --bind 0.0.0.0:8000
```

## Contributing 🤝

1. Fork the repository
2. Create your feature branch:

   ```bash
   git checkout -b feature/amazing-feature
   ```

3. Commit changes:

   ```bash
   git commit -m 'Add some amazing feature'
   ```

4. Push to branch:

   ```bash
   git push origin feature/amazing-feature
   ```

5. Open a Pull Request

## License 📝

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments 🙏

- [Hugging Face Transformers](https://huggingface.co/transformers)
- [FastAPI](https://fastapi.tiangolo.com/)
- [M2M100 Model](https://huggingface.co/facebook/m2m100_418M)
