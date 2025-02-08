from dotenv import load_dotenv
from config.model_config import ModelConfig

load_dotenv()  # Load .env file

config = ModelConfig()
# Access tokens through config.HUGGINGFACE_TOKEN
