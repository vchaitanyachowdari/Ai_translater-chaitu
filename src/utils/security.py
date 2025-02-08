import logging
from typing import Optional

logger = logging.getLogger(__name__)

def validate_environment(required_keys: list[str]) -> bool:
    missing = [key for key in required_keys if not os.getenv(key)]
    if missing:
        logger.error(f"Missing environment variables: {', '.join(missing)}")
        return False
    return True 