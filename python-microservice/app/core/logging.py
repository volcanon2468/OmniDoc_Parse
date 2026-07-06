import logging
import sys
from app.core.config import settings

def setup_logging():
    logger = logging.getLogger('ai_microservice')
    logger.setLevel(getattr(logging, settings.log_level.upper()))
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
logger = setup_logging()