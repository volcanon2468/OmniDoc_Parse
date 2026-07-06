import os
from pydantic_settings import BaseSettings
from typing import List, Dict

class Settings(BaseSettings):
    app_name: str = 'OmniDoc Parse Microservice'
    app_version: str = '1.0.0'
    api_v1_prefix: str = '/api/v1'
    cors_origins: List[str] = ['http://localhost:3000']
    yolo_model_path: str = '../yolov8n.pt'
    config_path: str = '../config.json'
    max_upload_size: int = 10 * 1024 * 1024
    processing_timeout: int = 120
    ai_max_dimension: int = 2048
    ai_noise_threshold: int = 100
    ai_tesseract_path: str = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
    ai_min_confidence: int = 60
    ai_top_keywords: int = 10
    ai_min_keyword_length: int = 3
    ai_yolo_confidence: float = 0.25
    ai_yolo_iou: float = 0.45
    ai_yolo_min_area: int = 100
    ai_yolo_max_detections: int = 100
    ai_easyocr_min_confidence: float = 10.0
    ai_doc_classes: Dict[str, List[str]] = {
        'invoice': ['invoice', 'amount', 'due', 'total', 'payment', 'bill'],
        'receipt': ['receipt', 'purchase', 'paid', 'total', 'change', 'cash'],
        'letter': ['dear', 'sincerely', 'regards', 'yours', 'thank you'],
        'form': ['name:', 'address:', 'signature:', 'date:', 'please fill'],
        'business_card': ['email', 'phone', 'company', 'mobile', 'fax', 'website']
    }
    log_level: str = 'INFO'
    environment: str = 'development'

    class Config:
        env_file = '.env'
        case_sensitive = False
settings = Settings()