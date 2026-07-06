import sys
import os
from pathlib import Path
import base64
import io
import numpy as np
from PIL import Image
import cv2
from typing import Dict, Any, Optional
from app.modules.preprocessor import ImagePreprocessor
from app.modules.ocr_engine import OCREngine
from app.modules.nlp_processor import NLPProcessor
from app.modules.object_detector import ObjectDetector
from app.core.config import settings
from app.core.logging import logger
import threading

class AIService:
    _instance = None
    _initialized = False
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(AIService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not AIService._initialized:
            with AIService._lock:
                if not AIService._initialized:
                    logger.info('Initializing AI modules...')
                    self.config = {
                        'preprocessing': {
                            'max_dimension': settings.ai_max_dimension,
                            'noise_threshold': settings.ai_noise_threshold
                        },
                        'ocr': {
                            'tesseract_path': settings.ai_tesseract_path,
                            'min_confidence': settings.ai_min_confidence,
                            'easyocr_min_confidence': settings.ai_easyocr_min_confidence
                        },
                        'nlp': {
                            'top_keywords': settings.ai_top_keywords,
                            'min_keyword_length': settings.ai_min_keyword_length,
                            'doc_classes': settings.ai_doc_classes
                        },
                        'object_detection': {
                            'model_path': settings.yolo_model_path,
                            'confidence_threshold': settings.ai_yolo_confidence,
                            'iou_threshold': settings.ai_yolo_iou,
                            'min_area': settings.ai_yolo_min_area,
                            'max_detections': settings.ai_yolo_max_detections
                        }
                    }
                    self.preprocessor = ImagePreprocessor(self.config['preprocessing'])
                    self.ocr_engine = OCREngine(self.config['ocr'])
                    self.nlp_processor = NLPProcessor(self.config['nlp'])
                    self.object_detector = ObjectDetector(self.config['object_detection'])
                    AIService._initialized = True
                    logger.info('AI modules initialized successfully')

    @staticmethod
    def encode_image_to_base64(image: np.ndarray) -> str:
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image)
        buffered = io.BytesIO()
        pil_image.save(buffered, format='PNG')
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f'data:image/png;base64,{img_str}'

    def preprocess_image(self, image_bytes: bytes, max_dimension: int=2048, noise_threshold: int=100, include_images: bool=False) -> Dict[str, Any]:
        try:
            original_config = self.preprocessor.config.copy()
            original_config['max_dimension'] = max_dimension
            original_config['noise_threshold'] = noise_threshold
            result = self.preprocessor.process(image_bytes, config_override=original_config)
            data = {'skew_angle': result['skew_angle'], 'quality_score': result['quality_score'], 'binarization_method': result['binarization_method']}
            if include_images:
                data['original_base64'] = self.encode_image_to_base64(result['original'])
                data['enhanced_base64'] = self.encode_image_to_base64(result['enhanced'])
                data['binary_base64'] = self.encode_image_to_base64(result['binary'])
            return data
        except Exception as e:
            logger.error(f'Preprocessing error: {str(e)}')
            raise e

    def extract_text(self, image_bytes: bytes, mode: str='balanced', engine: Optional[str]=None) -> Dict[str, Any]:
        try:
            preprocessed = self.preprocessor.process(image_bytes)
            result = self.ocr_engine.process(preprocessed, mode=mode)
            return {'data': result, 'text': result.get('text', ''), 'confidence': result.get('confidence', 0.0), 'word_count': result.get('word_count', 0), 'engine_used': result.get('engine', 'unknown')}
        except Exception as e:
            logger.error(f'OCR error: {str(e)}')
            raise e

    def analyze_text(self, text: str, extract_entities: bool=True, extract_keywords: bool=True, classify_document: bool=True) -> Dict[str, Any]:
        try:
            result = self.nlp_processor.process(text, extract_entities=extract_entities, extract_keywords=extract_keywords, classify_document=classify_document)
            return {'data': result, 'entities': result.get('entities', {}), 'keywords': result.get('keywords', []), 'classification': result.get('classification', {}), 'statistics': result.get('statistics', {}), 'patterns': result.get('patterns', {})}
        except Exception as e:
            logger.error(f'NLP error: {str(e)}')
            raise e

    def detect_objects(self, image_bytes: bytes, confidence_threshold: float=0.25, iou_threshold: float=0.45, annotate: bool=False, include_images: bool=False) -> Dict[str, Any]:
        try:
            result = self.object_detector.detect(image_bytes, annotate=annotate, confidence_threshold=confidence_threshold, iou_threshold=iou_threshold)
            response = {'objects': result.get('detections', []), 'object_count': len(result.get('detections', [])), 'processing_time': result.get('processing_time', 0.0)}
            if annotate and 'annotated_image' in result:
                if include_images:
                    response['annotated_image'] = self.encode_image_to_base64(result['annotated_image'])
                del result['annotated_image']
            response['data'] = result
            return response
        except Exception as e:
            logger.error(f'Object detection error: {str(e)}')
            raise e

    def is_healthy(self) -> bool:
        return all([self.preprocessor is not None, self.ocr_engine is not None, self.nlp_processor is not None, self.object_detector is not None])

def get_ai_service() -> 'AIService':
    return AIService()