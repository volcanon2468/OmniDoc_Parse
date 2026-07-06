from fastapi import APIRouter, HTTPException, File, UploadFile, Form, Depends
from typing import Optional
from app.models.response_models import ObjectDetectionResponse
from app.services.ai_service import get_ai_service, AIService
from app.core.logging import logger
router = APIRouter(prefix='/detect', tags=['object-detection'])

@router.post('/objects', response_model=ObjectDetectionResponse)
def detect_objects(file: UploadFile=File(...), confidence_threshold: float=Form(0.25), iou_threshold: float=Form(0.45), annotate: bool=Form(False), include_images: bool=Form(False), service: AIService=Depends(get_ai_service)):
    try:
        logger.info(f'Processing object detection request (annotate={annotate})')
        image_bytes = file.file.read()
        result = service.detect_objects(image_bytes=image_bytes, confidence_threshold=confidence_threshold, iou_threshold=iou_threshold, annotate=annotate, include_images=include_images)
        return ObjectDetectionResponse(success=True, **result)
    except Exception as e:
        logger.error(f'Object detection endpoint error: {str(e)}')
        raise HTTPException(status_code=400, detail=str(e))

@router.post('/annotate', response_model=ObjectDetectionResponse)
def detect_and_annotate(file: UploadFile=File(...), confidence_threshold: float=Form(0.25), iou_threshold: float=Form(0.45), include_images: bool=Form(True), service: AIService=Depends(get_ai_service)):
    try:
        logger.info('Processing object detection with annotation request')
        image_bytes = file.file.read()
        result = service.detect_objects(image_bytes=image_bytes, confidence_threshold=confidence_threshold, iou_threshold=iou_threshold, annotate=True, include_images=include_images)
        return ObjectDetectionResponse(success=True, **result)
    except Exception as e:
        logger.error(f'Object detection with annotation endpoint error: {str(e)}')
        raise HTTPException(status_code=400, detail=str(e))