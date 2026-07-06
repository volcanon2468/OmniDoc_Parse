from fastapi import APIRouter, HTTPException, File, UploadFile, Form, Depends
from typing import Optional
from app.models.response_models import PreprocessingResponse
from app.services.ai_service import get_ai_service, AIService
from app.core.logging import logger
router = APIRouter(prefix='/preprocess', tags=['preprocessing'])

@router.post('/', response_model=PreprocessingResponse)
def preprocess_image(file: UploadFile=File(...), max_dimension: Optional[int]=Form(2048), noise_threshold: Optional[int]=Form(100), include_images: bool=Form(False), service: AIService=Depends(get_ai_service)):
    try:
        logger.info('Processing preprocessing request')
        image_bytes = file.file.read()
        result = service.preprocess_image(image_bytes=image_bytes, max_dimension=max_dimension, noise_threshold=noise_threshold, include_images=include_images)
        return PreprocessingResponse(success=True, **result)
    except Exception as e:
        logger.error(f'Preprocessing endpoint error: {str(e)}')
        raise HTTPException(status_code=400, detail=str(e))