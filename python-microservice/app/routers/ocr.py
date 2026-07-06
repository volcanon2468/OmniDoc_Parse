from fastapi import APIRouter, HTTPException, File, UploadFile, Form, Depends
from typing import Optional
from app.models.response_models import OCRResponse, TextExtractionResponse
from app.services.ai_service import get_ai_service, AIService
from app.core.logging import logger
router = APIRouter(prefix='/ocr', tags=['ocr'])

@router.post('/extract', response_model=OCRResponse)
def extract_text_advanced(file: UploadFile=File(...), mode: str=Form('balanced'), engine: Optional[str]=Form(None), service: AIService=Depends(get_ai_service)):
    try:
        logger.info(f'Processing OCR request with mode: {mode}')
        image_bytes = file.file.read()
        result = service.extract_text(image_bytes=image_bytes, mode=mode, engine=engine)
        return OCRResponse(success=True, **result)
    except Exception as e:
        logger.error(f'OCR endpoint error: {str(e)}')
        raise HTTPException(status_code=400, detail=str(e))

@router.post('/extract-simple', response_model=TextExtractionResponse)
def extract_text_simple(file: UploadFile=File(...), service: AIService=Depends(get_ai_service)):
    try:
        logger.info('Processing simple text extraction request')
        image_bytes = file.file.read()
        result = service.extract_text(image_bytes=image_bytes, mode='fast')
        return TextExtractionResponse(success=True, text=result.get('text'), confidence=result.get('confidence'))
    except Exception as e:
        logger.error(f'Text extraction endpoint error: {str(e)}')
        raise HTTPException(status_code=400, detail=str(e))