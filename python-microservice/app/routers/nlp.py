from fastapi import APIRouter, HTTPException, Depends
from app.models.request_models import NLPRequest
from app.models.response_models import NLPResponse
from app.services.ai_service import get_ai_service, AIService
from app.core.logging import logger
router = APIRouter(prefix='/nlp', tags=['nlp'])

@router.post('/analyze', response_model=NLPResponse)
def analyze_text(request: NLPRequest, service: AIService=Depends(get_ai_service)):
    try:
        logger.info('Processing NLP analysis request')
        result = service.analyze_text(text=request.text, extract_entities=request.extract_entities, extract_keywords=request.extract_keywords, classify_document=request.classify_document)
        return NLPResponse(success=True, **result)
    except Exception as e:
        logger.error(f'NLP endpoint error: {str(e)}')
        raise HTTPException(status_code=400, detail=str(e))

@router.post('/entities', response_model=NLPResponse)
def extract_entities(request: NLPRequest, service: AIService=Depends(get_ai_service)):
    try:
        logger.info('Processing entity extraction request')
        full_result = service.analyze_text(text=request.text, extract_entities=True, extract_keywords=False, classify_document=False)
        return NLPResponse(success=True, entities=full_result.get('entities'))
    except Exception as e:
        logger.error(f'Entity extraction endpoint error: {str(e)}')
        raise HTTPException(status_code=400, detail=str(e))

@router.post('/keywords', response_model=NLPResponse)
def extract_keywords(request: NLPRequest, service: AIService=Depends(get_ai_service)):
    try:
        logger.info('Processing keyword extraction request')
        full_result = service.analyze_text(text=request.text, extract_entities=False, extract_keywords=True, classify_document=False)
        return NLPResponse(success=True, keywords=full_result.get('keywords'))
    except Exception as e:
        logger.error(f'Keyword extraction endpoint error: {str(e)}')
        raise HTTPException(status_code=400, detail=str(e))