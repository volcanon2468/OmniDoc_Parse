from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
import warnings
from app.core.config import settings
from app.core.logging import logger
from app.models.response_models import HealthResponse
from app.routers import preprocessing, ocr, nlp, detection
warnings.filterwarnings('ignore', message='.*pin_memory.*', category=UserWarning)
ai_service = None

class MaxUploadSizeMiddleware(BaseHTTPMiddleware):

    def __init__(self, app, max_size: int):
        super().__init__(app)
        self.max_size = max_size

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get('content-length')
        if content_length and int(content_length) > self.max_size:
            return Response(content='{"success": false, "error": "Request body too large"}', status_code=413, media_type='application/json')
        return await call_next(request)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global ai_service
    logger.info('Starting OmniDoc Parse Microservice...')
    logger.info(f'Environment: {settings.environment}')
    try:
        from app.services.ai_service import AIService
        ai_service = AIService()
        logger.info('AI models loaded and ready')
    except Exception as e:
        logger.error(f'Failed to initialize AI models: {str(e)}')
        raise
    yield
    logger.info('Shutting down OmniDoc Parse Microservice...')
app = FastAPI(title=settings.app_name, version=settings.app_version, description='AI-powered document processing microservice providing OCR, NLP, and object detection capabilities', docs_url='/docs', redoc_url='/redoc', lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_credentials=True, allow_methods=['*'], allow_headers=['*'])
app.add_middleware(MaxUploadSizeMiddleware, max_size=settings.max_upload_size)

@app.get('/', include_in_schema=False)
def root():
    return {'message': 'OmniDoc Parse Microservice', 'version': settings.app_version, 'docs': '/docs'}

@app.get('/health', response_model=HealthResponse, tags=['health'])
def health_check():
    models_loaded = ai_service is not None and ai_service.is_healthy()
    return HealthResponse(status='healthy' if models_loaded else 'degraded', version=settings.app_version, models_loaded=models_loaded)
app.include_router(preprocessing.router, prefix=settings.api_v1_prefix)
app.include_router(ocr.router, prefix=settings.api_v1_prefix)
app.include_router(nlp.router, prefix=settings.api_v1_prefix)
app.include_router(detection.router, prefix=settings.api_v1_prefix)

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f'Uncaught exception: {str(exc)}')
    return JSONResponse(status_code=500, content={'success': False, 'error': 'Internal server error', 'detail': str(exc) if settings.environment == 'development' else 'An error occurred'})
if __name__ == '__main__':
    import uvicorn
    uvicorn.run('app.main:app', host='0.0.0.0', port=8000, reload=True if settings.environment == 'development' else False)