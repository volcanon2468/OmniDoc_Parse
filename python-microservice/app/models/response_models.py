from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any

class BaseResponse(BaseModel):
    success: bool = Field(..., description='Whether the operation was successful')
    message: Optional[str] = Field(None, description='Optional message')
    error: Optional[str] = Field(None, description='Error message if failed')

class HealthResponse(BaseModel):
    status: str = Field(..., description='Service status')
    version: str = Field(..., description='API version')
    models_loaded: bool = Field(..., description='Whether AI models are loaded')

class PreprocessingResponse(BaseResponse):
    data: Optional[Dict[str, Any]] = Field(None, description='Preprocessing results')
    quality_score: Optional[float] = Field(None, description='Image quality score')
    skew_angle: Optional[float] = Field(None, description='Detected skew angle')
    binarization_method: Optional[str] = Field(None, description='Binarization method used')

class OCRResponse(BaseResponse):
    data: Optional[Dict[str, Any]] = Field(None, description='OCR results')
    text: Optional[str] = Field(None, description='Extracted text')
    confidence: Optional[float] = Field(None, description='Overall confidence score')
    word_count: Optional[int] = Field(None, description='Number of words detected')
    engine_used: Optional[str] = Field(None, description='OCR engine used')

class NLPResponse(BaseResponse):
    data: Optional[Dict[str, Any]] = Field(None, description='NLP analysis results')
    entities: Optional[Dict[str, List[Dict]]] = Field(None, description='Named entities')
    keywords: Optional[List[Dict]] = Field(None, description='Extracted keywords')
    classification: Optional[Dict[str, Any]] = Field(None, description='Document classification')
    statistics: Optional[Dict[str, Any]] = Field(None, description='Text statistics')
    patterns: Optional[Dict[str, List[str]]] = Field(None, description='Detected patterns')

class ObjectDetectionResponse(BaseResponse):
    data: Optional[Dict[str, Any]] = Field(None, description='Detection results')
    objects: Optional[List[Dict]] = Field(None, description='Detected objects')
    object_count: Optional[int] = Field(None, description='Number of objects detected')
    annotated_image: Optional[str] = Field(None, description='Base64 encoded annotated image')
    processing_time: Optional[float] = Field(None, description='Processing time in seconds')

class TextExtractionResponse(BaseResponse):
    text: Optional[str] = Field(None, description='Extracted text')
    confidence: Optional[float] = Field(None, description='Confidence score')