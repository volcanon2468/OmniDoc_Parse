from pydantic import BaseModel, Field

class NLPRequest(BaseModel):
    text: str = Field(..., description='Text to analyze')
    extract_entities: bool = Field(True, description='Extract named entities')
    extract_keywords: bool = Field(True, description='Extract keywords')
    classify_document: bool = Field(True, description='Classify document type')