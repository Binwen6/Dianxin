from pydantic import BaseModel, Field
from typing import Optional

class AudioToTextRequest(BaseModel):
    """音频转文字请求模型"""
    
    user: str = Field(..., description="用户标识", min_length=1, max_length=100)
    
    class Config:
        schema_extra = {
            "example": {
                "user": "user123"
            }
        } 