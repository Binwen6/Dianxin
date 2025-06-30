from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class AudioToTextResponse(BaseModel):
    """音频转文字响应模型"""
    
    text: str = Field(..., description="转换后的文字内容")
    user: str = Field(..., description="用户标识")
    file_name: str = Field(..., description="原始文件名")
    file_size: int = Field(..., description="文件大小（字节）")
    model_used: str = Field(..., description="使用的模型")
    processing_time: Optional[float] = Field(None, description="处理时间（秒）")
    timestamp: str = Field(..., description="处理时间戳")
    
    class Config:
        schema_extra = {
            "example": {
                "text": "这是转换后的文字内容",
                "user": "user123",
                "file_name": "audio.mp3",
                "file_size": 1024000,
                "model_used": "openai/whisper-large-v3-turbo",
                "processing_time": 2.5,
                "timestamp": "2024-01-01T12:00:00"
            }
        }

class ErrorResponse(BaseModel):
    """错误响应模型"""
    
    detail: str = Field(..., description="错误详情")
    error_code: Optional[str] = Field(None, description="错误代码")
    timestamp: str = Field(..., description="错误发生时间")
    
    class Config:
        schema_extra = {
            "example": {
                "detail": "文件格式不支持",
                "error_code": "INVALID_FORMAT",
                "timestamp": "2024-01-01T12:00:00"
            }
        }

class HealthCheckResponse(BaseModel):
    """健康检查响应模型"""
    
    message: str = Field(..., description="服务状态消息")
    version: str = Field(..., description="API版本")
    status: str = Field(..., description="服务状态")
    
    class Config:
        schema_extra = {
            "example": {
                "message": "Speech to Text API is running",
                "version": "1.0.0",
                "status": "healthy"
            }
        } 