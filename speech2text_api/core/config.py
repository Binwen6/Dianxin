import os
from typing import Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """应用配置设置"""
    
    # API设置
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    
    # API密钥设置
    API_KEY: str = "your-secret-api-key-here"
    
    # 文件上传设置
    MAX_FILE_SIZE: int = 15 * 1024 * 1024  # 15MB
    ALLOWED_EXTENSIONS: set = {'.mp3', '.mp4', '.mpeg', '.mpga', '.m4a', '.wav', '.webm'}
    
    # Whisper模型设置
    WHISPER_MODEL: str = "openai/whisper-large-v3-turbo"
    RETURN_TIMESTAMPS: bool = False
    CHUNK_LENGTH_S: int = 30
    
    # 日志设置
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # 临时文件设置
    TEMP_DIR: Optional[str] = None
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# 创建全局设置实例
settings = Settings() 