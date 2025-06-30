from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import tempfile
import logging
from typing import Optional
import uvicorn
from pathlib import Path

from .core.config import settings
from .core.auth import verify_api_key
from .services.transcription_service import TranscriptionService
from .models.request_models import AudioToTextRequest
from .models.response_models import AudioToTextResponse, ErrorResponse

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="Speech to Text API",
    description="语音转文字API服务，支持多种音频格式",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化转录服务
transcription_service = TranscriptionService()

@app.get("/")
async def root():
    """健康检查端点"""
    return {"message": "Speech to Text API is running", "version": "1.0.0"}

@app.post("/audio-to-text", response_model=AudioToTextResponse)
async def audio_to_text(
    file: UploadFile = File(..., description="语音文件。支持格式：mp3, mp4, mpeg, mpga, m4a, wav, webm"),
    user: str = Form(..., description="用户标识"),
    authorization: str = Depends(verify_api_key)
):
    """
    将语音文件转换为文字
    
    - **file**: 语音文件，支持格式：mp3, mp4, mpeg, mpga, m4a, wav, webm，大小限制：15MB
    - **user**: 用户标识
    - **authorization**: API密钥，格式：Bearer {API_KEY}
    
    返回转换后的文字内容
    """
    try:
        # 验证文件格式
        allowed_extensions = {'.mp3', '.mp4', '.mpeg', '.mpga', '.m4a', '.wav', '.webm'}
        file_extension = Path(file.filename).suffix.lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的文件格式: {file_extension}。支持的格式: {', '.join(allowed_extensions)}"
            )
        
        # 验证文件大小 (15MB = 15 * 1024 * 1024 bytes)
        max_size = 15 * 1024 * 1024
        if file.size and file.size > max_size:
            raise HTTPException(
                status_code=400,
                detail=f"文件大小超过限制: {file.size} bytes。最大允许: {max_size} bytes"
            )
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            # 读取上传的文件内容
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # 执行转录
            logger.info(f"开始处理用户 {user} 的文件: {file.filename}")
            result = transcription_service.transcribe_file(temp_file_path)
            
            # 检查是否有错误
            if "error" in result:
                raise HTTPException(
                    status_code=500,
                    detail=f"转录失败: {result['error']}"
                )
            
            # 构建响应
            response = AudioToTextResponse(
                text=result["transcription"],
                user=user,
                file_name=file.filename,
                file_size=result.get("file_size_bytes", 0),
                model_used=result.get("model_used", ""),
                processing_time=result.get("processing_time"),
                timestamp=result.get("timestamp")
            )
            
            logger.info(f"用户 {user} 的文件转录完成: {file.filename}")
            return response
            
        finally:
            # 清理临时文件
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"处理文件时发生错误: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"服务器内部错误: {str(e)}"
        )

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """全局异常处理器"""
    logger.error(f"未处理的异常: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "服务器内部错误"}
    )

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    ) 