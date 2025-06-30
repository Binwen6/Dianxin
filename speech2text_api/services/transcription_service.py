import sys
import os
import time
import logging
from pathlib import Path
from typing import Dict, Optional

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from speech2text.whisper import WhisperTranscriber
from ..core.config import settings

logger = logging.getLogger(__name__)

class TranscriptionService:
    """转录服务类，封装Whisper转录功能"""
    
    def __init__(self):
        """初始化转录服务"""
        try:
            logger.info("初始化转录服务...")
            self.transcriber = WhisperTranscriber(
                model_id=settings.WHISPER_MODEL,
                return_timestamps=settings.RETURN_TIMESTAMPS
            )
            logger.info("转录服务初始化完成")
        except Exception as e:
            logger.error(f"初始化转录服务失败: {str(e)}")
            raise
    
    def transcribe_file(self, audio_path: str) -> Dict:
        """
        转录单个音频文件
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            包含转录结果的字典
        """
        start_time = time.time()
        
        try:
            logger.info(f"开始转录文件: {audio_path}")
            
            # 执行转录
            result = self.transcriber.transcribe_file(
                audio_path, 
                chunk_length_s=settings.CHUNK_LENGTH_S
            )
            
            # 计算处理时间
            processing_time = time.time() - start_time
            result["processing_time"] = round(processing_time, 2)
            
            logger.info(f"转录完成，耗时: {processing_time:.2f}秒")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"转录文件失败: {str(e)}")
            
            return {
                "file_path": str(audio_path),
                "file_name": Path(audio_path).name,
                "error": str(e),
                "processing_time": round(processing_time, 2),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "model_used": settings.WHISPER_MODEL
            }
    
    def get_model_info(self) -> Dict:
        """获取模型信息"""
        return {
            "model_id": settings.WHISPER_MODEL,
            "return_timestamps": settings.RETURN_TIMESTAMPS,
            "chunk_length_s": settings.CHUNK_LENGTH_S,
            "device": self.transcriber.device if hasattr(self, 'transcriber') else "unknown"
        }
    
    def is_ready(self) -> bool:
        """检查服务是否就绪"""
        try:
            return hasattr(self, 'transcriber') and self.transcriber is not None
        except Exception:
            return False 