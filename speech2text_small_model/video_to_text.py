#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SenseVoice 工作流语音转文字处理器
专门设计用于视频转音频 -> 语音转文字 -> 文字转PPT 工作流的中间环节
支持视频文件自动提取音频并进行语音转文字
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import functools
import math
import tempfile
import shutil
import re
import subprocess

# 尝试导入python-dotenv库来读取.env文件
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    print("警告: python-dotenv 库未安装，无法读取.env文件。请安装: pip install python-dotenv")

# 加载.env文件
if DOTENV_AVAILABLE:
    # 方法1: 尝试从当前目录加载.env文件
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"已加载.env文件: {env_path}")
    else:
        # 方法2: 尝试从项目根目录加载.env文件
        project_root = Path(__file__).parent.parent
        root_env_path = project_root / '.env'
        if root_env_path.exists():
            load_dotenv(root_env_path)
            print(f"已加载项目根目录.env文件: {root_env_path}")
        else:
            # 方法3: 尝试从工作目录加载
            load_dotenv()
            print("尝试从工作目录加载.env文件")
else:
    print("python-dotenv不可用，无法加载.env文件")

from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

# 尝试导入音频处理库
try:
    import librosa
    import soundfile as sf
    AUDIO_LIBS_AVAILABLE = True
except ImportError:
    AUDIO_LIBS_AVAILABLE = False
    print("警告: librosa 或 soundfile 库未安装，智能分割功能将受限。请安装: pip install librosa soundfile")

# 尝试导入通义千问API
try:
    import dashscope
    from dashscope import Generation
    DASHSCOPE_AVAILABLE = True
except ImportError:
    DASHSCOPE_AVAILABLE = False
    print("警告: dashscope 库未安装，通义千问内容纠错功能将不可用。请安装: pip install dashscope")


def extract_audio_from_video(video_path: Union[str, Path], 
                            output_dir: Optional[Union[str, Path]] = None,
                            audio_quality: str = "4") -> Path:
    """
    从视频文件中提取音频
    
    Args:
        video_path: 视频文件路径
        output_dir: 音频输出目录，如果为None则使用临时目录
        audio_quality: 音频质量 (0-9, 0最高质量，9最低质量)
        
    Returns:
        提取的音频文件路径
        
    Raises:
        subprocess.CalledProcessError: ffmpeg执行失败
        FileNotFoundError: 视频文件不存在或ffmpeg未安装
    """
    video_path = Path(video_path)
    
    if not video_path.exists():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")
    
    # 检查ffmpeg是否可用
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise FileNotFoundError("ffmpeg未安装或不在PATH中。请安装ffmpeg: https://ffmpeg.org/download.html")
    
    # 确定输出目录
    if output_dir is None:
        output_dir = video_path.parent / "extracted_audios"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成音频文件名
    base_name = video_path.stem
    audio_path = output_dir / f"{base_name}.mp3"
    
    # 如果音频文件已存在，添加时间戳避免冲突
    if audio_path.exists():
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        audio_path = output_dir / f"{base_name}_{timestamp}.mp3"
    
    print(f"正在从视频提取音频: {video_path.name} -> {audio_path.name}")
    
    try:
        # 使用ffmpeg提取音频
        subprocess.run([
            "ffmpeg", 
            "-i", str(video_path),      # 输入视频文件
            "-vn",                      # 禁用视频
            "-acodec", "libmp3lame",    # 使用MP3编码器
            "-q:a", audio_quality,      # 音频质量
            "-y",                       # 覆盖输出文件
            str(audio_path)
        ], check=True, capture_output=True, text=True)
        
        print(f"音频提取成功: {audio_path}")
        return audio_path
        
    except subprocess.CalledProcessError as e:
        error_msg = f"ffmpeg执行失败: {e.stderr if e.stderr else str(e)}"
        print(f"错误: {error_msg}")
        raise subprocess.CalledProcessError(e.returncode, e.cmd, error_msg)


def is_video_file(file_path: Union[str, Path]) -> bool:
    """
    检查文件是否为视频文件
    
    Args:
        file_path: 文件路径
        
    Returns:
        是否为视频文件
    """
    video_extensions = {
        '.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v',
        '.3gp', '.3g2', '.asf', '.rm', '.rmvb', '.vob', '.ts', '.mts',
        '.m2ts', '.f4v', '.divx', '.xvid', '.ogv'
    }
    
    file_path = Path(file_path)
    return file_path.suffix.lower() in video_extensions


def is_audio_file(file_path: Union[str, Path]) -> bool:
    """
    检查文件是否为音频文件
    
    Args:
        file_path: 文件路径
        
    Returns:
        是否为音频文件
    """
    audio_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg', '.wma'}
    
    file_path = Path(file_path)
    return file_path.suffix.lower() in audio_extensions


def count_chinese_words(text: str) -> int:
    """
    准确统计中文字数（包括中文、英文、数字等）
    
    Args:
        text: 要统计的文本
        
    Returns:
        字数统计
    """
    if not text:
        return 0
    
    # 移除多余的空白字符
    text = re.sub(r'\s+', ' ', text.strip())
    
    # 统计中文字符（包括中文标点）
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]', text))
    
    # 统计英文单词
    english_words = len(re.findall(r'[a-zA-Z]+', text))
    
    # 统计数字
    numbers = len(re.findall(r'\d+', text))
    
    # 统计其他字符（标点符号等）
    other_chars = len(re.findall(r'[^\u4e00-\u9fff\u3000-\u303f\uff00-\uffef\w\s]', text))
    
    # 总字数 = 中文字符 + 英文单词 + 数字 + 其他字符
    total_words = chinese_chars + english_words + numbers + other_chars
    
    return total_words


def count_characters(text: str) -> int:
    """
    统计字符数（包括所有字符）
    
    Args:
        text: 要统计的文本
        
    Returns:
        字符数统计
    """
    if not text:
        return 0
    
    # 移除多余的空白字符但保留换行符
    text = re.sub(r'[ \t]+', ' ', text)
    
    return len(text)


def ensure_text_completeness(text: str, max_length: int = 10000) -> str:
    """
    确保文本完整性，避免截断
    
    Args:
        text: 原始文本
        max_length: 最大长度限制
        
    Returns:
        处理后的完整文本
    """
    if not text:
        return text
    
    # 移除末尾的不完整句子
    text = text.strip()
    
    # 如果文本以不完整的标点符号结尾，尝试找到最后一个完整句子
    incomplete_endings = ['，', ',', '。', '.', '！', '!', '？', '?', '；', ';', '：', ':']
    
    # 检查是否以不完整的标点符号结尾
    if text and text[-1] not in incomplete_endings:
        # 尝试找到最后一个完整句子的结束位置
        last_complete_pos = -1
        for ending in incomplete_endings:
            pos = text.rfind(ending)
            if pos > last_complete_pos:
                last_complete_pos = pos
        
        # 如果找到了完整句子的结束位置，截取到那里
        if last_complete_pos > 0:
            text = text[:last_complete_pos + 1]
    
    # 限制最大长度
    if len(text) > max_length:
        # 在最大长度范围内找到最后一个完整句子
        truncated_text = text[:max_length]
        last_complete_pos = -1
        for ending in incomplete_endings:
            pos = truncated_text.rfind(ending)
            if pos > last_complete_pos:
                last_complete_pos = pos
        
        if last_complete_pos > 0:
            text = truncated_text[:last_complete_pos + 1]
        else:
            text = truncated_text
    
    return text


def get_safe_max_workers(config: Dict[str, Any], default: int = 4) -> int:
    """
    安全地获取最大并发数，确保不为None
    
    Args:
        config: 配置字典
        default: 默认值
        
    Returns:
        最大并发数
    """
    max_workers = config.get('max_workers', default)
    if max_workers is None:
        max_workers = default
    return max_workers


def _correct_text_with_dashscope(text: str, config: Dict[str, Any]) -> str:
    """
    使用通义千问API对文本进行纠错
    
    Args:
        text: 需要纠错的文本
        config: 配置字典，包含通义千问API配置
        
    Returns:
        纠错后的文本
    """
    if not DASHSCOPE_AVAILABLE:
        return text
    
    if not text or not text.strip():
        return text
    
    # 检查通义千问配置
    api_key = config.get('dashscope_api_key') or os.environ.get('DASHSCOPE_API_KEY')
    if not api_key:
        print("警告: 未配置通义千问API密钥，跳过内容纠错")
        print(f"调试信息: config中的dashscope_api_key = {config.get('dashscope_api_key')}")
        print(f"调试信息: 环境变量DASHSCOPE_API_KEY = {os.environ.get('DASHSCOPE_API_KEY')}")
        return text
    
    try:
        # 设置通义千问API密钥
        dashscope.api_key = 'sk-9df5ae7489d44f18902719b7b1489b69'
        
        # 构建纠错提示
        system_prompt = """你是一个专业的文本纠错助手。请对输入的语音转文字结果进行纠错，主要任务包括：

1. 修正语法错误和用词错误
2. 修正标点符号使用
3. 保持原文的意思和语调
4. 修正同音字错误
5. 补充缺失的标点符号
6. 保持文本的自然流畅性

请直接返回纠错后的文本，不要添加任何解释或标记。"""

        # 调用通义千问API
        response = Generation.call(
            model=config.get('dashscope_model', 'qwen-turbo'),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"请对以下语音转文字结果进行纠错：\n\n{text}"}
            ],
            temperature=config.get('dashscope_temperature', 0.1),
            max_tokens=config.get('dashscope_max_tokens', 10000),
            timeout=config.get('dashscope_timeout', 30)
        )
        
        # 检查响应状态
        print(f"通义千问API响应状态码: {response.status_code}")
        print(f"通义千问API响应类型: {type(response)}")
        print(f"通义千问API响应属性: {dir(response)}")
        
        if response.status_code == 200:
            try:
                # 尝试多种方式获取响应内容
                corrected_text = None
                
                # 方法1: 直接访问response的文本内容（使用字典方式）
                if hasattr(response, 'get') and callable(response.get):
                    # response是一个字典对象
                    corrected_text = response.get('text', '').strip()
                    if corrected_text:
                        print("通过response['text']获取到内容")
                
                # 方法2: 访问output字段
                if not corrected_text and hasattr(response, 'get') and callable(response.get):
                    output = response.get('output', {})
                    if isinstance(output, dict):
                        # 尝试从output中获取文本
                        corrected_text = output.get('text', '').strip()
                        if corrected_text:
                            print("通过response['output']['text']获取到内容")
                        
                        # 如果没有text，尝试choices结构
                        if not corrected_text and 'choices' in output:
                            choices = output['choices']
                            if choices and len(choices) > 0:
                                choice = choices[0]
                                if isinstance(choice, dict) and 'message' in choice:
                                    message = choice['message']
                                    if isinstance(message, dict) and 'content' in message:
                                        corrected_text = message['content'].strip()
                                        print("通过response['output']['choices'][0]['message']['content']获取到内容")
                
                # 方法3: 尝试将response转换为字典
                if not corrected_text:
                    try:
                        # 如果response是对象，尝试获取其字典表示
                        if hasattr(response, '__dict__'):
                            response_dict = response.__dict__
                        elif hasattr(response, 'get') and callable(response.get):
                            response_dict = response
                        else:
                            response_dict = {}
                        
                        print(f"response字典内容: {response_dict}")
                        
                        # 查找可能的文本字段
                        for key, value in response_dict.items():
                            if isinstance(value, str) and len(value) > 10:  # 假设文本内容长度大于10
                                corrected_text = value.strip()
                                print(f"通过response['{key}']获取到内容")
                                break
                    except Exception as e:
                        print(f"转换response为字典失败: {e}")
                
                # 检查是否获取到内容
                if corrected_text:
                    # 如果返回的文本为空或异常，返回原文本
                    if not corrected_text or corrected_text == text:
                        print("通义千问API返回内容与原文本相同或为空，使用原文本")
                        return text
                    
                    return corrected_text
                else:
                    print("通义千问API响应格式错误：无法获取文本内容")
                    print(f"response内容: {response}")
                    return text
            except Exception as e:
                print(f"解析通义千问API响应时出错: {e}")
                return text
        else:
            print(f"通义千问API调用失败: 状态码 {response.status_code}")
            if hasattr(response, 'message'):
                print(f"错误信息: {response.message}")
            return text
        
    except Exception as e:
        # 如果API调用失败，返回原文本
        print(f"通义千问API纠错失败: {e}")
        return text


def _process_single_audio_worker(media_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    独立的工作进程函数，用于处理单个音频或视频文件
    
    Args:
        media_path: 音频或视频文件路径
        config: 配置字典
        
    Returns:
        处理结果字典
    """
    input_path = Path(media_path)
    
    if not input_path.exists():
        return {
            'source_file': str(input_path),
            'file_name': input_path.name,
            'transcription': '',
            'processing_time': 0,
            'timestamp': datetime.now().isoformat(),
            'status': 'error',
            'error': f"文件不存在: {input_path}",
            'metadata': {}
        }
    
    original_file_path = input_path
    extracted_audio_path = None
    
    # 检查输入文件类型并进行相应处理
    if is_video_file(input_path):
        try:
            # 从视频提取音频
            extracted_audio_path = extract_audio_from_video(
                input_path, 
                output_dir=input_path.parent / "extracted_audios",
                audio_quality=config.get('video_audio_quality', '4')
            )
            audio_path = extracted_audio_path
        except Exception as e:
            return {
                'source_file': str(original_file_path),
                'file_name': original_file_path.name,
                'transcription': '',
                'processing_time': 0,
                'timestamp': datetime.now().isoformat(),
                'status': 'error',
                'error': f"视频音频提取失败: {str(e)}",
                'metadata': {
                    'input_type': 'video',
                    'extraction_attempted': True,
                    'extraction_successful': False
                }
            }
    elif is_audio_file(input_path):
        audio_path = input_path
    else:
        supported_formats = "支持的格式: 音频(.mp3, .wav, .m4a, .flac, .aac, .ogg, .wma) 和视频(.mp4, .avi, .mkv, .mov等)"
        return {
            'source_file': str(original_file_path),
            'file_name': original_file_path.name,
            'transcription': '',
            'processing_time': 0,
            'timestamp': datetime.now().isoformat(),
            'status': 'error',
            'error': f"不支持的文件格式: {input_path.suffix}。{supported_formats}",
            'metadata': {
                'input_type': 'unsupported',
                'file_extension': input_path.suffix
            }
        }
    
    try:
        # 开始计时（包括模型加载时间）
        start_time = time.time()
        
        # 在子进程中创建新的模型实例，禁用更新检查以加快启动速度
        model = AutoModel(
            model=config['model_dir'],
            trust_remote_code=True,
            remote_code=config.get('remote_code', './model.py'),
            vad_model=config.get('vad_model', 'fsmn-vad'),
            vad_kwargs=config.get('vad_kwargs', {
                "max_single_segment_time": 30000
            }),
            device=config.get('device', 'cuda:0'),
            disable_update=True,  # 禁用更新检查
        )
        
        # 执行转写
        res = model.generate(
            input=str(audio_path),
            cache={},
            language=config.get('language', 'auto'),
            use_itn=config.get('use_itn', True),
            batch_size_s=config.get('batch_size_s', 60),
            merge_vad=config.get('merge_vad', True),
            merge_length_s=config.get('merge_length_s', 15),
        )
        
        # 后处理文本
        text = rich_transcription_postprocess(res[0]["text"])
        
        # 使用通义千问API进行内容纠错
        if config.get('enable_dashscope_correction', False):
            correction_start_time = time.time()
            original_text = text
            text = _correct_text_with_dashscope(text, config)
            correction_time = time.time() - correction_start_time
            correction_info = {
                'enabled': True,
                'correction_time': correction_time,
                'original_text': original_text,
                'corrected_text': text,
                'text_changed': original_text != text
            }
        else:
            correction_info = {
                'enabled': False,
                'correction_time': 0,
                'original_text': text,
                'corrected_text': text,
                'text_changed': False
            }
        
        processing_time = time.time() - start_time
        
        # 估算音频时长
        def estimate_duration(audio_path: Path) -> float:
            try:
                file_size = audio_path.stat().st_size
                estimated_duration = file_size / (128 * 1024 / 8)
                return max(0, estimated_duration)
            except:
                return 0.0
        
        # 构建结果
        result = {
            'source_file': str(audio_path),
            'file_name': audio_path.name,
            'file_size': audio_path.stat().st_size,
            'original_source_file': str(original_file_path),
            'original_file_name': original_file_path.name,
            'transcription': text,  # 这里已经是纠错后的文本
            'processing_time': processing_time,
            'timestamp': datetime.now().isoformat(),
            'status': 'success',
            'metadata': {
                'model_used': config['model_dir'],
                'language': config.get('language', 'auto'),
                'device': config.get('device', 'cuda:0'),
                'audio_duration': estimate_duration(audio_path),
                'word_count': count_chinese_words(text) if text else 0,
                'character_count': count_characters(text) if text else 0,
                'dashscope_correction': correction_info,
                'input_type': 'video' if extracted_audio_path else 'audio',
                'extraction_successful': extracted_audio_path is not None,
                'extracted_audio_path': str(extracted_audio_path) if extracted_audio_path else None,
                'audio_quality': config.get('video_audio_quality', '4') if extracted_audio_path else None
            }
        }
        
        return result
        
    except Exception as e:
        return {
            'source_file': str(audio_path),
            'file_name': audio_path.name,
            'original_source_file': str(original_file_path),
            'original_file_name': original_file_path.name,
            'transcription': '',
            'processing_time': 0,
            'timestamp': datetime.now().isoformat(),
            'status': 'error',
            'error': str(e),
            'metadata': {
                'input_type': 'video' if extracted_audio_path else 'audio',
                'extraction_successful': extracted_audio_path is not None
            }
        }
    finally:
        # 可选：清理临时提取的音频文件
        if (extracted_audio_path and 
            config.get('cleanup_extracted_audio', False) and 
            extracted_audio_path.exists()):
            try:
                extracted_audio_path.unlink()
            except Exception:
                pass  # 忽略清理失败


class WorkflowSpeechProcessor:
    """工作流语音转文字处理器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化工作流处理器
        
        Args:
            config: 配置字典，如果为None则使用默认配置
        """
        # 获取默认配置并用传入的配置更新
        self.config = self._get_default_config()
        if config:
            self.config.update(config)
        self.model = None
        self.logger = self._setup_logger()
        
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        # 检测系统类型，在macOS上默认使用CPU
        import platform
        default_device = 'cpu' if platform.system() == 'Darwin' else 'cuda:0'
        
        return {
            'model_dir': 'iic/SenseVoiceSmall',
            'remote_code': './model.py',
            'vad_model': 'fsmn-vad',
            'vad_kwargs': {
                'max_single_segment_time': 30000
            },
            'device': default_device,
            'language': 'auto',
            'use_itn': True,
            'batch_size_s': 60,
            'merge_vad': True,
            'merge_length_s': 15,
            'output_format': 'json',  # json, txt, both
            'include_timestamps': True,
            'segment_by_speaker': False,
            'min_segment_length': 1.0,  # 最小片段长度（秒）
            'max_segment_length': 30.0,  # 最大片段长度（秒）
            # 并行处理配置
            'parallel_processing': True,  # 是否启用并行处理
            'max_workers': 4,  # 最大工作进程数，统一限制为3个
            'chunk_size': 1,  # 每个进程处理的文件数量
            # 智能分割配置
            'enable_smart_segmentation': True,  # 是否启用智能分割
            'max_audio_duration_for_single_process': 600.0,  # 超过此时长（秒）的音频将进行智能分割
            'processing_time_ratio': 1/17,  # 音频处理时间与原始时间的比值（研究结果）
            'min_segment_duration': 30.0,  # 最小分割段时长（秒）
            'max_segment_duration': 300.0,  # 最大分割段时长（秒）
            'overlap_duration': 5.0,  # 分割段之间的重叠时长（秒）
            'include_segment_markers': False,  # 是否在合并文本中包含分段标识
            # 通义千问内容纠错配置
            'enable_dashscope_correction': False,  # 是否启用通义千问内容纠错
            'dashscope_api_key': None,  # 通义千问API密钥
            'dashscope_model': 'qwen-turbo',  # 使用的模型
            'dashscope_temperature': 0.1,  # 温度参数
            'dashscope_max_tokens': 10000,  # 最大token数
            'dashscope_timeout': 30,  # API超时时间（秒）
            # 视频处理配置
            'video_audio_quality': '4',  # 视频音频提取质量 (0-9, 0最高质量，9最低质量)
            'cleanup_extracted_audio': False,  # 是否清理临时提取的音频文件
        }
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger('WorkflowSpeechProcessor')
        logger.setLevel(logging.INFO)
        
        # 避免重复添加处理器
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def load_model(self):
        """加载 SenseVoice 模型"""
        if self.model is not None:
            return
            
        try:
            self.logger.info("正在加载 SenseVoice 模型...")
            self.model = AutoModel(
                model=self.config['model_dir'],
                trust_remote_code=True,
                remote_code=self.config.get('remote_code', './model.py'),
                vad_model=self.config.get('vad_model', 'fsmn-vad'),
                vad_kwargs=self.config.get('vad_kwargs', {
                    "max_single_segment_time": 30000
                }),
                device=self.config.get('device', 'cuda:0'),
            )
            self.logger.info("模型加载成功")
        except Exception as e:
            self.logger.error(f"模型加载失败: {e}")
            raise
    
    def process_single_audio(self, audio_path: Union[str, Path]) -> Dict[str, Any]:
        """
        处理单个音频文件或视频文件（支持智能分割和视频自动提取音频）
        
        Args:
            audio_path: 音频文件路径或视频文件路径
            
        Returns:
            处理结果字典，包含转写文本和元数据
        """
        input_path = Path(audio_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"文件不存在: {input_path}")
        
        original_file_path = input_path
        extracted_audio_path = None
        
        # 检查输入文件类型
        if is_video_file(input_path):
            self.logger.info(f"检测到视频文件: {input_path.name}，开始提取音频")
            try:
                # 从视频提取音频
                extracted_audio_path = extract_audio_from_video(
                    input_path, 
                    output_dir=input_path.parent / "extracted_audios",
                    audio_quality=self.config.get('video_audio_quality', '4')
                )
                audio_path = extracted_audio_path
                self.logger.info(f"视频音频提取完成: {extracted_audio_path.name}")
            except Exception as e:
                self.logger.error(f"视频音频提取失败: {e}")
                return {
                    'source_file': str(original_file_path),
                    'file_name': original_file_path.name,
                    'transcription': '',
                    'processing_time': 0,
                    'timestamp': datetime.now().isoformat(),
                    'status': 'error',
                    'error': f"视频音频提取失败: {str(e)}",
                    'metadata': {
                        'input_type': 'video',
                        'extraction_attempted': True,
                        'extraction_successful': False
                    }
                }
        elif is_audio_file(input_path):
            self.logger.info(f"检测到音频文件: {input_path.name}")
            audio_path = input_path
        else:
            # 不支持的文件类型
            supported_formats = "支持的格式: 音频(.mp3, .wav, .m4a, .flac, .aac, .ogg, .wma) 和视频(.mp4, .avi, .mkv, .mov等)"
            error_msg = f"不支持的文件格式: {input_path.suffix}。{supported_formats}"
            self.logger.error(error_msg)
            return {
                'source_file': str(original_file_path),
                'file_name': original_file_path.name,
                'transcription': '',
                'processing_time': 0,
                'timestamp': datetime.now().isoformat(),
                'status': 'error',
                'error': error_msg,
                'metadata': {
                    'input_type': 'unsupported',
                    'file_extension': input_path.suffix,
                    'supported_formats': supported_formats
                }
            }
        
        try:
            # 处理音频文件
            if self._should_segment_audio(audio_path):
                result = self._process_audio_with_segmentation(audio_path)
            else:
                result = self._process_audio_directly(audio_path)
            
            # 更新结果中的源文件信息
            result['original_source_file'] = str(original_file_path)
            result['original_file_name'] = original_file_path.name
            
            # 添加视频相关的元数据
            if extracted_audio_path:
                result['metadata']['input_type'] = 'video'
                result['metadata']['extracted_audio_path'] = str(extracted_audio_path)
                result['metadata']['extraction_successful'] = True
                result['metadata']['audio_quality'] = self.config.get('video_audio_quality', '4')
            else:
                result['metadata']['input_type'] = 'audio'
                result['metadata']['extraction_successful'] = False
            
            return result
            
        except Exception as e:
            self.logger.error(f"文件处理失败: {e}")
            return {
                'source_file': str(original_file_path),
                'file_name': original_file_path.name,
                'original_source_file': str(original_file_path),
                'original_file_name': original_file_path.name,
                'transcription': '',
                'processing_time': 0,
                'timestamp': datetime.now().isoformat(),
                'status': 'error',
                'error': str(e),
                'metadata': {
                    'input_type': 'video' if extracted_audio_path else 'audio',
                    'extraction_successful': extracted_audio_path is not None
                }
            }
        finally:
            # 可选：清理临时提取的音频文件
            if (extracted_audio_path and 
                self.config.get('cleanup_extracted_audio', False) and 
                extracted_audio_path.exists()):
                try:
                    extracted_audio_path.unlink()
                    self.logger.info(f"已清理临时音频文件: {extracted_audio_path.name}")
                except Exception as e:
                    self.logger.warning(f"清理临时音频文件失败: {e}")
    
    def _process_audio_directly(self, audio_path: Path) -> Dict[str, Any]:
        """
        直接处理音频文件（不分割）
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            处理结果字典
        """
        # 加载模型
        self.load_model()
        
        try:
            self.logger.info(f"正在直接处理音频文件: {audio_path.name}")
            start_time = time.time()
            
            # 执行转写
            res = self.model.generate(
                input=str(audio_path),
                cache={},
                language=self.config.get('language', 'auto'),
                use_itn=self.config.get('use_itn', True),
                batch_size_s=self.config.get('batch_size_s', 60),
                merge_vad=self.config.get('merge_vad', True),
                merge_length_s=self.config.get('merge_length_s', 15),
            )
            
            # 后处理文本
            text = rich_transcription_postprocess(res[0]["text"])
            
            # 确保文本完整性
            text = ensure_text_completeness(text)
            
            # 使用通义千问API进行内容纠错
            if self.config.get('enable_dashscope_correction', False):
                correction_start_time = time.time()
                original_text = text
                text = _correct_text_with_dashscope(text, self.config)
                correction_time = time.time() - correction_start_time
                correction_info = {
                    'enabled': True,
                    'correction_time': correction_time,
                    'original_text': original_text,
                    'corrected_text': text,
                    'text_changed': original_text != text
                }
                self.logger.info(f"通义千问纠错完成: {audio_path.name} (纠错耗时: {correction_time:.2f}s, 文本是否变化: {original_text != text})")
            else:
                correction_info = {
                    'enabled': False,
                    'correction_time': 0,
                    'original_text': text,
                    'corrected_text': text,
                    'text_changed': False
                }
            
            processing_time = time.time() - start_time
            
            # 构建结果
            result = {
                'source_file': str(audio_path),
                'file_name': audio_path.name,
                'file_size': audio_path.stat().st_size,
                'transcription': text,  # 这里已经是纠错后的文本
                'processing_time': processing_time,
                'timestamp': datetime.now().isoformat(),
                'status': 'success',
                'metadata': {
                    'model_used': self.config['model_dir'],
                    'language': self.config.get('language', 'auto'),
                    'device': self.config.get('device', 'cuda:0'),
                    'audio_duration': self._estimate_duration(audio_path),
                    'word_count': count_chinese_words(text) if text else 0,
                    'character_count': count_characters(text) if text else 0,
                    'dashscope_correction': correction_info,
                    'segmentation_info': {
                        'total_segments': 1,
                        'successful_segments': 1,
                        'failed_segments': 0,
                        'segmentation_method': 'direct_processing'
                    }
                }
            }
            
            self.logger.info(f"音频处理完成: {audio_path.name} (耗时: {processing_time:.2f}s)")
            return result
            
        except Exception as e:
            self.logger.error(f"音频处理失败 {audio_path}: {e}")
            return {
                'source_file': str(audio_path),
                'file_name': audio_path.name,
                'transcription': '',
                'processing_time': 0,
                'timestamp': datetime.now().isoformat(),
                'status': 'error',
                'error': str(e),
                'metadata': {}
            }
    
    def _process_audio_with_segmentation(self, audio_path: Path) -> Dict[str, Any]:
        """
        使用智能分割处理长音频文件
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            处理结果字典
        """
        try:
            start_time = time.time()
            self.logger.info(f"开始智能分割处理长音频: {audio_path.name}")
            
            # 获取音频时长
            audio_duration = self._estimate_duration(audio_path)
            
            # 计算最佳分割段数
            processing_time_ratio = self.config.get('processing_time_ratio', 1/17)
            num_segments = self._calculate_optimal_segments(audio_duration, processing_time_ratio)
            
            # 获取最大并发数，确保不为None
            max_workers = get_safe_max_workers(self.config, 4)
            self.logger.info(f"音频时长: {audio_duration:.2f}s, 最终分段数: {num_segments}, 最大并发数: {max_workers}")
            
            # 分割音频
            segment_paths = self._segment_audio(audio_path, num_segments)
            
            if len(segment_paths) <= 1:
                self.logger.warning("音频分割失败或只有一段，回退到直接处理")
                return self._process_audio_directly(audio_path)
            
            # 使用现有的多音频并行处理机制处理分割段
            self.logger.info(f"开始并行处理 {len(segment_paths)} 个分割段")
            
            # 检查是否启用并行处理
            if (self.config.get('parallel_processing', True) and 
                len(segment_paths) > 1 and 
                len(segment_paths) >= 2):  # 分割段数量>=2时使用并行
                
                # 使用专门的分段并行处理逻辑，确保顺序正确
                segment_results = self._process_segments_parallel(segment_paths)
            else:
                # 串行处理分割段
                if len(segment_paths) > 1:
                    self.logger.info(f"分割段数量较少({len(segment_paths)})，使用串行处理以避免模型加载开销")
                segment_results = []
                for i, segment_path in enumerate(segment_paths):
                    self.logger.info(f"处理分割段 {i+1}/{len(segment_paths)}: {segment_path.name}")
                    segment_result = self._process_audio_directly(segment_path)
                    segment_results.append(segment_result)
            
            # 合并结果
            merged_result = self._merge_segment_results(segment_results, audio_path)
            
            # 计算总处理时间（包括分割和合并的时间）
            total_processing_time = time.time() - start_time
            merged_result['processing_time'] = total_processing_time
            
            self.logger.info(f"智能分割处理完成: {audio_path.name} (总耗时: {total_processing_time:.2f}s)")
            return merged_result
            
        except Exception as e:
            self.logger.error(f"智能分割处理失败 {audio_path}: {e}")
            # 回退到直接处理
            self.logger.info("回退到直接处理模式")
            return self._process_audio_directly(audio_path)
    
    def _process_segments_parallel(self, segment_paths: List[Path]) -> List[Dict[str, Any]]:
        """
        并行处理音频分段，确保结果按正确顺序返回
        
        Args:
            segment_paths: 分段音频文件路径列表（已按顺序排列）
            
        Returns:
            按顺序排列的处理结果列表
        """
        self.logger.info("使用分段并行处理模式")
        
        # 限制最大进程数为配置的最大并发数，且不超过分段数
        config_max_workers = get_safe_max_workers(self.config, 4)
        max_workers = min(config_max_workers, len(segment_paths))
        self.logger.info(f"使用 {max_workers} 个工作进程进行分段并行处理 (分段数: {len(segment_paths)}, 最大并发数: {config_max_workers})")
        
        # 准备参数，保持顺序信息
        segment_tasks = [(i, str(path)) for i, path in enumerate(segment_paths)]
        
        # 使用进程池进行并行处理
        start_time = time.time()
        results = [None] * len(segment_paths)  # 预分配结果列表，保持顺序
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务，保持索引信息
            future_to_index = {
                executor.submit(_process_single_audio_worker, audio_path, self.config): (index, audio_path)
                for index, audio_path in segment_tasks
            }
            
            # 收集结果，按原始顺序排列
            completed_count = 0
            
            for future in as_completed(future_to_index):
                index, audio_path = future_to_index[future]
                try:
                    result = future.result()
                    results[index] = result  # 按原始索引放置结果
                    completed_count += 1
                    
                    if result['status'] == 'success':
                        self.logger.info(f"完成分段 [{completed_count}/{len(segment_paths)}]: {Path(audio_path).name} (耗时: {result['processing_time']:.2f}s)")
                    else:
                        self.logger.error(f"分段失败 [{completed_count}/{len(segment_paths)}]: {Path(audio_path).name} - {result.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    self.logger.error(f"分段处理失败 {audio_path}: {e}")
                    results[index] = {
                        'source_file': audio_path,
                        'file_name': Path(audio_path).name,
                        'transcription': '',
                        'processing_time': 0,
                        'timestamp': datetime.now().isoformat(),
                        'status': 'error',
                        'error': str(e),
                        'metadata': {}
                    }
                    completed_count += 1
        
        total_processing_time = time.time() - start_time
        self.logger.info(f"分段并行处理完成，总耗时: {total_processing_time:.2f}s")
        
        return results
    
    def process_audio_directory(self, audio_dir: Union[str, Path], 
                              output_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        处理音频目录中的所有音频文件
        
        Args:
            audio_dir: 音频文件目录
            output_dir: 输出目录，如果为None则使用默认输出目录
            
        Returns:
            批量处理结果字典
        """
        audio_dir = Path(audio_dir)
        output_dir = Path(output_dir) if output_dir else Path('output/speech2text')
        
        if not audio_dir.exists():
            raise FileNotFoundError(f"音频目录不存在: {audio_dir}")
        
        # 获取所有音频和视频文件
        audio_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg', '.wma'}
        video_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v',
                           '.3gp', '.3g2', '.asf', '.rm', '.rmvb', '.vob', '.ts', '.mts',
                           '.m2ts', '.f4v', '.divx', '.xvid', '.ogv'}
        
        media_files = []
        
        # 查找音频文件
        for ext in audio_extensions:
            media_files.extend(audio_dir.rglob(f'*{ext}'))
            media_files.extend(audio_dir.rglob(f'*{ext.upper()}'))
        
        # 查找视频文件
        for ext in video_extensions:
            media_files.extend(audio_dir.rglob(f'*{ext}'))
            media_files.extend(audio_dir.rglob(f'*{ext.upper()}'))
        
        media_files = sorted(media_files)
        
        if not media_files:
            self.logger.warning(f"在目录 {audio_dir} 中未找到音频或视频文件")
            return {
                'status': 'no_files_found',
                'processed_files': [],
                'total_files': 0,
                'successful': 0,
                'failed': 0
            }
        
        # 统计文件类型
        audio_count = sum(1 for f in media_files if is_audio_file(f))
        video_count = sum(1 for f in media_files if is_video_file(f))
        
        self.logger.info(f"找到 {len(media_files)} 个媒体文件 (音频: {audio_count}, 视频: {video_count})")
        
        # 检查是否启用并行处理
        # 对于小文件数量，串行处理可能更快（避免模型加载开销）
        if (self.config.get('parallel_processing', True) and 
            len(media_files) > 1 and 
            len(media_files) >= 4):  # 只有文件数量>=4时才使用并行
            return self._process_audio_directory_parallel(media_files, output_dir)
        else:
            if len(media_files) > 1:
                self.logger.info(f"文件数量较少({len(media_files)})，使用串行处理以避免模型加载开销")
            return self._process_audio_directory_sequential(media_files, output_dir)
    
    def _process_audio_directory_parallel(self, media_files: List[Path], 
                                        output_dir: Path) -> Dict[str, Any]:
        """
        并行处理目录中的所有音频和视频文件
        
        Args:
            media_files: 音频和视频文件列表
            output_dir: 输出目录
            
        Returns:
            批量处理结果字典
        """
        self.logger.info("使用并行处理模式")
        
        # 确定工作进程数 - 统一限制为最多3个进程
        max_workers = self.config.get('max_workers')
        if max_workers is None:
            # 统一限制为最多3个进程，无论GPU还是CPU模式
            max_workers = min(4, len(media_files))
        else:
            # 如果用户指定了max_workers，也要确保不超过3个
            max_workers = min(max_workers, 4, len(media_files))
        
        self.logger.info(f"使用 {max_workers} 个工作进程进行并行处理")
        self.logger.info(f"设备: {self.config.get('device', 'cuda:0')}")
        
        # 准备参数
        media_paths = [str(f) for f in media_files]
        
        # 使用进程池进行并行处理
        start_time = time.time()
        results = []
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_media = {
                executor.submit(_process_single_audio_worker, media_path, self.config): media_path 
                for media_path in media_paths
            }
            
            # 收集结果
            completed_count = 0
            individual_times = []
            
            for future in as_completed(future_to_media):
                media_path = future_to_media[future]
                try:
                    result = future.result()
                    results.append(result)
                    completed_count += 1
                    
                    if result['status'] == 'success':
                        individual_times.append(result['processing_time'])
                        self.logger.info(f"完成 [{completed_count}/{len(media_files)}]: {Path(media_path).name} (耗时: {result['processing_time']:.2f}s)")
                    else:
                        self.logger.error(f"失败 [{completed_count}/{len(media_files)}]: {Path(media_path).name} - {result.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    self.logger.error(f"处理失败 {media_path}: {e}")
                    results.append({
                        'source_file': media_path,
                        'file_name': Path(media_path).name,
                        'transcription': '',
                        'processing_time': 0,
                        'timestamp': datetime.now().isoformat(),
                        'status': 'error',
                        'error': str(e),
                        'metadata': {}
                    })
                    completed_count += 1
        
        total_processing_time = time.time() - start_time
        
        # 计算并行效率 - 修正计算方式
        if individual_times:
            # 串行处理的真实时间估算：
            # 模型加载时间（约7秒）+ 所有文件的实际处理时间
            estimated_model_load_time = 7.0  # 模型加载时间
            actual_processing_times = sum(individual_times)
            estimated_sequential_time = estimated_model_load_time + actual_processing_times
            
            parallel_efficiency = estimated_sequential_time / total_processing_time if total_processing_time > 0 else 0
            self.logger.info(f"并行效率估算: {parallel_efficiency:.2f}x")
            self.logger.info(f"  预估串行时间: {estimated_sequential_time:.2f}s (模型加载: {estimated_model_load_time:.1f}s + 处理: {actual_processing_times:.1f}s)")
            self.logger.info(f"  实际并行时间: {total_processing_time:.2f}s")
            
            # 分析性能瓶颈
            if parallel_efficiency < 1.0:
                self.logger.info("并行效率低于1.0，可能的原因:")
                self.logger.info("1. 模型加载开销大于并行收益")
                self.logger.info("2. 文件数量较少，建议使用串行处理")
                self.logger.info("3. 系统资源限制")
                self.logger.info("4. 预估串行时间可能不准确")
            else:
                self.logger.info(f"并行处理可能有效，预估节省了 {parallel_efficiency - 1:.1f} 倍时间")
                self.logger.info("注意：这是基于估算的串行时间，实际效果需要对比测试验证")
        
        # 统计信息
        successful = [r for r in results if r['status'] == 'success']
        failed = [r for r in results if r['status'] == 'error']
        
        batch_result = {
            'status': 'completed',
            'source_directory': str(media_files[0].parent),
            'output_directory': str(output_dir),
            'processed_files': results,
            'total_files': len(results),
            'successful': len(successful),
            'failed': len(failed),
            'total_processing_time': total_processing_time,
            'average_processing_time': total_processing_time / len(results) if results else 0,
            'timestamp': datetime.now().isoformat(),
            'processing_mode': 'parallel',
            'max_workers': max_workers,
            'estimated_parallel_efficiency': parallel_efficiency if individual_times else 0,
            'summary': {
                'total_words': sum(r.get('metadata', {}).get('word_count', 0) for r in successful),
                'total_characters': sum(r.get('metadata', {}).get('character_count', 0) for r in successful),
                'total_audio_duration': sum(r.get('metadata', {}).get('audio_duration', 0) for r in successful)
            }
        }
        
        # 保存结果
        self._save_batch_results(batch_result, output_dir)
        
        return batch_result
    
    def _process_audio_directory_sequential(self, media_files: List[Path], 
                                          output_dir: Path) -> Dict[str, Any]:
        """
        串行处理目录中的所有音频和视频文件
        
        Args:
            media_files: 音频和视频文件列表
            output_dir: 输出目录
            
        Returns:
            批量处理结果字典
        """
        self.logger.info("使用串行处理模式")
        
        # 批量处理
        results = []
        total_processing_time = 0
        
        for i, media_file in enumerate(media_files, 1):
            self.logger.info(f"处理进度 [{i}/{len(media_files)}]: {media_file.name}")
            result = self.process_single_audio(media_file)
            results.append(result)
            total_processing_time += result.get('processing_time', 0)
        
        # 统计信息
        successful = [r for r in results if r['status'] == 'success']
        failed = [r for r in results if r['status'] == 'error']
        
        batch_result = {
            'status': 'completed',
            'source_directory': str(media_files[0].parent),
            'output_directory': str(output_dir),
            'processed_files': results,
            'total_files': len(results),
            'successful': len(successful),
            'failed': len(failed),
            'total_processing_time': total_processing_time,
            'average_processing_time': total_processing_time / len(results) if results else 0,
            'timestamp': datetime.now().isoformat(),
            'processing_mode': 'sequential',
            'summary': {
                'total_words': sum(r.get('metadata', {}).get('word_count', 0) for r in successful),
                'total_characters': sum(r.get('metadata', {}).get('character_count', 0) for r in successful),
                'total_audio_duration': sum(r.get('metadata', {}).get('audio_duration', 0) for r in successful)
            }
        }
        
        # 保存结果
        self._save_batch_results(batch_result, output_dir)
        
        return batch_result
    
    def _save_batch_results(self, batch_result: Dict[str, Any], output_dir: Path):
        """保存批量处理结果"""
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 为每个音频文件创建独立的子目录
        for result in batch_result['processed_files']:
            if result['status'] == 'success':
                # 创建以音频文件名命名的子目录
                audio_name = Path(result['file_name']).stem  # 去掉扩展名
                audio_dir = output_dir / audio_name
                audio_dir.mkdir(parents=True, exist_ok=True)
                
                # 保存该音频的详细结果
                json_file = audio_dir / f'{audio_name}_result.json'
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                
                # 保存该音频的transcripts.json格式
                transcripts_file = audio_dir / 'transcripts.json'
                transcripts_data = self._format_single_transcript_json(result)
                with open(transcripts_file, 'w', encoding='utf-8') as f:
                    json.dump(transcripts_data, f, ensure_ascii=False, indent=2)
                
                # 保存该音频的纯文本格式
                txt_file = audio_dir / f'{audio_name}_content.txt'
                with open(txt_file, 'w', encoding='utf-8') as f:
                    f.write(f"音频文件: {result['file_name']}\n")
                    f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(f"转写内容:\n{result['transcription']}\n\n")
                    f.write(f"字数: {result['metadata']['word_count']}\n")
                    f.write(f"字符数: {result['metadata']['character_count']}\n")
                    f.write(f"处理时间: {result['processing_time']:.2f}秒\n")
                    f.write(f"音频时长: {result['metadata']['audio_duration']:.2f}秒\n")
                    
                    # 添加通义千问纠错信息
                    correction_info = result['metadata'].get('dashscope_correction', {})
                    if correction_info.get('enabled', False):
                        f.write(f"通义千问纠错: 启用\n")
                        f.write(f"纠错耗时: {correction_info.get('correction_time', 0):.2f}秒\n")
                        f.write(f"文本是否变化: {'是' if correction_info.get('text_changed', False) else '否'}\n")
                        if correction_info.get('text_changed', False):
                            f.write(f"原始文本:\n{correction_info.get('original_text', '')}\n\n")
                            f.write(f"纠错后文本:\n{correction_info.get('corrected_text', '')}\n\n")
                    else:
                        f.write(f"通义千问纠错: 未启用\n")
        
        # 保存整体批量处理结果（JSON格式）
        batch_json_file = output_dir / f'batch_results_{timestamp}.json'
        with open(batch_json_file, 'w', encoding='utf-8') as f:
            json.dump(batch_result, f, ensure_ascii=False, indent=2)
        
        # 保存整体transcripts.json格式（包含所有音频）
        all_transcripts_file = output_dir / 'all_transcripts.json'
        all_transcripts_data = self._format_transcripts_json(batch_result)
        with open(all_transcripts_file, 'w', encoding='utf-8') as f:
            json.dump(all_transcripts_data, f, ensure_ascii=False, indent=2)
        
        # 保存整体纯文本格式（用于PPT生成）
        all_txt_file = output_dir / f'all_content_{timestamp}.txt'
        with open(all_txt_file, 'w', encoding='utf-8') as f:
            f.write(f"批量语音转文字结果 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n\n")
            
            for result in batch_result['processed_files']:
                if result['status'] == 'success':
                    f.write(f"文件: {result['file_name']}\n")
                    f.write(f"转写内容:\n{result['transcription']}\n")
                    f.write(f"字数: {result['metadata']['word_count']}\n")
                    f.write(f"处理时间: {result['processing_time']:.2f}秒\n")
                    f.write("-" * 40 + "\n\n")
        
        # 保存工作流元数据（供下游PPT生成使用）
        workflow_file = output_dir / f'workflow_metadata_{timestamp}.json'
        workflow_metadata = {
            'workflow_step': 'speech2text',
            'input_type': 'audio',
            'output_type': 'text',
            'total_files': batch_result['total_files'],
            'successful_files': batch_result['successful'],
            'total_words': batch_result['summary']['total_words'],
            'total_characters': batch_result['summary']['total_characters'],
            'total_audio_duration': batch_result['summary']['total_audio_duration'],
            'processing_timestamp': batch_result['timestamp'],
            'output_structure': 'individual_directories',
            'output_files': {
                'batch_results': str(batch_json_file),
                'all_transcripts_json': str(all_transcripts_file),
                'all_text_content': str(all_txt_file),
                'workflow_metadata': str(workflow_file),
                'individual_directories': [str(output_dir / Path(r['file_name']).stem) for r in batch_result['processed_files'] if r['status'] == 'success']
            }
        }
        
        with open(workflow_file, 'w', encoding='utf-8') as f:
            json.dump(workflow_metadata, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"结果已保存到: {output_dir}")
        self.logger.info(f"处理完成: {batch_result['successful']}/{batch_result['total_files']} 个文件成功")
        self.logger.info(f"每个音频文件的结果已保存到独立子目录中")
        self.logger.info(f"整体结果文件: {all_transcripts_file}")
    
    def _format_single_transcript_json(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        格式化单个音频的transcripts.json数据
        
        Args:
            result: 单个音频处理结果
            
        Returns:
            格式化的单个音频transcripts.json数据
        """
        transcript_entry = {
            "file_name": result.get('file_name', 'Unknown'),
            "file_path": result.get('source_file', 'Unknown'),
            "file_size_bytes": result.get('file_size', 0),
            "timestamp": result.get('timestamp', ''),
            "model_used": result.get('metadata', {}).get('model_used', self.config['model_dir']),
            "device_used": result.get('metadata', {}).get('device', self.config.get('device', 'cuda:0')),
            "language": result.get('metadata', {}).get('language', self.config.get('language', 'auto'))
        }
        
        if result.get('status') == 'error':
            transcript_entry["status"] = "error"
            transcript_entry["error"] = result.get('error', 'Unknown error')
            transcript_entry["transcription"] = ""
            transcript_entry["segments"] = []
            transcript_entry["word_count"] = 0
            transcript_entry["character_count"] = 0
            transcript_entry["audio_duration"] = 0
            transcript_entry["processing_time"] = 0
        else:
            transcript_entry["status"] = "success"
            transcript_entry["transcription"] = result.get('transcription', '')
            transcript_entry["processing_time"] = result.get('processing_time', 0)
            transcript_entry["word_count"] = result.get('metadata', {}).get('word_count', 0)
            transcript_entry["character_count"] = result.get('metadata', {}).get('character_count', 0)
            transcript_entry["audio_duration"] = result.get('metadata', {}).get('audio_duration', 0)
            
            # 生成时间戳信息（基于音频时长和文本长度估算）
            segments = self._generate_timestamps_for_transcript(
                result.get('transcription', ''),
                result.get('metadata', {}).get('audio_duration', 0)
            )
            transcript_entry["segments"] = segments
        
        return {
            "metadata": {
                "total_files": 1,
                "successful_transcriptions": 1 if result.get('status') == 'success' else 0,
                "failed_transcriptions": 1 if result.get('status') == 'error' else 0,
                "total_words": result.get('metadata', {}).get('word_count', 0),
                "total_characters": result.get('metadata', {}).get('character_count', 0),
                "total_audio_duration": result.get('metadata', {}).get('audio_duration', 0),
                "total_processing_time": result.get('processing_time', 0),
                "timestamp": result.get('timestamp', ''),
                "model_used": result.get('metadata', {}).get('model_used', self.config['model_dir']),
                "device_used": result.get('metadata', {}).get('device', self.config.get('device', 'cuda:0')),
                "language": result.get('metadata', {}).get('language', self.config.get('language', 'auto')),
                "workflow_step": "speech2text"
            },
            "transcripts": [transcript_entry]
        }
    
    def _format_transcripts_json(self, batch_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        格式化transcripts.json数据
        
        Args:
            batch_result: 批量处理结果
            
        Returns:
            格式化的transcripts.json数据
        """
        transcripts_data = {
            "metadata": {
                "total_files": batch_result['total_files'],
                "successful_transcriptions": batch_result['successful'],
                "failed_transcriptions": batch_result['failed'],
                "total_words": batch_result['summary']['total_words'],
                "total_characters": batch_result['summary']['total_characters'],
                "total_audio_duration": batch_result['summary']['total_audio_duration'],
                "total_processing_time": batch_result['total_processing_time'],
                "average_processing_time": batch_result['average_processing_time'],
                "timestamp": batch_result['timestamp'],
                "model_used": self.config['model_dir'],
                "device_used": self.config.get('device', 'cuda:0'),
                "language": self.config.get('language', 'auto'),
                "workflow_step": "speech2text"
            },
            "transcripts": []
        }
        
        for result in batch_result['processed_files']:
            transcript_entry = {
                "file_name": result.get('file_name', 'Unknown'),
                "file_path": result.get('source_file', 'Unknown'),
                "file_size_bytes": result.get('file_size', 0),
                "timestamp": result.get('timestamp', ''),
                "model_used": result.get('metadata', {}).get('model_used', self.config['model_dir']),
                "device_used": result.get('metadata', {}).get('device', self.config.get('device', 'cuda:0')),
                "language": result.get('metadata', {}).get('language', self.config.get('language', 'auto'))
            }
            
            if result.get('status') == 'error':
                transcript_entry["status"] = "error"
                transcript_entry["error"] = result.get('error', 'Unknown error')
                transcript_entry["transcription"] = ""
                transcript_entry["segments"] = []
                transcript_entry["word_count"] = 0
                transcript_entry["character_count"] = 0
                transcript_entry["audio_duration"] = 0
                transcript_entry["processing_time"] = 0
            else:
                transcript_entry["status"] = "success"
                transcript_entry["transcription"] = result.get('transcription', '')
                transcript_entry["processing_time"] = result.get('processing_time', 0)
                transcript_entry["word_count"] = result.get('metadata', {}).get('word_count', 0)
                transcript_entry["character_count"] = result.get('metadata', {}).get('character_count', 0)
                transcript_entry["audio_duration"] = result.get('metadata', {}).get('audio_duration', 0)
                
                # 生成时间戳信息（基于音频时长和文本长度估算）
                segments = self._generate_timestamps_for_transcript(
                    result.get('transcription', ''),
                    result.get('metadata', {}).get('audio_duration', 0)
                )
                transcript_entry["segments"] = segments
            
            transcripts_data["transcripts"].append(transcript_entry)
        
        return transcripts_data
    
    def _generate_timestamps_for_transcript(self, text: str, audio_duration: float) -> List[Dict[str, Any]]:
        """
        为转录文本生成时间戳信息
        
        Args:
            text: 转录文本
            audio_duration: 音频时长（秒）
            
        Returns:
            时间戳段落列表
        """
        if not text or audio_duration <= 0:
            return []
        
        # 简单的文本分段（按句号、问号、感叹号分割）
        import re
        sentences = re.split(r'[。！？.!?]', text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return [{
                "text": text,
                "start": 0.0,
                "end": audio_duration,
                "duration": audio_duration
            }]
        
        # 按句子长度比例分配时间
        total_chars = sum(len(s) for s in sentences)
        segments = []
        current_time = 0.0
        
        for sentence in sentences:
            if total_chars > 0:
                segment_duration = (len(sentence) / total_chars) * audio_duration
            else:
                segment_duration = audio_duration / len(sentences)
            
            segment = {
                "text": sentence,
                "start": round(current_time, 2),
                "end": round(current_time + segment_duration, 2),
                "duration": round(segment_duration, 2)
            }
            segments.append(segment)
            current_time += segment_duration
        
        return segments
    
    def _estimate_duration(self, audio_path: Path) -> float:
        """估算音频文件时长（优先使用librosa，回退到文件大小估算）"""
        try:
            if AUDIO_LIBS_AVAILABLE:
                # 使用librosa获取准确的音频时长
                audio, sr = librosa.load(str(audio_path), sr=None)
                duration = len(audio) / sr
                return duration
            else:
                # 回退到文件大小估算
                file_size = audio_path.stat().st_size
                # 假设平均比特率约为128kbps
                estimated_duration = file_size / (128 * 1024 / 8)
                return max(0, estimated_duration)
        except Exception as e:
            self.logger.warning(f"音频时长检测失败，使用文件大小估算: {e}")
            try:
                file_size = audio_path.stat().st_size
                estimated_duration = file_size / (128 * 1024 / 8)
                return max(0, estimated_duration)
            except:
                return 0.0
    
    def _calculate_optimal_segments(self, audio_duration: float, processing_time_ratio: float = 1/17) -> int:
        """
        根据研究结果计算最佳分割段数，并确保不超过最大并发数
        
        Args:
            audio_duration: 音频时长（秒）
            processing_time_ratio: 处理时间与音频时长的比值（默认1:17）
            
        Returns:
            最佳分割段数
        """
        # 根据研究：根号下（C比7）的最近整数为最佳等切分段数
        # C = 不包括模型加载时间的单线程纯处理花费的时间
        # C ≈ 音频时长 * 处理时间比值
        C = audio_duration * processing_time_ratio
        
        # 计算根号下（C比7）
        optimal_segments = math.sqrt(C / 7)
        
        # 取最近整数
        optimal_segments = round(optimal_segments)
        
        # 获取最大并发数，确保不为None
        max_workers = get_safe_max_workers(self.config, 4)
        
        # 确保至少为1，最多不超过音频时长除以最小段时长，且不超过最大并发数
        min_segments = 1
        max_segments_by_duration = max(1, int(audio_duration / self.config.get('min_segment_duration', 30.0)))
        max_segments_by_concurrency = max_workers
        
        # 取最小值，确保不超过最大并发数
        max_segments = min(max_segments_by_duration, max_segments_by_concurrency)
        optimal_segments = max(min_segments, min(optimal_segments, max_segments))
        
        self.logger.info(f"音频时长: {audio_duration:.2f}s, 计算得到最佳分割段数: {optimal_segments} (最大并发数: {max_workers})")
        return optimal_segments
    
    def _should_segment_audio(self, audio_path: Path) -> bool:
        """
        判断是否需要对音频进行智能分割
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            是否需要分割
        """
        if not self.config.get('enable_smart_segmentation', True):
            self.logger.info(f"智能分割功能已禁用")
            return False
        
        # 检查并行处理是否被禁用
        if not self.config.get('parallel_processing', True):
            self.logger.info(f"并行处理已禁用，智能分割功能也被禁用")
            return False
        
        # 获取音频时长
        audio_duration = self._estimate_duration(audio_path)
        
        # 使用研究公式计算最佳分割段数
        processing_time_ratio = self.config.get('processing_time_ratio', 1/17)
        optimal_segments = self._calculate_optimal_segments(audio_duration, processing_time_ratio)
        
        # 获取最大并发数，确保不为None
        max_workers = get_safe_max_workers(self.config, 4)
        
        # 只有当最佳分割段数 > 2 且不超过最大并发数时才真正启用智能分割
        should_segment = optimal_segments > 2 and optimal_segments <= max_workers
        
        self.logger.info(f"音频 {audio_path.name} 时长: {audio_duration:.2f}s, 计算得到最佳段数: {optimal_segments}, 最大并发数: {max_workers}, 是否启用智能分割: {should_segment}")
        
        return should_segment
    
    def _segment_audio(self, audio_path: Path, num_segments: int) -> List[Path]:
        """
        将音频文件分割为指定数量的段，确保完整覆盖整个音频时长
        
        Args:
            audio_path: 音频文件路径
            num_segments: 分割段数
            
        Returns:
            分割后的音频文件路径列表
        """
        if not AUDIO_LIBS_AVAILABLE:
            self.logger.error("librosa 或 soundfile 库未安装，无法进行音频分割")
            return [audio_path]
        
        try:
            # 加载音频
            audio, sr = librosa.load(str(audio_path), sr=None)
            total_duration = len(audio) / sr
            total_samples = len(audio)
            
            # 计算重叠时长
            overlap_duration = self.config.get('overlap_duration', 5.0)
            overlap_samples = int(overlap_duration * sr)
            
            # 确保每段时长在合理范围内
            min_segment_duration = self.config.get('min_segment_duration', 30.0)
            max_segment_duration = self.config.get('max_segment_duration', 300.0)
            
            # 计算每段的基础时长（不包含重叠）
            base_segment_duration = total_duration / num_segments
            base_segment_samples = int(base_segment_duration * sr)
            
            # 调整基础段时长，确保在合理范围内
            min_segment_samples = int(min_segment_duration * sr)
            max_segment_samples = int(max_segment_duration * sr)
            
            base_segment_samples = max(min_segment_samples, 
                                     min(base_segment_samples, max_segment_samples))
            
            # 创建临时目录
            temp_dir = Path(tempfile.mkdtemp(prefix=f"audio_segments_{audio_path.stem}_"))
            segment_paths = []
            
            config_max_workers = get_safe_max_workers(self.config, 4)
            self.logger.info(f"开始分割音频 {audio_path.name} 为 {num_segments} 段 (总时长: {total_duration:.2f}s, 最大并发数: {config_max_workers})")
            
            # 重新计算实际的分割点，确保完整覆盖
            segment_boundaries = []
            for i in range(num_segments + 1):
                if i == 0:
                    # 第一段开始
                    start_sample = 0
                elif i == num_segments:
                    # 最后一段结束
                    start_sample = total_samples
                else:
                    # 中间分割点
                    start_sample = int(i * total_samples / num_segments)
                segment_boundaries.append(start_sample)
            
            # 创建分割段，确保完整覆盖
            for i in range(num_segments):
                # 当前段的起始和结束位置
                start_sample = segment_boundaries[i]
                end_sample = segment_boundaries[i + 1]
                
                # 添加重叠（除了第一段和最后一段）
                if i > 0:  # 不是第一段，向前扩展
                    start_sample = max(0, start_sample - overlap_samples)
                if i < num_segments - 1:  # 不是最后一段，向后扩展
                    end_sample = min(total_samples, end_sample + overlap_samples)
                
                # 确保段长度合理
                segment_samples = end_sample - start_sample
                if segment_samples < min_segment_samples:
                    # 如果段太短，扩展它
                    if i == 0:  # 第一段
                        end_sample = min(total_samples, start_sample + min_segment_samples)
                    elif i == num_segments - 1:  # 最后一段
                        start_sample = max(0, end_sample - min_segment_samples)
                    else:  # 中间段，向两边扩展
                        extension = (min_segment_samples - segment_samples) // 2
                        start_sample = max(0, start_sample - extension)
                        end_sample = min(total_samples, end_sample + extension)
                
                # 提取音频段
                segment_audio = audio[start_sample:end_sample]
                
                # 保存音频段
                segment_filename = f"{audio_path.stem}_segment_{i+1:03d}.wav"
                segment_path = temp_dir / segment_filename
                
                sf.write(str(segment_path), segment_audio, sr)
                segment_paths.append(segment_path)
                
                segment_duration = len(segment_audio) / sr
                self.logger.info(f"  段 {i+1}: {segment_duration:.2f}s ({start_sample/sr:.2f}s - {end_sample/sr:.2f}s)")
            
            # 验证分割是否完整覆盖
            total_segmented_duration = sum(len(librosa.load(str(path), sr=None)[0]) / sr for path in segment_paths)
            coverage_ratio = total_segmented_duration / total_duration
            self.logger.info(f"音频分割完成，共生成 {len(segment_paths)} 个文件")
            self.logger.info(f"总时长覆盖比例: {coverage_ratio:.2%} (原始: {total_duration:.2f}s, 分割后: {total_segmented_duration:.2f}s)")
            
            if coverage_ratio < 0.95:
                self.logger.warning(f"音频分割覆盖比例较低 ({coverage_ratio:.2%})，可能存在内容丢失")
            
            return segment_paths
            
        except Exception as e:
            self.logger.error(f"音频分割失败: {e}")
            return [audio_path]
    
    def _merge_segment_results(self, segment_results: List[Dict[str, Any]], 
                              original_audio_path: Path) -> Dict[str, Any]:
        """
        合并分割段的处理结果
        
        Args:
            segment_results: 各分割段的处理结果列表
            original_audio_path: 原始音频文件路径
            
        Returns:
            合并后的结果
        """
        if not segment_results:
            return {
                'source_file': str(original_audio_path),
                'file_name': original_audio_path.name,
                'transcription': '',
                'processing_time': 0,
                'timestamp': datetime.now().isoformat(),
                'status': 'error',
                'error': '没有有效的分割段结果',
                'metadata': {}
            }
        
        # 合并文本
        merged_text = ""
        segment_processing_times = []  # 收集各段处理时间
        total_word_count = 0
        total_character_count = 0
        successful_segments = 0
        
        for i, result in enumerate(segment_results):
            if result.get('status') == 'success':
                segment_text = result.get('transcription', '').strip()
                if segment_text:
                    # 添加段标识，确保顺序正确
                    if self.config.get('include_segment_markers', False):
                        merged_text += f"[段{i+1}] {segment_text}\n"
                    else:
                        # 即使不显示段标识，也要确保按顺序合并
                        merged_text += f"{segment_text}\n"
                    
                    segment_processing_times.append(result.get('processing_time', 0))
                    total_word_count += result.get('metadata', {}).get('word_count', 0)
                    total_character_count += result.get('metadata', {}).get('character_count', 0)
                    successful_segments += 1
                    self.logger.info(f"成功合并分段 {i+1}: {len(segment_text)} 字符 (处理时间: {result.get('processing_time', 0):.2f}s)")
            else:
                self.logger.warning(f"分割段 {i+1} 处理失败: {result.get('error', 'Unknown error')}")
                # 为失败的分段添加占位符，保持顺序
                if self.config.get('include_segment_markers', False):
                    merged_text += f"[段{i+1} - 处理失败]\n"
        
        # 计算总处理时间：并行处理时取最大值，串行处理时取总和
        if len(segment_processing_times) > 1 and self.config.get('parallel_processing', True):
            # 并行处理：取各段处理时间的最大值
            total_processing_time = max(segment_processing_times) if segment_processing_times else 0
            self.logger.info(f"并行处理模式：总处理时间取最大值 {total_processing_time:.2f}s (各段时间: {[f'{t:.2f}s' for t in segment_processing_times]})")
        else:
            # 串行处理：取各段处理时间的总和
            total_processing_time = sum(segment_processing_times) if segment_processing_times else 0
            self.logger.info(f"串行处理模式：总处理时间取总和 {total_processing_time:.2f}s (各段时间: {[f'{t:.2f}s' for t in segment_processing_times]})")
        
        # 确保合并后的文本完整性
        merged_text = ensure_text_completeness(merged_text.strip())
        
        # 重新计算字数统计，确保准确性
        final_word_count = count_chinese_words(merged_text)
        final_character_count = count_characters(merged_text)
        
        # 如果重新计算的统计与累加值差异较大，使用重新计算的值
        if abs(final_word_count - total_word_count) > 10 or abs(final_character_count - total_character_count) > 50:
            self.logger.info(f"字数统计差异较大，使用重新计算的值: 字数 {total_word_count} -> {final_word_count}, 字符数 {total_character_count} -> {final_character_count}")
            total_word_count = final_word_count
            total_character_count = final_character_count
        
        # 清理临时文件
        for result in segment_results:
            if result.get('status') == 'success':
                segment_path = Path(result.get('source_file', ''))
                if segment_path.exists() and 'segment_' in segment_path.name:
                    try:
                        segment_path.unlink()
                        # 尝试删除临时目录
                        temp_dir = segment_path.parent
                        if temp_dir.exists() and temp_dir.name.startswith('audio_segments_'):
                            shutil.rmtree(temp_dir)
                    except Exception as e:
                        self.logger.warning(f"清理临时文件失败: {e}")
        
        # 构建合并结果
        merged_result = {
            'source_file': str(original_audio_path),
            'file_name': original_audio_path.name,
            'file_size': original_audio_path.stat().st_size,
            'transcription': merged_text,
            'processing_time': total_processing_time,
            'timestamp': datetime.now().isoformat(),
            'status': 'success' if successful_segments > 0 else 'error',
            'metadata': {
                'model_used': self.config['model_dir'],
                'language': self.config.get('language', 'auto'),
                'device': self.config.get('device', 'cuda:0'),
                'audio_duration': self._estimate_duration(original_audio_path),
                'word_count': total_word_count,
                'character_count': total_character_count,
                'segmentation_info': {
                    'total_segments': len(segment_results),
                    'successful_segments': successful_segments,
                    'failed_segments': len(segment_results) - successful_segments,
                    'segmentation_method': 'smart_segmentation'
                }
            }
        }
        
        if successful_segments == 0:
            merged_result['error'] = '所有分割段处理失败'
        
        self.logger.info(f"分割段结果合并完成: {successful_segments}/{len(segment_results)} 段成功")
        return merged_result
    
    def _check_ffmpeg_available(self) -> bool:
        """检查ffmpeg是否可用"""
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def get_workflow_status(self) -> Dict[str, Any]:
        """获取工作流状态信息"""
        return {
            'processor_type': 'speech2text',
            'model_loaded': self.model is not None,
            'model_info': {
                'model_dir': self.config['model_dir'],
                'device': self.config.get('device', 'cuda:0'),
                'language': self.config.get('language', 'auto')
            },
            'supported_formats': {
                'audio': ['.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg', '.wma'],
                'video': ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v',
                         '.3gp', '.3g2', '.asf', '.rm', '.rmvb', '.vob', '.ts', '.mts',
                         '.m2ts', '.f4v', '.divx', '.xvid', '.ogv']
            },
            'parallel_processing': {
                'enabled': self.config.get('parallel_processing', True),
                'max_workers': self.config.get('max_workers'),
                'cpu_count': mp.cpu_count()
            },
            'smart_segmentation': {
                'enabled': self.config.get('enable_smart_segmentation', True),
                'max_duration_for_single_process': self.config.get('max_audio_duration_for_single_process', 600.0),
                'processing_time_ratio': self.config.get('processing_time_ratio', 1/17),
                'min_segment_duration': self.config.get('min_segment_duration', 30.0),
                'max_segment_duration': self.config.get('max_segment_duration', 300.0),
                'overlap_duration': self.config.get('overlap_duration', 5.0),
                'include_segment_markers': self.config.get('include_segment_markers', False),
                'audio_libs_available': AUDIO_LIBS_AVAILABLE
            },
            'dashscope_correction': {
                'enabled': self.config.get('enable_dashscope_correction', False),
                'api_available': DASHSCOPE_AVAILABLE,
                'api_key_configured': bool(self.config.get('dashscope_api_key')),
                'model': self.config.get('dashscope_model', 'qwen-turbo'),
                'temperature': self.config.get('dashscope_temperature', 0.1),
                'max_tokens': self.config.get('dashscope_max_tokens', 10000),
                'timeout': self.config.get('dashscope_timeout', 30)
            },
            'video_extraction': {
                'ffmpeg_available': self._check_ffmpeg_available(),
                'video_audio_quality': self.config.get('video_audio_quality', '4'),
                'cleanup_extracted_audio': self.config.get('cleanup_extracted_audio', False)
            },
            'timestamp': datetime.now().isoformat()
        }


# 便捷函数，用于工作流集成
def process_audio_for_workflow(audio_input: Union[str, Path], 
                             output_dir: Optional[Union[str, Path]] = None,
                             config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    工作流集成函数：处理音频文件并返回结果
    
    Args:
        audio_input: 音频文件路径或目录
        output_dir: 输出目录
        config: 配置参数
        
    Returns:
        处理结果字典
    """
    processor = WorkflowSpeechProcessor(config)
    
    audio_path = Path(audio_input)
    
    if audio_path.is_file():
        # 处理单个文件
        result = processor.process_single_audio(audio_path)
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 创建以音频文件名命名的子目录
            audio_name = Path(result['file_name']).stem  # 去掉扩展名
            audio_dir = output_path / audio_name
            audio_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存该音频的详细结果
            json_file = audio_dir / f'{audio_name}_result.json'
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            # 保存该音频的transcripts.json格式
            transcripts_file = audio_dir / 'transcripts.json'
            transcripts_data = processor._format_single_transcript_json(result)
            with open(transcripts_file, 'w', encoding='utf-8') as f:
                json.dump(transcripts_data, f, ensure_ascii=False, indent=2)
            
            # 保存该音频的纯文本格式
            txt_file = audio_dir / f'{audio_name}_content.txt'
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write(f"音频文件: {result['file_name']}\n")
                f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"转写内容:\n{result.get('transcription', '')}\n\n")
                f.write(f"字数: {result.get('metadata', {}).get('word_count', 0)}\n")
                f.write(f"字符数: {result.get('metadata', {}).get('character_count', 0)}\n")
                f.write(f"处理时间: {result.get('processing_time', 0):.2f}秒\n")
                f.write(f"音频时长: {result.get('metadata', {}).get('audio_duration', 0):.2f}秒\n")
                f.write(f"分段信息: {result.get('metadata', {}).get('segmentation_info', {})}\n")
        
        return result
    
    elif audio_path.is_dir():
        # 处理目录
        return processor.process_audio_directory(audio_path, output_dir)
    
    else:
        raise FileNotFoundError(f"输入路径不存在: {audio_input}")


if __name__ == '__main__':
    # 测试示例
    import argparse
    
    parser = argparse.ArgumentParser(description='工作流语音转文字处理器 (支持音频和视频文件)')
    parser.add_argument('input', help='输入音频/视频文件或目录')
    parser.add_argument('-o', '--output', help='输出目录')
    # 检测系统类型，在macOS上默认使用CPU
    import platform
    default_device = 'cpu' if platform.system() == 'Darwin' else 'cuda:0'
    
    parser.add_argument('--device', default=default_device, help='设备 (cuda:0, cpu)')
    parser.add_argument('--language', default='auto', help='语言')
    parser.add_argument('--parallel', action='store_true', default=True, help='启用并行处理 (默认启用)')
    parser.add_argument('--no-parallel', dest='parallel', action='store_false', help='禁用并行处理和智能分割功能')
    parser.add_argument('--max-workers', type=int, default=None, help='最大工作进程数 (默认使用CPU核心数)')
    parser.add_argument('--sequential', action='store_true', help='强制使用串行处理模式')
    parser.add_argument('--show-config', action='store_true', help='显示智能分割配置信息')
    parser.add_argument('--disable-smart-seg', action='store_true', help='禁用智能分割功能')
    parser.add_argument('--enable-correction', action='store_true', help='启用通义千问内容纠错')
    parser.add_argument('--dashscope-api-key', help='通义千问API密钥')
    parser.add_argument('--dashscope-model', default='qwen-turbo', help='通义千问模型名称')
    parser.add_argument('--dashscope-temperature', type=float, default=0.1, help='通义千问温度参数')
    parser.add_argument('--dashscope-max-tokens', type=int, default=10000, help='通义千问最大token数')
    parser.add_argument('--dashscope-timeout', type=int, default=30, help='通义千问API超时时间（秒）')
    parser.add_argument('--video-audio-quality', default='4', help='视频音频提取质量 (0-9, 0最高质量，9最低质量)')
    parser.add_argument('--cleanup-extracted-audio', action='store_true', help='处理完成后清理临时提取的音频文件')
    
    args = parser.parse_args()
    
    config = {
        'device': args.device,
        'language': args.language,
        'parallel_processing': args.parallel and not args.sequential,
        'max_workers': args.max_workers,
        'enable_smart_segmentation': not args.disable_smart_seg and args.parallel and not args.sequential,
        'enable_dashscope_correction': args.enable_correction,
        'dashscope_api_key': args.dashscope_api_key,
        'dashscope_model': args.dashscope_model,
        'dashscope_temperature': args.dashscope_temperature,
        'dashscope_max_tokens': args.dashscope_max_tokens,
        'dashscope_timeout': args.dashscope_timeout,
        'video_audio_quality': args.video_audio_quality,
        'cleanup_extracted_audio': args.cleanup_extracted_audio
    }
    
    # 如果只是显示配置，不执行处理
    if args.show_config:
        processor = WorkflowSpeechProcessor(config)
        status = processor.get_workflow_status()
        
        print("=== 智能分割配置信息 ===")
        print(f"智能分割: {'启用' if status['smart_segmentation']['enabled'] else '禁用'}")
        print(f"激活条件: 最佳分割段数 > 2")
        print(f"处理时间比值: 1:{int(1/status['smart_segmentation']['processing_time_ratio'])}")
        print(f"最小段时长: {status['smart_segmentation']['min_segment_duration']}秒")
        print(f"最大段时长: {status['smart_segmentation']['max_segment_duration']}秒")
        print(f"重叠时长: {status['smart_segmentation']['overlap_duration']}秒")
        print(f"音频库可用: {'是' if status['smart_segmentation']['audio_libs_available'] else '否'}")
        print()
        print("=== 并行处理配置 ===")
        print(f"并行处理: {'启用' if status['parallel_processing']['enabled'] else '禁用'}")
        print(f"最大工作进程数: {status['parallel_processing']['max_workers'] or '自动 (CPU核心数)'}")
        print(f"CPU核心数: {status['parallel_processing']['cpu_count']}")
        print()
        print("=== 通义千问内容纠错配置 ===")
        print(f"内容纠错: {'启用' if status['dashscope_correction']['enabled'] else '禁用'}")
        print(f"通义千问库可用: {'是' if status['dashscope_correction']['api_available'] else '否'}")
        print(f"API密钥配置: {'是' if status['dashscope_correction']['api_key_configured'] else '否'}")
        print(f"使用模型: {status['dashscope_correction']['model']}")
        print(f"温度参数: {status['dashscope_correction']['temperature']}")
        print(f"最大token数: {status['dashscope_correction']['max_tokens']}")
        print(f"超时时间: {status['dashscope_correction']['timeout']}秒")
        print()
        print("=== 视频音频提取配置 ===")
        print(f"ffmpeg可用: {'是' if status['video_extraction']['ffmpeg_available'] else '否'}")
        print(f"音频质量: {status['video_extraction']['video_audio_quality']} (0-9, 0最高)")
        print(f"清理临时文件: {'是' if status['video_extraction']['cleanup_extracted_audio'] else '否'}")
        print(f"支持的音频格式: {', '.join(status['supported_formats']['audio'])}")
        print(f"支持的视频格式: {', '.join(status['supported_formats']['video'])}")
        sys.exit(0)
    
    print(f"配置信息:")
    print(f"  输入: {args.input}")
    print(f"  输出: {args.output}")
    print(f"  设备: {args.device}")
    print(f"  语言: {args.language}")
    print(f"  并行处理: {config['parallel_processing']}")
    print(f"  智能分割: {config['enable_smart_segmentation']}")
    if config['enable_smart_segmentation']:
        print(f"  智能分割激活条件: 最佳分割段数 > 2")
    else:
        if not config['parallel_processing']:
            print(f"  智能分割已禁用: 并行处理被禁用")
        else:
            print(f"  智能分割已禁用: 用户手动禁用")
    print(f"  通义千问内容纠错: {config['enable_dashscope_correction']}")
    if config['enable_dashscope_correction']:
        print(f"  通义千问模型: {config['dashscope_model']}")
        print(f"  通义千问温度: {config['dashscope_temperature']}")
        print(f"  通义千问最大token: {config['dashscope_max_tokens']}")
        print(f"  通义千问超时: {config['dashscope_timeout']}秒")
    print(f"  视频音频提取质量: {config['video_audio_quality']} (0-9)")
    print(f"  清理临时音频文件: {config['cleanup_extracted_audio']}")
    if config['max_workers']:
        print(f"  最大工作进程数: {config['max_workers']}")
    else:
        print(f"  最大工作进程数: 自动 (CPU核心数)")
    print()
    
    result = process_audio_for_workflow(args.input, args.output, config)
    print(f"处理完成，状态: {result.get('status', 'unknown')}")
    
    if result.get('status') == 'completed':
        print(f"处理模式: {result.get('processing_mode', 'unknown')}")
        print(f"总文件数: {result.get('total_files', 0)}")
        print(f"成功: {result.get('successful', 0)}")
        print(f"失败: {result.get('failed', 0)}")
        print(f"总处理时间: {result.get('total_processing_time', 0):.2f}秒")
        print(f"平均处理时间: {result.get('average_processing_time', 0):.2f}秒")
        if result.get('processing_mode') == 'parallel':
            print(f"并行工作进程数: {result.get('max_workers', 0)}")
            if result.get('estimated_parallel_efficiency', 0) > 0:
                print(f"预估并行效率: {result.get('estimated_parallel_efficiency', 0):.2f}x")
                print(f"注意：这是基于估算的串行时间，实际效果需要对比测试验证") 


# 使用示例:
# 处理音频文件:
# python mp3_to_text.py /Users/eureka/VSCodeProjects/Dianxin/datasets/mp3/test3.mp3 --enable-correction -o /Users/eureka/VSCodeProjects/Dianxin/output/test3
# python mp3_to_text.py /Users/eureka/VSCodeProjects/Dianxin/datasets/dingling.mp3 --enable-correction -o /Users/eureka/VSCodeProjects/Dianxin/output/dingling

# 处理视频文件:
# python mp3_to_text.py /path/to/video.mp4 --enable-correction -o /path/to/output
# python video_to_text.py /Users/eureka/VSCodeProjects/Dianxin/datasets/video --parallel --enable-correction -o /Users/eureka/VSCodeProjects/Dianxin/output

# 视频相关选项:
# --video-audio-quality 4      # 设置音频提取质量 (0-9, 0最高质量)
# --cleanup-extracted-audio    # 处理完成后清理临时音频文件