#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SenseVoice 工作流语音转文字处理器
专门设计用于视频转音频 -> 语音转文字 -> 文字转PPT 工作流的中间环节
"""

import os
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

# 尝试导入通义千问API库
try:
    import dashscope
    from dashscope import Generation
    DASHSCOPE_AVAILABLE = True
except ImportError:
    DASHSCOPE_AVAILABLE = False
    print("警告: dashscope 库未安装，内容纠错功能将不可用。请安装: pip install dashscope")


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
        return text
    
    try:
        # 设置通义千问API密钥
        dashscope.api_key = api_key
        
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
            max_tokens=config.get('dashscope_max_tokens', 4000),
            timeout=config.get('dashscope_timeout', 30)
        )
        
        if response.status_code == 200:
            corrected_text = response.output.choices[0].message.content.strip()
            
            # 如果返回的文本为空或异常，返回原文本
            if not corrected_text:
                return text
            
            return corrected_text
        else:
            print(f"通义千问API调用失败: {response.message}")
            return text
        
    except Exception as e:
        # 如果API调用失败，返回原文本
        print(f"通义千问API纠错失败: {e}")
        return text


def _process_single_audio_worker(audio_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    独立的工作进程函数，用于处理单个音频文件
    
    Args:
        audio_path: 音频文件路径
        config: 配置字典
        
    Returns:
        处理结果字典
    """
    audio_path = Path(audio_path)
    
    if not audio_path.exists():
        return {
            'source_file': str(audio_path),
            'file_name': audio_path.name,
            'transcription': '',
            'processing_time': 0,
            'timestamp': datetime.now().isoformat(),
            'status': 'error',
            'error': f"音频文件不存在: {audio_path}",
            'metadata': {}
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
            'transcription': text,  # 这里已经是纠错后的文本
            'processing_time': processing_time,
            'timestamp': datetime.now().isoformat(),
            'status': 'success',
            'metadata': {
                'model_used': config['model_dir'],
                'language': config.get('language', 'auto'),
                'device': config.get('device', 'cuda:0'),
                'audio_duration': estimate_duration(audio_path),
                'word_count': len(text.split()) if text else 0,
                'character_count': len(text) if text else 0,
                'dashscope_correction': correction_info
            }
        }
        
        return result
        
    except Exception as e:
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
            'max_workers': None,  # 最大工作进程数，None表示使用CPU核心数
            'chunk_size': 1,  # 每个进程处理的文件数量
            # 智能分割配置
            'enable_smart_segmentation': True,  # 是否启用智能分割
            'max_audio_duration_for_single_process': 600.0,  # 超过此时长（秒）的音频将进行智能分割
            'processing_time_ratio': 1/17,  # 音频处理时间与原始时间的比值（研究结果）
            'min_segment_duration': 30.0,  # 最小分割段时长（秒）
            'max_segment_duration': 300.0,  # 最大分割段时长（秒）
            'overlap_duration': 5.0,  # 分割段之间的重叠时长（秒）
            # 通义千问内容纠错配置
            'enable_dashscope_correction': False,  # 是否启用通义千问内容纠错
            'dashscope_api_key': None,  # 通义千问API密钥
            'dashscope_model': 'qwen-turbo',  # 使用的模型
            'dashscope_temperature': 0.1,  # 温度参数
            'dashscope_max_tokens': 4000,  # 最大token数
            'dashscope_timeout': 30,  # API超时时间（秒）
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
        处理单个音频文件（支持智能分割）
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            处理结果字典，包含转写文本和元数据
        """
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")
        
        # 检查是否需要智能分割
        if self._should_segment_audio(audio_path):
            return self._process_audio_with_segmentation(audio_path)
        else:
            return self._process_audio_directly(audio_path)
    
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
                    'word_count': len(text.split()) if text else 0,
                    'character_count': len(text) if text else 0,
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
            self.logger.info(f"开始智能分割处理长音频: {audio_path.name}")
            
            # 获取音频时长
            audio_duration = self._estimate_duration(audio_path)
            
            # 计算最佳分割段数
            processing_time_ratio = self.config.get('processing_time_ratio', 1/17)
            num_segments = self._calculate_optimal_segments(audio_duration, processing_time_ratio)
            
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
                
                # 使用现有的并行处理逻辑
                segment_results = self._process_audio_directory_parallel(segment_paths, Path(tempfile.gettempdir()))
                
                # 提取处理结果
                if segment_results.get('status') == 'completed':
                    segment_results = segment_results.get('processed_files', [])
                else:
                    # 如果并行处理失败，回退到串行
                    self.logger.warning("并行处理分割段失败，回退到串行处理")
                    segment_results = []
                    for i, segment_path in enumerate(segment_paths):
                        self.logger.info(f"处理分割段 {i+1}/{len(segment_paths)}: {segment_path.name}")
                        segment_result = self._process_audio_directly(segment_path)
                        segment_results.append(segment_result)
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
            
            self.logger.info(f"智能分割处理完成: {audio_path.name}")
            return merged_result
            
        except Exception as e:
            self.logger.error(f"智能分割处理失败 {audio_path}: {e}")
            # 回退到直接处理
            self.logger.info("回退到直接处理模式")
            return self._process_audio_directly(audio_path)
    
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
        
        # 获取所有音频文件
        audio_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg'}
        audio_files = []
        
        for ext in audio_extensions:
            audio_files.extend(audio_dir.rglob(f'*{ext}'))
            audio_files.extend(audio_dir.rglob(f'*{ext.upper()}'))
        
        audio_files = sorted(audio_files)
        
        if not audio_files:
            self.logger.warning(f"在目录 {audio_dir} 中未找到音频文件")
            return {
                'status': 'no_files_found',
                'processed_files': [],
                'total_files': 0,
                'successful': 0,
                'failed': 0
            }
        
        self.logger.info(f"找到 {len(audio_files)} 个音频文件")
        
        # 检查是否启用并行处理
        # 对于小文件数量，串行处理可能更快（避免模型加载开销）
        if (self.config.get('parallel_processing', True) and 
            len(audio_files) > 1 and 
            len(audio_files) >= 4):  # 只有文件数量>=4时才使用并行
            return self._process_audio_directory_parallel(audio_files, output_dir)
        else:
            if len(audio_files) > 1:
                self.logger.info(f"文件数量较少({len(audio_files)})，使用串行处理以避免模型加载开销")
            return self._process_audio_directory_sequential(audio_files, output_dir)
    
    def _process_audio_directory_parallel(self, audio_files: List[Path], 
                                        output_dir: Path) -> Dict[str, Any]:
        """
        并行处理音频目录中的所有音频文件
        
        Args:
            audio_files: 音频文件列表
            output_dir: 输出目录
            
        Returns:
            批量处理结果字典
        """
        self.logger.info("使用并行处理模式")
        
        # 确定工作进程数 - 根据设备类型优化并发数
        max_workers = self.config.get('max_workers')
        if max_workers is None:
            # 如果使用GPU，限制并发数以避免GPU内存不足
            if self.config.get('device', 'cuda:0').startswith('cuda'):
                max_workers = min(2, len(audio_files))  # GPU模式下最多2个并发
            else:
                # CPU模式下可以使用更多并发，但也要考虑内存使用
                max_workers = min(mp.cpu_count(), len(audio_files), 4)  # CPU模式下最多4个并发
        
        self.logger.info(f"使用 {max_workers} 个工作进程进行并行处理")
        self.logger.info(f"设备: {self.config.get('device', 'cuda:0')}")
        
        # 准备参数
        audio_paths = [str(f) for f in audio_files]
        
        # 使用进程池进行并行处理
        start_time = time.time()
        results = []
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_audio = {
                executor.submit(_process_single_audio_worker, audio_path, self.config): audio_path 
                for audio_path in audio_paths
            }
            
            # 收集结果
            completed_count = 0
            individual_times = []
            
            for future in as_completed(future_to_audio):
                audio_path = future_to_audio[future]
                try:
                    result = future.result()
                    results.append(result)
                    completed_count += 1
                    
                    if result['status'] == 'success':
                        individual_times.append(result['processing_time'])
                        self.logger.info(f"完成 [{completed_count}/{len(audio_files)}]: {Path(audio_path).name} (耗时: {result['processing_time']:.2f}s)")
                    else:
                        self.logger.error(f"失败 [{completed_count}/{len(audio_files)}]: {Path(audio_path).name} - {result.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    self.logger.error(f"处理失败 {audio_path}: {e}")
                    results.append({
                        'source_file': audio_path,
                        'file_name': Path(audio_path).name,
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
            'source_directory': str(audio_files[0].parent),
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
    
    def _process_audio_directory_sequential(self, audio_files: List[Path], 
                                          output_dir: Path) -> Dict[str, Any]:
        """
        串行处理音频目录中的所有音频文件
        
        Args:
            audio_files: 音频文件列表
            output_dir: 输出目录
            
        Returns:
            批量处理结果字典
        """
        self.logger.info("使用串行处理模式")
        
        # 批量处理
        results = []
        total_processing_time = 0
        
        for i, audio_file in enumerate(audio_files, 1):
            self.logger.info(f"处理进度 [{i}/{len(audio_files)}]: {audio_file.name}")
            result = self.process_single_audio(audio_file)
            results.append(result)
            total_processing_time += result.get('processing_time', 0)
        
        # 统计信息
        successful = [r for r in results if r['status'] == 'success']
        failed = [r for r in results if r['status'] == 'error']
        
        batch_result = {
            'status': 'completed',
            'source_directory': str(audio_files[0].parent),
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
        根据研究结果计算最佳分割段数
        
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
        
        # 确保至少为1，最多不超过音频时长除以最小段时长
        min_segments = 1
        max_segments = max(1, int(audio_duration / self.config.get('min_segment_duration', 30.0)))
        
        optimal_segments = max(min_segments, min(optimal_segments, max_segments))
        
        self.logger.info(f"音频时长: {audio_duration:.2f}s, 计算得到最佳分割段数: {optimal_segments}")
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
            return False
        
        # 获取音频时长
        audio_duration = self._estimate_duration(audio_path)
        
        # 使用研究公式计算最佳分割段数
        processing_time_ratio = self.config.get('processing_time_ratio', 1/17)
        optimal_segments = self._calculate_optimal_segments(audio_duration, processing_time_ratio)
        
        # 只有当最佳分割段数 > 2 时才真正启用智能分割
        should_segment = optimal_segments > 2
        
        self.logger.info(f"音频 {audio_path.name} 时长: {audio_duration:.2f}s, 计算得到最佳段数: {optimal_segments}, 是否启用智能分割: {should_segment}")
        
        return should_segment
    
    def _segment_audio(self, audio_path: Path, num_segments: int) -> List[Path]:
        """
        将音频文件分割为指定数量的段
        
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
            duration = len(audio) / sr
            
            # 计算每段时长（包含重叠）
            overlap_duration = self.config.get('overlap_duration', 5.0)
            overlap_samples = int(overlap_duration * sr)
            
            # 计算每段的基础时长
            base_segment_duration = duration / num_segments
            base_segment_samples = int(base_segment_duration * sr)
            
            # 确保每段时长在合理范围内
            min_segment_samples = int(self.config.get('min_segment_duration', 30.0) * sr)
            max_segment_samples = int(self.config.get('max_segment_duration', 300.0) * sr)
            
            base_segment_samples = max(min_segment_samples, 
                                     min(base_segment_samples, max_segment_samples))
            
            # 创建临时目录
            temp_dir = Path(tempfile.mkdtemp(prefix=f"audio_segments_{audio_path.stem}_"))
            segment_paths = []
            
            self.logger.info(f"开始分割音频 {audio_path.name} 为 {num_segments} 段")
            
            for i in range(num_segments):
                # 计算当前段的起始和结束位置
                start_sample = i * (base_segment_samples - overlap_samples)
                end_sample = min(start_sample + base_segment_samples, len(audio))
                
                # 确保最后一段不会太短
                if i == num_segments - 1 and end_sample - start_sample < min_segment_samples:
                    start_sample = max(0, end_sample - min_segment_samples)
                
                # 提取音频段
                segment_audio = audio[start_sample:end_sample]
                
                # 保存音频段
                segment_filename = f"{audio_path.stem}_segment_{i+1:03d}.wav"
                segment_path = temp_dir / segment_filename
                
                sf.write(str(segment_path), segment_audio, sr)
                segment_paths.append(segment_path)
                
                segment_duration = len(segment_audio) / sr
                self.logger.info(f"  段 {i+1}: {segment_duration:.2f}s ({start_sample/sr:.2f}s - {end_sample/sr:.2f}s)")
            
            self.logger.info(f"音频分割完成，共生成 {len(segment_paths)} 个文件")
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
        total_processing_time = 0
        total_word_count = 0
        total_character_count = 0
        successful_segments = 0
        
        for i, result in enumerate(segment_results):
            if result.get('status') == 'success':
                segment_text = result.get('transcription', '').strip()
                if segment_text:
                    # 添加段标识（可选）
                    if self.config.get('include_segment_markers', False):
                        merged_text += f"[段{i+1}] {segment_text}\n"
                    else:
                        merged_text += f"{segment_text}\n"
                    
                    total_processing_time += result.get('processing_time', 0)
                    total_word_count += result.get('metadata', {}).get('word_count', 0)
                    total_character_count += result.get('metadata', {}).get('character_count', 0)
                    successful_segments += 1
            else:
                self.logger.warning(f"分割段 {i+1} 处理失败: {result.get('error', 'Unknown error')}")
        
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
            'transcription': merged_text.strip(),
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
            'supported_formats': ['.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg'],
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
                'audio_libs_available': AUDIO_LIBS_AVAILABLE
            },
            'dashscope_correction': {
                'enabled': self.config.get('enable_dashscope_correction', False),
                'api_available': DASHSCOPE_AVAILABLE,
                'api_key_configured': bool(self.config.get('dashscope_api_key')),
                'model': self.config.get('dashscope_model', 'qwen-turbo'),
                'temperature': self.config.get('dashscope_temperature', 0.1),
                'max_tokens': self.config.get('dashscope_max_tokens', 4000),
                'timeout': self.config.get('dashscope_timeout', 30)
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
        
        return result
    
    elif audio_path.is_dir():
        # 处理目录
        return processor.process_audio_directory(audio_path, output_dir)
    
    else:
        raise FileNotFoundError(f"输入路径不存在: {audio_input}")


if __name__ == '__main__':
    # 测试示例
    import argparse
    
    parser = argparse.ArgumentParser(description='工作流语音转文字处理器')
    parser.add_argument('input', help='输入音频文件或目录')
    parser.add_argument('-o', '--output', help='输出目录')
    # 检测系统类型，在macOS上默认使用CPU
    import platform
    default_device = 'cpu' if platform.system() == 'Darwin' else 'cuda:0'
    
    parser.add_argument('--device', default=default_device, help='设备 (cuda:0, cpu)')
    parser.add_argument('--language', default='auto', help='语言')
    parser.add_argument('--parallel', action='store_true', default=True, help='启用并行处理 (默认启用)')
    parser.add_argument('--no-parallel', dest='parallel', action='store_false', help='禁用并行处理')
    parser.add_argument('--max-workers', type=int, default=None, help='最大工作进程数 (默认使用CPU核心数)')
    parser.add_argument('--sequential', action='store_true', help='强制使用串行处理模式')
    parser.add_argument('--show-config', action='store_true', help='显示智能分割配置信息')
    parser.add_argument('--disable-smart-seg', action='store_true', help='禁用智能分割功能')
    parser.add_argument('--enable-correction', action='store_true', help='启用通义千问内容纠错')
    parser.add_argument('--dashscope-api-key', help='通义千问API密钥')
    parser.add_argument('--dashscope-model', default='qwen-turbo', help='通义千问模型名称')
    parser.add_argument('--dashscope-temperature', type=float, default=0.1, help='通义千问温度参数')
    parser.add_argument('--dashscope-max-tokens', type=int, default=4000, help='通义千问最大token数')
    parser.add_argument('--dashscope-timeout', type=int, default=30, help='通义千问API超时时间（秒）')
    
    args = parser.parse_args()
    
    config = {
        'device': args.device,
        'language': args.language,
        'parallel_processing': args.parallel and not args.sequential,
        'max_workers': args.max_workers,
        'enable_smart_segmentation': not args.disable_smart_seg,
        'enable_dashscope_correction': args.enable_correction,
        'dashscope_api_key': args.dashscope_api_key,
        'dashscope_model': args.dashscope_model,
        'dashscope_temperature': args.dashscope_temperature,
        'dashscope_max_tokens': args.dashscope_max_tokens,
        'dashscope_timeout': args.dashscope_timeout
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
        sys.exit(0)
    
    print(f"配置信息:")
    print(f"  输入: {args.input}")
    print(f"  输出: {args.output}")
    print(f"  设备: {args.device}")
    print(f"  语言: {args.language}")
    print(f"  并行处理: {config['parallel_processing']}")
    print(f"  智能分割: {config['enable_smart_segmentation']}")
    print(f"  智能分割激活条件: 最佳分割段数 > 2")
    print(f"  通义千问内容纠错: {config['enable_dashscope_correction']}")
    if config['enable_dashscope_correction']:
        print(f"  通义千问模型: {config['dashscope_model']}")
        print(f"  通义千问温度: {config['dashscope_temperature']}")
        print(f"  通义千问最大token: {config['dashscope_max_tokens']}")
        print(f"  通义千问超时: {config['dashscope_timeout']}秒")
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