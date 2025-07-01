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

from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess


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
        return {
            'model_dir': 'iic/SenseVoiceSmall',
            'remote_code': './model.py',
            'vad_model': 'fsmn-vad',
            'vad_kwargs': {
                'max_single_segment_time': 30000
            },
            'device': 'cuda:0',
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
        处理单个音频文件
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            处理结果字典，包含转写文本和元数据
        """
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")
        
        # 加载模型
        self.load_model()
        
        try:
            self.logger.info(f"正在处理音频文件: {audio_path.name}")
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
            
            processing_time = time.time() - start_time
            
            # 构建结果
            result = {
                'source_file': str(audio_path),
                'file_name': audio_path.name,
                'file_size': audio_path.stat().st_size,
                'transcription': text,
                'processing_time': processing_time,
                'timestamp': datetime.now().isoformat(),
                'status': 'success',
                'metadata': {
                    'model_used': self.config['model_dir'],
                    'language': self.config.get('language', 'auto'),
                    'device': self.config.get('device', 'cuda:0'),
                    'audio_duration': self._estimate_duration(audio_path),
                    'word_count': len(text.split()) if text else 0,
                    'character_count': len(text) if text else 0
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
        
        # 批量处理
        results = []
        total_processing_time = 0
        
        for audio_file in audio_files:
            result = self.process_single_audio(audio_file)
            results.append(result)
            total_processing_time += result.get('processing_time', 0)
        
        # 统计信息
        successful = [r for r in results if r['status'] == 'success']
        failed = [r for r in results if r['status'] == 'error']
        
        batch_result = {
            'status': 'completed',
            'source_directory': str(audio_dir),
            'output_directory': str(output_dir),
            'processed_files': results,
            'total_files': len(results),
            'successful': len(successful),
            'failed': len(failed),
            'total_processing_time': total_processing_time,
            'average_processing_time': total_processing_time / len(results) if results else 0,
            'timestamp': datetime.now().isoformat(),
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
        """估算音频文件时长（简单估算）"""
        try:
            # 这里可以集成更准确的音频时长检测库
            # 目前使用文件大小进行简单估算
            file_size = audio_path.stat().st_size
            # 假设平均比特率约为128kbps
            estimated_duration = file_size / (128 * 1024 / 8)
            return max(0, estimated_duration)
        except:
            return 0.0
    
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
    parser.add_argument('--device', default='cuda:0', help='设备 (cuda:0, cpu)')
    parser.add_argument('--language', default='auto', help='语言')
    
    args = parser.parse_args()
    
    config = {
        'device': args.device,
        'language': args.language
    }
    
    result = process_audio_for_workflow(args.input, args.output, config)
    print(f"处理完成，状态: {result.get('status', 'unknown')}") 