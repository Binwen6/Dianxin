#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SenseVoice 批处理工作流
支持目录级语音转文字处理，包含进度显示、日志记录和结果保存
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from tqdm import tqdm
import time

from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess


class SenseVoiceBatchProcessor:
    """SenseVoice 批处理处理器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化批处理处理器
        
        Args:
            config: 配置字典，包含模型路径、设备等参数
        """
        self.config = config
        self.model = None
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger('SenseVoiceBatchProcessor')
        logger.setLevel(logging.INFO)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 创建文件处理器
        log_dir = Path(self.config.get('log_dir', 'logs'))
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'sensevoice_batch_{timestamp}.log'
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 设置格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        return logger
    
    def load_model(self):
        """加载 SenseVoice 模型"""
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
    
    def get_audio_files(self, input_path: str) -> List[Path]:
        """
        获取指定目录下的所有音频文件
        
        Args:
            input_path: 输入路径（文件或目录）
            
        Returns:
            音频文件路径列表
        """
        input_path = Path(input_path)
        audio_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg'}
        
        if input_path.is_file():
            if input_path.suffix.lower() in audio_extensions:
                return [input_path]
            else:
                self.logger.warning(f"不支持的文件格式: {input_path}")
                return []
        
        elif input_path.is_dir():
            audio_files = []
            for ext in audio_extensions:
                audio_files.extend(input_path.rglob(f'*{ext}'))
                audio_files.extend(input_path.rglob(f'*{ext.upper()}'))
            return sorted(audio_files)
        
        else:
            self.logger.error(f"输入路径不存在: {input_path}")
            return []
    
    def transcribe_audio(self, audio_path: Path) -> Dict[str, Any]:
        """
        转写单个音频文件
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            转写结果字典
        """
        try:
            self.logger.info(f"正在处理: {audio_path}")
            start_time = time.time()
            
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
            
            result = {
                'file_path': str(audio_path),
                'file_name': audio_path.name,
                'text': text,
                'processing_time': processing_time,
                'timestamp': datetime.now().isoformat(),
                'status': 'success'
            }
            
            self.logger.info(f"处理完成: {audio_path.name} (耗时: {processing_time:.2f}s)")
            return result
            
        except Exception as e:
            self.logger.error(f"处理失败 {audio_path}: {e}")
            return {
                'file_path': str(audio_path),
                'file_name': audio_path.name,
                'text': '',
                'processing_time': 0,
                'timestamp': datetime.now().isoformat(),
                'status': 'error',
                'error': str(e)
            }
    
    def save_results(self, results: List[Dict[str, Any]], output_dir: str):
        """
        保存转写结果
        
        Args:
            results: 转写结果列表
            output_dir: 输出目录
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存 JSON 格式的详细结果
        json_file = output_path / f'transcription_results_{timestamp}.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 保存纯文本格式的结果
        txt_file = output_path / f'transcription_results_{timestamp}.txt'
        with open(txt_file, 'w', encoding='utf-8') as f:
            for result in results:
                if result['status'] == 'success':
                    f.write(f"文件: {result['file_name']}\n")
                    f.write(f"转写结果:\n{result['text']}\n")
                    f.write(f"处理时间: {result['processing_time']:.2f}秒\n")
                    f.write("-" * 50 + "\n\n")
                else:
                    f.write(f"文件: {result['file_name']} - 处理失败\n")
                    f.write(f"错误: {result.get('error', '未知错误')}\n")
                    f.write("-" * 50 + "\n\n")
        
        # 保存统计信息
        stats_file = output_path / f'processing_stats_{timestamp}.json'
        successful = [r for r in results if r['status'] == 'success']
        failed = [r for r in results if r['status'] == 'error']
        
        stats = {
            'total_files': len(results),
            'successful': len(successful),
            'failed': len(failed),
            'total_processing_time': sum(r['processing_time'] for r in successful),
            'average_processing_time': sum(r['processing_time'] for r in successful) / len(successful) if successful else 0,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"结果已保存到: {output_path}")
        self.logger.info(f"处理统计: 总计 {len(results)} 个文件, 成功 {len(successful)} 个, 失败 {len(failed)} 个")
    
    def process_batch(self, input_path: str, output_dir: str):
        """
        批量处理音频文件
        
        Args:
            input_path: 输入路径（文件或目录）
            output_dir: 输出目录
        """
        # 加载模型
        self.load_model()
        
        # 获取音频文件列表
        audio_files = self.get_audio_files(input_path)
        
        if not audio_files:
            self.logger.warning("未找到可处理的音频文件")
            return
        
        self.logger.info(f"找到 {len(audio_files)} 个音频文件")
        
        # 批量处理
        results = []
        for audio_file in tqdm(audio_files, desc="处理进度"):
            result = self.transcribe_audio(audio_file)
            results.append(result)
        
        # 保存结果
        self.save_results(results, output_dir)


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    default_config = {
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
        'log_dir': 'logs'
    }
    
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
            default_config.update(user_config)
        except Exception as e:
            print(f"加载配置文件失败: {e}")
    
    return default_config


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='SenseVoice 批处理语音转文字工具')
    parser.add_argument('input', help='输入路径（文件或目录）')
    parser.add_argument('-o', '--output', default='output', help='输出目录 (默认: output)')
    parser.add_argument('-c', '--config', help='配置文件路径')
    parser.add_argument('--model-dir', help='模型目录路径')
    parser.add_argument('--device', help='设备 (cuda:0, cpu)')
    parser.add_argument('--language', help='语言 (auto, zn, en, yue, ja, ko)')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 命令行参数覆盖配置文件
    if args.model_dir:
        config['model_dir'] = args.model_dir
    if args.device:
        config['device'] = args.device
    if args.language:
        config['language'] = args.language
    
    # 创建处理器并执行批处理
    processor = SenseVoiceBatchProcessor(config)
    processor.process_batch(args.input, args.output)


if __name__ == '__main__':
    main() 