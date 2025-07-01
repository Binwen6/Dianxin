#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试新的目录结构功能
"""

import os
import json
from pathlib import Path
from workflow_processor import WorkflowSpeechProcessor

def test_directory_structure():
    """测试新的目录结构"""
    
    # 创建测试配置
    config = {
        'device': 'cpu',  # 使用CPU进行测试
        'language': 'auto'
    }
    
    # 创建处理器
    processor = WorkflowSpeechProcessor(config)
    
    # 模拟处理结果（不实际加载模型）
    test_results = [
        {
            'source_file': '/path/to/test1.mp3',
            'file_name': 'test1.mp3',
            'file_size': 1024000,
            'transcription': '这是第一个测试音频的转写内容。',
            'processing_time': 5.2,
            'timestamp': '2024-01-01T10:00:00',
            'status': 'success',
            'metadata': {
                'model_used': 'iic/SenseVoiceSmall',
                'language': 'auto',
                'device': 'cpu',
                'audio_duration': 30.5,
                'word_count': 12,
                'character_count': 15
            }
        },
        {
            'source_file': '/path/to/test2.wav',
            'file_name': 'test2.wav',
            'file_size': 2048000,
            'transcription': '这是第二个测试音频的转写内容，包含更多的文字。',
            'processing_time': 8.7,
            'timestamp': '2024-01-01T10:05:00',
            'status': 'success',
            'metadata': {
                'model_used': 'iic/SenseVoiceSmall',
                'language': 'auto',
                'device': 'cpu',
                'audio_duration': 45.2,
                'word_count': 18,
                'character_count': 25
            }
        }
    ]
    
    # 创建批量处理结果
    batch_result = {
        'status': 'completed',
        'source_directory': '/path/to/audio',
        'output_directory': 'test_output',
        'processed_files': test_results,
        'total_files': 2,
        'successful': 2,
        'failed': 0,
        'total_processing_time': 13.9,
        'average_processing_time': 6.95,
        'timestamp': '2024-01-01T10:10:00',
        'summary': {
            'total_words': 30,
            'total_characters': 40,
            'total_audio_duration': 75.7
        }
    }
    
    # 测试保存功能
    output_dir = Path('test_output')
    processor._save_batch_results(batch_result, output_dir)
    
    print("测试完成！请检查 test_output 目录结构：")
    print(f"输出目录: {output_dir.absolute()}")
    
    # 显示目录结构
    def show_directory_structure(path, prefix=""):
        if not path.exists():
            return
        
        items = sorted(path.iterdir())
        for i, item in enumerate(items):
            is_last = i == len(items) - 1
            current_prefix = "└── " if is_last else "├── "
            print(f"{prefix}{current_prefix}{item.name}")
            
            if item.is_dir():
                next_prefix = prefix + ("    " if is_last else "│   ")
                show_directory_structure(item, next_prefix)
    
    show_directory_structure(output_dir)

if __name__ == '__main__':
    test_directory_structure() 