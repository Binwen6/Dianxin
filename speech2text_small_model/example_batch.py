#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SenseVoice 批处理使用示例
"""

from batch_processor import SenseVoiceBatchProcessor, load_config

def main():
    """示例：批量处理音频文件"""
    
    # 配置参数
    config = {
        'model_dir': 'iic/SenseVoiceSmall',
        'device': 'cuda:0',  # 如果没有GPU，改为 'cpu'
        'language': 'auto',
        'log_dir': 'logs'
    }
    
    # 创建处理器
    processor = SenseVoiceBatchProcessor(config)
    
    # 处理单个文件
    print("=== 处理单个文件 ===")
    processor.process_batch(
        input_path='../datasets/mp3/test2.mp3',
        output_dir='../output/single_file'
    )
    
    # 处理整个目录
    print("\n=== 处理整个目录 ===")
    processor.process_batch(
        input_path='../datasets/mp3',
        output_dir='../output/batch_processing'
    )

if __name__ == '__main__':
    main() 