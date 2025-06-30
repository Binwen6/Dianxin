#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M4A转换使用示例
演示如何使用M4AToMP3Converter类进行批量转换
"""

import sys
import os
from pathlib import Path

# 添加utils目录到Python路径
sys.path.append(str(Path(__file__).parent / 'utils'))

from m4a_to_mp3 import M4AToMP3Converter


def example_basic_conversion():
    """基本转换示例"""
    print("=== 基本转换示例 ===")
    
    # 转换datasets目录下的所有m4a文件
    converter = M4AToMP3Converter(
        input_dir="datasets/m4a",
        output_dir="datasets/mp3",  # 输出到同一目录
        quality="192k"
    )
    
    results = converter.convert_batch()
    print(f"转换结果: {results}")


def example_high_quality_conversion():
    """高质量转换示例"""
    print("\n=== 高质量转换示例 ===")
    
    # 创建输出目录
    output_dir = "converted_mp3"
    os.makedirs(output_dir, exist_ok=True)
    
    # 使用高质量设置转换
    converter = M4AToMP3Converter(
        input_dir="datasets",
        output_dir=output_dir,
        quality="320k",  # 高质量
        max_workers=2    # 减少并发数
    )
    
    results = converter.convert_batch()
    print(f"高质量转换结果: {results}")


def example_custom_conversion():
    """自定义转换示例"""
    print("\n=== 自定义转换示例 ===")
    
    # 自定义设置
    converter = M4AToMP3Converter(
        input_dir="datasets",
        output_dir="custom_output",
        quality="128k",  # 较低质量，文件更小
        max_workers=1    # 单线程
    )
    
    results = converter.convert_batch()
    print(f"自定义转换结果: {results}")


if __name__ == "__main__":
    print("M4A转换示例脚本")
    print("=" * 50)
    
    # 检查datasets目录是否存在
    if not os.path.exists("datasets"):
        print("错误: datasets目录不存在")
        print("请确保datasets目录中有m4a文件")
        sys.exit(1)
    
    # 运行示例
    try:
        example_basic_conversion()
        # example_high_quality_conversion()
        # example_custom_conversion()
        
        print("\n所有示例执行完成!")
        print("请检查输出目录查看转换结果")
        
    except Exception as e:
        print(f"执行示例时发生错误: {e}")
        sys.exit(1) 