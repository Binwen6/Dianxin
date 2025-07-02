#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频音频分离使用示例
"""

import sys
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append(str(Path(__file__).parent))

from video_audio_extractor import VideoAudioExtractor


def example_single_file():
    """单个文件提取示例"""
    print("=== 单个文件音频提取示例 ===")
    
    # 创建提取器
    extractor = VideoAudioExtractor(output_dir="output/audio")
    
    # 示例：提取单个视频文件的音频
    # 请将下面的路径替换为实际的视频文件路径
    video_file = "datasets/video/example.mp4"  # 替换为实际路径
    
    if Path(video_file).exists():
        # 提取为MP3格式
        result = extractor.extract_audio(video_file, output_format="mp3")
        if result:
            print(f"✅ 音频提取成功: {result}")
        else:
            print("❌ 音频提取失败")
    else:
        print(f"⚠️  示例视频文件不存在: {video_file}")
        print("请将视频文件放在 datasets/video/ 目录下")


def example_batch_extract():
    """批量提取示例"""
    print("\n=== 批量音频提取示例 ===")
    
    # 创建提取器
    extractor = VideoAudioExtractor(output_dir="output/audio")
    
    # 批量提取datasets/video目录下的所有视频文件
    video_dir = "datasets/video"
    
    if Path(video_dir).exists():
        results = extractor.batch_extract(video_dir, output_format="mp3", recursive=True)
        
        if results:
            print(f"✅ 批量提取完成，成功处理 {len(results)} 个文件:")
            for video_path, audio_path in results:
                print(f"  📹 {Path(video_path).name} → 🎵 {Path(audio_path).name}")
        else:
            print("⚠️  没有找到可处理的视频文件")
    else:
        print(f"⚠️  视频目录不存在: {video_dir}")


def example_get_info():
    """获取视频信息示例"""
    print("\n=== 视频信息获取示例 ===")
    
    # 创建提取器
    extractor = VideoAudioExtractor()
    
    # 示例：获取视频文件信息
    video_file = "datasets/video/example.mp4"  # 替换为实际路径
    
    if Path(video_file).exists():
        info = extractor.get_video_info(video_file)
        if info:
            print("📹 视频信息:")
            print(f"  文件路径: {info['file_path']}")
            print(f"  文件大小: {info['file_size']:,} 字节")
            print(f"  时长: {info['duration']:.2f} 秒")
            print(f"  FPS: {info['fps']}")
            print(f"  尺寸: {info['size']}")
            print(f"  有音频: {'是' if info['has_audio'] else '否'}")
            if info['has_audio']:
                print(f"  音频FPS: {info['audio_fps']}")
                print(f"  音频声道数: {info['audio_nchannels']}")
        else:
            print("❌ 获取视频信息失败")
    else:
        print(f"⚠️  示例视频文件不存在: {video_file}")


def example_different_formats():
    """不同格式提取示例"""
    print("\n=== 不同音频格式提取示例 ===")
    
    # 创建提取器
    extractor = VideoAudioExtractor(output_dir="output/audio")
    
    video_file = "datasets/video/example.mp4"  # 替换为实际路径
    
    if Path(video_file).exists():
        # 提取为不同格式
        formats = ["mp3", "wav", "m4a"]
        
        for fmt in formats:
            print(f"正在提取为 {fmt.upper()} 格式...")
            result = extractor.extract_audio(video_file, output_format=fmt)
            if result:
                print(f"✅ {fmt.upper()} 格式提取成功: {Path(result).name}")
            else:
                print(f"❌ {fmt.upper()} 格式提取失败")
    else:
        print(f"⚠️  示例视频文件不存在: {video_file}")


def main():
    """主函数"""
    print("🎬 视频音频分离工具使用示例")
    print("=" * 50)
    
    # 运行各种示例
    example_single_file()
    example_batch_extract()
    example_get_info()
    example_different_formats()
    
    print("\n" + "=" * 50)
    print("📝 使用说明:")
    print("1. 将视频文件放在 datasets/video/ 目录下")
    print("2. 运行示例脚本查看效果")
    print("3. 使用命令行工具进行批量处理:")
    print("   python utils/video_audio_extractor.py datasets/video -f mp3")
    print("4. 查看视频信息:")
    print("   python utils/video_audio_extractor.py video.mp4 --info")


if __name__ == "__main__":
    main() 