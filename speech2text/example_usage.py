#!/usr/bin/env python3
"""
Whisper转录器使用示例 - 包含时间戳功能
"""

from whisper import WhisperTranscriber
import json
from pathlib import Path

def example_with_timestamps():
    """使用时间戳功能的示例"""
    print("=== 带时间戳的转录示例 ===\n")
    
    # 初始化转录器（启用时间戳）
    transcriber = WhisperTranscriber(return_timestamps=True)
    
    # 转录单个文件
    audio_file = "datasets/mp3/test.mp3"
    if Path(audio_file).exists():
        print(f"转录文件: {audio_file}")
        result = transcriber.transcribe_file(audio_file)
        
        if 'error' not in result:
            print(f"转录文本: {result['transcription']}")
            print(f"文件大小: {result['file_size_bytes']} bytes")
            print(f"使用模型: {result['model_used']}")
            print(f"使用设备: {result['device_used']}")
            
            # 显示段落级别时间戳
            if 'segment_timestamps' in result:
                print(f"\n段落级别时间戳:")
                for i, segment in enumerate(result['segment_timestamps'], 1):
                    print(f"  段落 {i}: [{segment['start']:.2f}s - {segment['end']:.2f}s] {segment['text']}")
            
            # 显示词级别时间戳（前10个词）
            if 'word_timestamps' in result:
                print(f"\n词级别时间戳（前10个）:")
                for i, word_info in enumerate(result['word_timestamps'][:10], 1):
                    print(f"  词 {i}: [{word_info['start']:.2f}s - {word_info['end']:.2f}s] {word_info['word']}")
        else:
            print(f"转录失败: {result['error']}")
    else:
        print(f"文件不存在: {audio_file}")

def example_without_timestamps():
    """不使用时间戳功能的示例"""
    print("\n=== 不带时间戳的转录示例 ===\n")
    
    # 初始化转录器（不启用时间戳）
    transcriber = WhisperTranscriber(return_timestamps=False)
    
    # 转录单个文件
    audio_file = "datasets/mp3/test.mp3"
    if Path(audio_file).exists():
        print(f"转录文件: {audio_file}")
        result = transcriber.transcribe_file(audio_file)
        
        if 'error' not in result:
            print(f"转录文本: {result['transcription']}")
            print(f"文件大小: {result['file_size_bytes']} bytes")
            print(f"使用模型: {result['model_used']}")
            print(f"使用设备: {result['device_used']}")
        else:
            print(f"转录失败: {result['error']}")
    else:
        print(f"文件不存在: {audio_file}")

def example_batch_processing():
    """批量处理示例"""
    print("\n=== 批量处理示例 ===\n")
    
    # 初始化转录器（启用时间戳）
    transcriber = WhisperTranscriber(return_timestamps=True)
    
    # 批量转录
    input_dir = "datasets"
    output_dir = "output"
    
    if Path(input_dir).exists():
        print(f"批量转录目录: {input_dir}")
        results = transcriber.transcribe_directory(input_dir, output_dir)
        
        print(f"\n处理结果:")
        print(f"总文件数: {len(results)}")
        
        successful = len([r for r in results if 'error' not in r])
        failed = len([r for r in results if 'error' in r])
        
        print(f"成功: {successful}")
        print(f"失败: {failed}")
        
        # 显示成功转录的文件信息
        for result in results:
            if 'error' not in result:
                print(f"\n文件: {result['file_name']}")
                print(f"转录长度: {len(result['transcription'])} 字符")
                if 'segment_timestamps' in result:
                    print(f"段落数: {len(result['segment_timestamps'])}")
                if 'word_timestamps' in result:
                    print(f"词数: {len(result['word_timestamps'])}")
    else:
        print(f"目录不存在: {input_dir}")

def example_srt_generation():
    """SRT字幕生成示例"""
    print("\n=== SRT字幕生成示例 ===\n")
    
    # 初始化转录器（启用时间戳）
    transcriber = WhisperTranscriber(return_timestamps=True)
    
    # 转录文件
    audio_file = "datasets/test.mp3"
    if Path(audio_file).exists():
        print(f"转录文件并生成SRT字幕: {audio_file}")
        result = transcriber.transcribe_file(audio_file)
        
        if 'error' not in result and 'segment_timestamps' in result:
            # 手动生成SRT文件
            srt_content = []
            subtitle_index = 1
            
            for segment in result['segment_timestamps']:
                # 转换时间格式
                start_time = transcriber._seconds_to_srt_time(segment['start'])
                end_time = transcriber._seconds_to_srt_time(segment['end'])
                
                srt_content.append(f"{subtitle_index}")
                srt_content.append(f"{start_time} --> {end_time}")
                srt_content.append(f"{segment['text']}")
                srt_content.append("")
                
                subtitle_index += 1
            
            # 保存SRT文件
            srt_file = "output/manual_subtitle.srt"
            Path("output").mkdir(exist_ok=True)
            
            with open(srt_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(srt_content))
            
            print(f"SRT字幕已保存到: {srt_file}")
            print(f"包含 {len(result['segment_timestamps'])} 个字幕段落")
            
            # 显示前几个字幕
            print("\n前3个字幕段落:")
            for i, segment in enumerate(result['segment_timestamps'][:3], 1):
                start_time = transcriber._seconds_to_srt_time(segment['start'])
                end_time = transcriber._seconds_to_srt_time(segment['end'])
                print(f"  {i}. [{start_time} --> {end_time}] {segment['text']}")
        else:
            print(f"转录失败或没有时间戳信息")
    else:
        print(f"文件不存在: {audio_file}")

def main():
    """主函数"""
    print("Whisper转录器时间戳功能演示")
    print("=" * 50)
    
    # 运行各种示例
    example_with_timestamps()
    example_without_timestamps()
    example_batch_processing()
    example_srt_generation()
    
    print("\n" + "=" * 50)
    print("演示完成！")
    print("\n输出文件说明:")
    print("- JSON文件: 包含完整的转录结果和时间戳信息")
    print("- 文本文件: 人类可读的转录结果")
    print("- SRT文件: 标准字幕格式，可用于视频播放器")
    print("- 摘要文件: 处理统计信息")

if __name__ == "__main__":
    main() 