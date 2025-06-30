#!/usr/bin/env python3
"""
Whisper转录器命令行工具
支持时间戳功能的语音转录CLI
"""

import argparse
import sys
from pathlib import Path
from whisper import WhisperTranscriber

def main():
    parser = argparse.ArgumentParser(
        description="Whisper语音转录器 - 支持时间戳功能",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 转录单个文件（不启用时间戳）
  python cli.py transcribe datasets/mp3/test.mp3

  # 转录单个文件（启用时间戳）
  python cli.py transcribe datasets/mp3/test.mp3 --timestamps

  # 批量转录目录
  python cli.py batch datasets output --timestamps

  # 使用不同模型
  python cli.py transcribe test.mp3 --model openai/whisper-base --timestamps

  # 自定义音频分块长度
  python cli.py transcribe long_audio.mp3 --timestamps --chunk-length 60
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 单个文件转录命令
    transcribe_parser = subparsers.add_parser('transcribe', help='转录单个音频文件')
    transcribe_parser.add_argument('file', help='音频文件路径')
    transcribe_parser.add_argument('--timestamps', action='store_true', help='启用时间戳功能')
    transcribe_parser.add_argument('--model', default='openai/whisper-large-v3-turbo', help='使用的模型ID')
    transcribe_parser.add_argument('--chunk-length', type=int, default=30, help='音频分块长度（秒）')
    transcribe_parser.add_argument('--output', help='输出目录（默认：output）')
    
    # 批量转录命令
    batch_parser = subparsers.add_parser('batch', help='批量转录目录中的音频文件')
    batch_parser.add_argument('input_dir', help='输入目录路径')
    batch_parser.add_argument('output_dir', help='输出目录路径')
    batch_parser.add_argument('--timestamps', action='store_true', help='启用时间戳功能')
    batch_parser.add_argument('--model', default='openai/whisper-large-v3-turbo', help='使用的模型ID')
    
    # 信息命令
    info_parser = subparsers.add_parser('info', help='显示系统信息')
    
    args = parser.parse_args()
    
    if args.command == 'transcribe':
        transcribe_single_file(args)
    elif args.command == 'batch':
        transcribe_batch(args)
    elif args.command == 'info':
        show_info()
    else:
        parser.print_help()
        sys.exit(1)

def transcribe_single_file(args):
    """转录单个文件"""
    file_path = Path(args.file)
    
    if not file_path.exists():
        print(f"错误: 文件不存在 - {args.file}")
        sys.exit(1)
    
    print(f"转录文件: {args.file}")
    print(f"使用模型: {args.model}")
    print(f"时间戳功能: {'启用' if args.timestamps else '禁用'}")
    print("-" * 50)
    
    # 初始化转录器
    transcriber = WhisperTranscriber(
        model_id=args.model,
        return_timestamps=args.timestamps
    )
    
    # 转录文件
    result = transcriber.transcribe_file(args.file, chunk_length_s=args.chunk_length)
    
    if 'error' in result:
        print(f"转录失败: {result['error']}")
        sys.exit(1)
    
    # 显示结果
    print(f"\n转录结果:")
    print(f"文件: {result['file_name']}")
    print(f"大小: {result['file_size_bytes']} bytes")
    print(f"设备: {result['device_used']}")
    print(f"文本: {result['transcription']}")
    
    # 显示时间戳信息
    if args.timestamps and 'segment_timestamps' in result:
        print(f"\n时间戳信息:")
        for i, segment in enumerate(result['segment_timestamps'], 1):
            print(f"  {i:2d}. [{segment['start']:6.2f}s - {segment['end']:6.2f}s] {segment['text']}")
    
    # 保存结果
    output_dir = args.output or "output"
    Path(output_dir).mkdir(exist_ok=True)
    
    # 保存JSON结果
    import json
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    json_file = Path(output_dir) / f"transcription_{timestamp}.json"
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存到: {json_file}")
    
    # 如果启用了时间戳，保存SRT文件
    if args.timestamps and 'segment_timestamps' in result:
        srt_file = Path(output_dir) / f"subtitle_{timestamp}.srt"
        transcriber._save_srt_file([result], srt_file)
        print(f"字幕文件已保存到: {srt_file}")

def transcribe_batch(args):
    """批量转录"""
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    
    if not input_path.exists():
        print(f"错误: 输入目录不存在 - {args.input_dir}")
        sys.exit(1)
    
    print(f"批量转录:")
    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"使用模型: {args.model}")
    print(f"时间戳功能: {'启用' if args.timestamps else '禁用'}")
    print("-" * 50)
    
    # 初始化转录器
    transcriber = WhisperTranscriber(
        model_id=args.model,
        return_timestamps=args.timestamps
    )
    
    # 批量转录
    results = transcriber.transcribe_directory(args.input_dir, args.output_dir)
    
    # 显示摘要
    successful = len([r for r in results if 'error' not in r])
    failed = len([r for r in results if 'error' in r])
    
    print(f"\n处理完成!")
    print(f"总文件数: {len(results)}")
    print(f"成功: {successful}")
    print(f"失败: {failed}")
    
    # 显示失败的文件
    if failed > 0:
        print(f"\n失败的文件:")
        for result in results:
            if 'error' in result:
                print(f"  - {result['file_name']}: {result['error']}")
    
    # 显示成功文件的时间戳示例
    if args.timestamps:
        for result in results:
            if 'error' not in result and 'segment_timestamps' in result:
                print(f"\n文件: {result['file_name']}")
                print("时间戳示例:")
                for segment in result['segment_timestamps'][:3]:
                    print(f"  [{segment['start']:.2f}s - {segment['end']:.2f}s] {segment['text']}")
                break

def show_info():
    """显示系统信息"""
    import torch
    
    print("Whisper转录器系统信息")
    print("=" * 50)
    
    # CUDA信息
    cuda_available = torch.cuda.is_available()
    print(f"CUDA可用: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # PyTorch信息
    print(f"PyTorch版本: {torch.__version__}")
    
    # 可用模型
    print(f"\n可用模型:")
    models = [
        "openai/whisper-tiny",
        "openai/whisper-base", 
        "openai/whisper-small",
        "openai/whisper-medium",
        "openai/whisper-large-v3",
        "openai/whisper-large-v3-turbo"
    ]
    
    for model in models:
        print(f"  - {model}")
    
    # 支持的音频格式
    print(f"\n支持的音频格式:")
    formats = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac']
    for fmt in formats:
        print(f"  - {fmt}")
    
    print(f"\n输出格式:")
    print(f"  - JSON: 完整转录结果和时间戳")
    print(f"  - TXT: 人类可读的转录文本")
    print(f"  - SRT: 标准字幕格式")
    print(f"  - 摘要: 处理统计信息")

if __name__ == "__main__":
    main() 