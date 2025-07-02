#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频音频轨道分离工具
使用 moviepy 库从视频文件中提取音频轨道
"""

import os
import sys
from pathlib import Path
from typing import Optional, List, Tuple
import logging
from datetime import datetime

try:
    from moviepy.editor import VideoFileClip
except ImportError:
    print("请先安装 moviepy: pip install moviepy")
    sys.exit(1)


class VideoAudioExtractor:
    """视频音频分离器"""
    
    def __init__(self, output_dir: str = "output/audio", log_dir: str = "log"):
        """
        初始化音频提取器
        
        Args:
            output_dir: 音频输出目录
            log_dir: 日志目录
        """
        self.output_dir = Path(output_dir)
        self.log_dir = Path(log_dir)
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self._setup_logging()
        
        # 支持的视频格式
        self.supported_formats = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v'}
        
        # 支持的音频格式
        self.audio_formats = {'.mp3', '.wav', '.m4a', '.aac', '.flac', '.ogg'}
    
    def _setup_logging(self):
        """设置日志记录"""
        log_file = self.log_dir / f"video_audio_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def extract_audio(self, video_path: str, output_format: str = 'mp3', 
                     output_name: Optional[str] = None) -> Optional[str]:
        """
        从视频文件中提取音频
        
        Args:
            video_path: 视频文件路径
            output_format: 输出音频格式 (mp3, wav, m4a, aac, flac, ogg)
            output_name: 输出文件名（可选，默认使用视频文件名）
            
        Returns:
            提取的音频文件路径，如果失败返回None
        """
        video_path = Path(video_path)
        
        # 检查文件是否存在
        if not video_path.exists():
            self.logger.error(f"视频文件不存在: {video_path}")
            return None
        
        # 检查文件格式
        if video_path.suffix.lower() not in self.supported_formats:
            self.logger.error(f"不支持的视频格式: {video_path.suffix}")
            return None
        
        # 检查输出格式
        if output_format.lower() not in [fmt[1:] for fmt in self.audio_formats]:
            self.logger.error(f"不支持的音频格式: {output_format}")
            return None
        
        try:
            self.logger.info(f"开始处理视频: {video_path}")
            
            # 加载视频文件
            video = VideoFileClip(str(video_path))
            
            # 获取视频信息
            duration = video.duration
            fps = video.fps
            size = video.size
            
            self.logger.info(f"视频信息 - 时长: {duration:.2f}秒, FPS: {fps}, 尺寸: {size}")
            
            # 确定输出文件名
            if output_name is None:
                output_name = video_path.stem
            
            output_path = self.output_dir / f"{output_name}.{output_format}"
            
            # 提取音频
            self.logger.info(f"正在提取音频到: {output_path}")
            audio = video.audio
            
            if audio is None:
                self.logger.error("视频文件中没有音频轨道")
                video.close()
                return None
            
            # 保存音频文件
            audio.write_audiofile(str(output_path), verbose=False, logger=None)
            
            # 清理资源
            audio.close()
            video.close()
            
            self.logger.info(f"音频提取完成: {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"提取音频时发生错误: {str(e)}")
            return None
    
    def batch_extract(self, video_dir: str, output_format: str = 'mp3', 
                     recursive: bool = False) -> List[Tuple[str, str]]:
        """
        批量提取视频音频
        
        Args:
            video_dir: 视频目录路径
            output_format: 输出音频格式
            recursive: 是否递归搜索子目录
            
        Returns:
            成功提取的文件列表 [(视频路径, 音频路径), ...]
        """
        video_dir = Path(video_dir)
        
        if not video_dir.exists():
            self.logger.error(f"视频目录不存在: {video_dir}")
            return []
        
        # 查找视频文件
        video_files = []
        if recursive:
            video_files = list(video_dir.rglob("*"))
        else:
            video_files = list(video_dir.glob("*"))
        
        # 过滤视频文件
        video_files = [f for f in video_files if f.is_file() and f.suffix.lower() in self.supported_formats]
        
        if not video_files:
            self.logger.warning(f"在目录 {video_dir} 中没有找到支持的视频文件")
            return []
        
        self.logger.info(f"找到 {len(video_files)} 个视频文件")
        
        results = []
        for video_file in video_files:
            self.logger.info(f"处理文件 {video_file.name} ({video_files.index(video_file) + 1}/{len(video_files)})")
            
            audio_path = self.extract_audio(str(video_file), output_format)
            if audio_path:
                results.append((str(video_file), audio_path))
        
        self.logger.info(f"批量提取完成，成功处理 {len(results)}/{len(video_files)} 个文件")
        return results
    
    def get_video_info(self, video_path: str) -> Optional[dict]:
        """
        获取视频文件信息
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            视频信息字典，如果失败返回None
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            self.logger.error(f"视频文件不存在: {video_path}")
            return None
        
        try:
            video = VideoFileClip(str(video_path))
            
            info = {
                'file_path': str(video_path),
                'file_size': video_path.stat().st_size,
                'duration': video.duration,
                'fps': video.fps,
                'size': video.size,
                'has_audio': video.audio is not None,
                'audio_fps': video.audio.fps if video.audio else None,
                'audio_nchannels': video.audio.nchannels if video.audio else None
            }
            
            video.close()
            return info
            
        except Exception as e:
            self.logger.error(f"获取视频信息时发生错误: {str(e)}")
            return None


def main():
    """主函数 - 命令行接口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="视频音频轨道分离工具")
    parser.add_argument("input", help="输入视频文件或目录路径")
    parser.add_argument("-o", "--output", default="output/audio", help="输出目录 (默认: output/audio)")
    parser.add_argument("-f", "--format", default="mp3", choices=["mp3", "wav", "m4a", "aac", "flac", "ogg"], 
                       help="输出音频格式 (默认: mp3)")
    parser.add_argument("-r", "--recursive", action="store_true", help="递归搜索子目录")
    parser.add_argument("--info", action="store_true", help="只显示视频信息，不提取音频")
    
    args = parser.parse_args()
    
    extractor = VideoAudioExtractor(args.output)
    
    if args.info:
        # 只显示信息
        if os.path.isfile(args.input):
            info = extractor.get_video_info(args.input)
            if info:
                print(f"视频信息:")
                print(f"  文件路径: {info['file_path']}")
                print(f"  文件大小: {info['file_size']} 字节")
                print(f"  时长: {info['duration']:.2f} 秒")
                print(f"  FPS: {info['fps']}")
                print(f"  尺寸: {info['size']}")
                print(f"  有音频: {info['has_audio']}")
                if info['has_audio']:
                    print(f"  音频FPS: {info['audio_fps']}")
                    print(f"  音频声道数: {info['audio_nchannels']}")
        else:
            print("请指定一个视频文件来查看信息")
    else:
        # 提取音频
        if os.path.isfile(args.input):
            # 单个文件
            result = extractor.extract_audio(args.input, args.format)
            if result:
                print(f"音频提取成功: {result}")
            else:
                print("音频提取失败")
        elif os.path.isdir(args.input):
            # 目录批量处理
            results = extractor.batch_extract(args.input, args.format, args.recursive)
            print(f"批量处理完成，成功提取 {len(results)} 个音频文件")
        else:
            print(f"输入路径不存在: {args.input}")


if __name__ == "__main__":
    main() 