#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M4A批量转换为MP3脚本
支持批量转换指定目录下的所有m4a文件为mp3格式
"""

import os
import sys
import argparse
import subprocess
import logging
from pathlib import Path
from typing import List, Optional
import concurrent.futures
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('m4a_conversion.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class M4AToMP3Converter:
    """M4A到MP3转换器"""
    
    def __init__(self, input_dir: str, output_dir: Optional[str] = None, 
                 quality: str = "192k", max_workers: int = 4):
        """
        初始化转换器
        
        Args:
            input_dir: 输入目录路径
            output_dir: 输出目录路径，如果为None则使用输入目录
            quality: MP3质量 (如: 128k, 192k, 320k)
            max_workers: 最大并发工作线程数
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir) if output_dir else self.input_dir
        self.quality = quality
        self.max_workers = max_workers
        
        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 检查ffmpeg是否可用
        self._check_ffmpeg()
    
    def _check_ffmpeg(self):
        """检查ffmpeg是否已安装"""
        try:
            subprocess.run(['ffmpeg', '-version'], 
                         capture_output=True, check=True)
            logger.info("FFmpeg 已安装")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("FFmpeg 未安装或不在PATH中")
            logger.error("请安装FFmpeg: https://ffmpeg.org/download.html")
            sys.exit(1)
    
    def find_m4a_files(self) -> List[Path]:
        """查找所有m4a文件"""
        m4a_files = list(self.input_dir.rglob("*.m4a"))
        logger.info(f"找到 {len(m4a_files)} 个m4a文件")
        return m4a_files
    
    def convert_single_file(self, m4a_file: Path) -> bool:
        """
        转换单个m4a文件为mp3
        
        Args:
            m4a_file: m4a文件路径
            
        Returns:
            bool: 转换是否成功
        """
        try:
            # 计算相对路径以保持目录结构
            relative_path = m4a_file.relative_to(self.input_dir)
            mp3_file = self.output_dir / relative_path.with_suffix('.mp3')
            
            # 确保输出文件的父目录存在
            mp3_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 构建ffmpeg命令
            cmd = [
                'ffmpeg',
                '-i', str(m4a_file),
                '-acodec', 'libmp3lame',
                '-ab', self.quality,
                '-y',  # 覆盖已存在的文件
                str(mp3_file)
            ]
            
            # 执行转换
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            logger.debug(f"成功转换: {m4a_file.name} -> {mp3_file.name}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"转换失败 {m4a_file.name}: {e}")
            logger.error(f"错误输出: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"转换 {m4a_file.name} 时发生未知错误: {e}")
            return False
    
    def convert_batch(self) -> dict:
        """
        批量转换所有m4a文件
        
        Returns:
            dict: 转换结果统计
        """
        m4a_files = self.find_m4a_files()
        
        if not m4a_files:
            logger.warning("未找到任何m4a文件")
            return {"total": 0, "success": 0, "failed": 0}
        
        logger.info(f"开始批量转换 {len(m4a_files)} 个文件...")
        logger.info(f"输出目录: {self.output_dir}")
        logger.info(f"MP3质量: {self.quality}")
        
        success_count = 0
        failed_count = 0
        
        # 使用线程池进行并发转换
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有转换任务
            future_to_file = {
                executor.submit(self.convert_single_file, m4a_file): m4a_file 
                for m4a_file in m4a_files
            }
            
            # 使用tqdm显示进度
            with tqdm(total=len(m4a_files), desc="转换进度") as pbar:
                for future in concurrent.futures.as_completed(future_to_file):
                    m4a_file = future_to_file[future]
                    try:
                        success = future.result()
                        if success:
                            success_count += 1
                        else:
                            failed_count += 1
                    except Exception as e:
                        logger.error(f"处理 {m4a_file.name} 时发生异常: {e}")
                        failed_count += 1
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        '成功': success_count,
                        '失败': failed_count
                    })
        
        # 输出统计结果
        logger.info(f"转换完成! 成功: {success_count}, 失败: {failed_count}")
        
        return {
            "total": len(m4a_files),
            "success": success_count,
            "failed": failed_count
        }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="批量将M4A文件转换为MP3格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python m4a_to_mp3.py datasets/                    # 转换datasets目录下的所有m4a文件
  python m4a_to_mp3.py datasets/ -o output/         # 指定输出目录
  python m4a_to_mp3.py datasets/ -q 320k            # 设置MP3质量为320k
  python m4a_to_mp3.py datasets/ -w 8               # 使用8个并发线程
        """
    )
    
    parser.add_argument(
        'input_dir',
        help='包含m4a文件的输入目录'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='输出目录 (默认与输入目录相同)'
    )
    
    parser.add_argument(
        '-q', '--quality',
        default='192k',
        help='MP3质量 (默认: 192k, 可选: 128k, 192k, 320k)'
    )
    
    parser.add_argument(
        '-w', '--workers',
        type=int,
        default=4,
        help='最大并发工作线程数 (默认: 4)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='显示详细日志'
    )
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 检查输入目录是否存在
    if not os.path.exists(args.input_dir):
        logger.error(f"输入目录不存在: {args.input_dir}")
        sys.exit(1)
    
    # 创建转换器并执行转换
    converter = M4AToMP3Converter(
        input_dir=args.input_dir,
        output_dir=args.output,
        quality=args.quality,
        max_workers=args.workers
    )
    
    results = converter.convert_batch()
    
    # 输出最终结果
    print(f"\n转换统计:")
    print(f"总文件数: {results['total']}")
    print(f"成功转换: {results['success']}")
    print(f"转换失败: {results['failed']}")
    
    if results['failed'] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

