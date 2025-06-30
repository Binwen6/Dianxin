#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
免费语音转文字API转换脚本
支持多种免费语音转文字API服务
"""

import os
import sys
import argparse
import logging
import json
import time
import requests
from pathlib import Path
from typing import List, Optional, Dict, Any
import concurrent.futures
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('speech2text_conversion.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class FreeSpeechToTextAPI:
    """免费语音转文字API服务类"""
    
    def __init__(self):
        self.apis = {
            'whisper': {
                'name': 'OpenAI Whisper (免费)',
                'url': 'https://api.openai.com/v1/audio/transcriptions',
                'headers': {
                    'Authorization': 'Bearer sk-...'  # 需要用户提供API key
                },
                'params': {
                    'model': 'whisper-1',
                    'response_format': 'json'
                }
            },
            'assemblyai': {
                'name': 'AssemblyAI (免费额度)',
                'url': 'https://api.assemblyai.com/v2/upload',
                'headers': {
                    'Authorization': 'your-api-key-here'  # 需要用户提供API key
                }
            },
            'speechmatics': {
                'name': 'Speechmatics (免费试用)',
                'url': 'https://asr.api.speechmatics.com/v2/jobs',
                'headers': {
                    'Authorization': 'Bearer your-api-key-here'  # 需要用户提供API key
                }
            },
            'azure': {
                'name': 'Azure Speech (免费额度)',
                'url': 'https://eastus.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1',
                'headers': {
                    'Ocp-Apim-Subscription-Key': 'your-api-key-here',  # 需要用户提供API key
                    'Content-Type': 'audio/wav'
                }
            }
        }
    
    def transcribe_with_whisper(self, audio_file: Path, api_key: str) -> Optional[str]:
        """使用OpenAI Whisper API转录"""
        try:
            headers = {
                'Authorization': f'Bearer {api_key}'
            }
            
            with open(audio_file, 'rb') as f:
                files = {'file': f}
                data = {
                    'model': 'whisper-1',
                    'response_format': 'json'
                }
                
                response = requests.post(
                    'https://api.openai.com/v1/audio/transcriptions',
                    headers=headers,
                    files=files,
                    data=data
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get('text', '')
                else:
                    logger.error(f"Whisper API错误: {response.status_code} - {response.text}")
                    return None
                    
        except Exception as e:
            logger.error(f"Whisper转录失败: {e}")
            return None
    
    def transcribe_with_assemblyai(self, audio_file: Path, api_key: str) -> Optional[str]:
        """使用AssemblyAI API转录"""
        try:
            headers = {
                'Authorization': api_key
            }
            
            # 上传音频文件
            with open(audio_file, 'rb') as f:
                upload_response = requests.post(
                    'https://api.assemblyai.com/v2/upload',
                    headers=headers,
                    data=f
                )
                
                if upload_response.status_code != 200:
                    logger.error(f"AssemblyAI上传失败: {upload_response.status_code}")
                    return None
                
                upload_url = upload_response.json()['upload_url']
            
            # 创建转录任务
            transcript_request = {
                'audio_url': upload_url,
                'language_code': 'zh'  # 中文
            }
            
            transcript_response = requests.post(
                'https://api.assemblyai.com/v2/transcript',
                json=transcript_request,
                headers=headers
            )
            
            if transcript_response.status_code != 200:
                logger.error(f"AssemblyAI转录请求失败: {transcript_response.status_code}")
                return None
            
            transcript_id = transcript_response.json()['id']
            
            # 轮询获取结果
            polling_endpoint = f"https://api.assemblyai.com/v2/transcript/{transcript_id}"
            
            while True:
                polling_response = requests.get(polling_endpoint, headers=headers)
                polling_response = polling_response.json()
                
                if polling_response['status'] == 'completed':
                    return polling_response['text']
                elif polling_response['status'] == 'error':
                    logger.error(f"AssemblyAI转录错误: {polling_response['error']}")
                    return None
                
                time.sleep(3)
                
        except Exception as e:
            logger.error(f"AssemblyAI转录失败: {e}")
            return None
    
    def transcribe_with_azure(self, audio_file: Path, api_key: str, region: str = 'eastus') -> Optional[str]:
        """使用Azure Speech API转录"""
        try:
            headers = {
                'Ocp-Apim-Subscription-Key': api_key,
                'Content-Type': 'audio/wav'
            }
            
            with open(audio_file, 'rb') as f:
                response = requests.post(
                    f'https://{region}.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1',
                    headers=headers,
                    data=f
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get('DisplayText', '')
                else:
                    logger.error(f"Azure Speech API错误: {response.status_code} - {response.text}")
                    return None
                    
        except Exception as e:
            logger.error(f"Azure转录失败: {e}")
            return None


class SpeechToTextConverter:
    """语音转文字转换器"""
    
    def __init__(self, input_dir: str, output_dir: Optional[str] = None, 
                 api_type: str = 'whisper', api_key: str = '', max_workers: int = 4):
        """
        初始化转换器
        
        Args:
            input_dir: 输入目录路径
            output_dir: 输出目录路径
            api_type: API类型 (whisper, assemblyai, azure)
            api_key: API密钥
            max_workers: 最大并发工作线程数
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir) if output_dir else Path('output')
        self.api_type = api_type
        self.api_key = api_key
        self.max_workers = max_workers
        
        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化API服务
        self.api_service = FreeSpeechToTextAPI()
        
        # 支持的音频格式
        self.supported_formats = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac'}
    
    def find_audio_files(self) -> List[Path]:
        """查找所有支持的音频文件"""
        audio_files = []
        for format_ext in self.supported_formats:
            audio_files.extend(self.input_dir.rglob(f"*{format_ext}"))
        
        logger.info(f"找到 {len(audio_files)} 个音频文件")
        return audio_files
    
    def transcribe_single_file(self, audio_file: Path) -> Dict[str, Any]:
        """
        转录单个音频文件
        
        Args:
            audio_file: 音频文件路径
            
        Returns:
            dict: 转录结果
        """
        try:
            logger.info(f"开始转录: {audio_file.name}")
            
            # 根据API类型调用相应的转录方法
            if self.api_type == 'whisper':
                text = self.api_service.transcribe_with_whisper(audio_file, self.api_key)
            elif self.api_type == 'assemblyai':
                text = self.api_service.transcribe_with_assemblyai(audio_file, self.api_key)
            elif self.api_type == 'azure':
                text = self.api_service.transcribe_with_azure(audio_file, self.api_key)
            else:
                logger.error(f"不支持的API类型: {self.api_type}")
                return {
                    'file': audio_file.name,
                    'success': False,
                    'error': f'不支持的API类型: {self.api_type}'
                }
            
            if text:
                # 保存转录结果
                result = {
                    'file': audio_file.name,
                    'success': True,
                    'text': text,
                    'api_type': self.api_type,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                }
                
                # 保存为文本文件
                output_file = self.output_dir / f"{audio_file.stem}.txt"
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(text)
                
                # 保存为JSON文件
                json_file = self.output_dir / f"{audio_file.stem}.json"
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                
                logger.info(f"转录完成: {audio_file.name}")
                return result
            else:
                return {
                    'file': audio_file.name,
                    'success': False,
                    'error': '转录失败，未获取到文本'
                }
                
        except Exception as e:
            logger.error(f"转录 {audio_file.name} 时发生错误: {e}")
            return {
                'file': audio_file.name,
                'success': False,
                'error': str(e)
            }
    
    def transcribe_batch(self) -> dict:
        """
        批量转录所有音频文件
        
        Returns:
            dict: 转录结果统计
        """
        audio_files = self.find_audio_files()
        
        if not audio_files:
            logger.warning("未找到任何支持的音频文件")
            return {"total": 0, "success": 0, "failed": 0}
        
        logger.info(f"开始批量转录 {len(audio_files)} 个文件...")
        logger.info(f"使用API: {self.api_type}")
        logger.info(f"输出目录: {self.output_dir}")
        
        success_count = 0
        failed_count = 0
        results = []
        
        # 使用线程池进行并发转录
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有转录任务
            future_to_file = {
                executor.submit(self.transcribe_single_file, audio_file): audio_file 
                for audio_file in audio_files
            }
            
            # 使用tqdm显示进度
            with tqdm(total=len(audio_files), desc="转录进度") as pbar:
                for future in concurrent.futures.as_completed(future_to_file):
                    audio_file = future_to_file[future]
                    try:
                        result = future.result()
                        results.append(result)
                        
                        if result['success']:
                            success_count += 1
                        else:
                            failed_count += 1
                    except Exception as e:
                        logger.error(f"处理 {audio_file.name} 时发生异常: {e}")
                        failed_count += 1
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        '成功': success_count,
                        '失败': failed_count
                    })
        
        # 保存汇总结果
        summary = {
            "total": len(audio_files),
            "success": success_count,
            "failed": failed_count,
            "api_type": self.api_type,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "results": results
        }
        
        summary_file = self.output_dir / "transcription_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        # 输出统计结果
        logger.info(f"转录完成! 成功: {success_count}, 失败: {failed_count}")
        logger.info(f"汇总结果已保存到: {summary_file}")
        
        return summary


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="使用免费API批量将音频文件转换为文字",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
支持的API服务:
  whisper     - OpenAI Whisper API (需要API key)
  assemblyai  - AssemblyAI API (需要API key，有免费额度)
  azure       - Azure Speech API (需要API key，有免费额度)

示例用法:
  python SenseVoice.py datasets/ --api whisper --key your-api-key
  python SenseVoice.py datasets/ -o output/ --api assemblyai --key your-api-key
  python SenseVoice.py datasets/ --api azure --key your-api-key --workers 2
        """
    )
    
    parser.add_argument(
        'input_dir',
        help='包含音频文件的输入目录'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='输出目录 (默认: output/)'
    )
    
    parser.add_argument(
        '--api',
        choices=['whisper', 'assemblyai', 'azure'],
        default='whisper',
        help='选择API服务 (默认: whisper)'
    )
    
    parser.add_argument(
        '--key',
        required=True,
        help='API密钥'
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
    
    # 创建转换器并执行转换
    converter = SpeechToTextConverter(
        input_dir=args.input_dir,
        output_dir=args.output,
        api_type=args.api,
        api_key=args.key,
        max_workers=args.workers
    )
    
    result = converter.transcribe_batch()
    
    # 输出最终统计
    print(f"\n转录统计:")
    print(f"总文件数: {result['total']}")
    print(f"成功转录: {result['success']}")
    print(f"转录失败: {result['failed']}")
    print(f"成功率: {result['success']/result['total']*100:.1f}%" if result['total'] > 0 else "成功率: 0%")


if __name__ == "__main__":
    main()