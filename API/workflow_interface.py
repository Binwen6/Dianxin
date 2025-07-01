#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工作流集成接口
提供标准化的输入输出格式，便于与视频转音频和文字转PPT环节对接
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

from workflow_processor import WorkflowSpeechProcessor, process_audio_for_workflow


class WorkflowInterface:
    """工作流集成接口"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化工作流接口
        
        Args:
            config: 配置参数
        """
        self.config = config or {}
        self.processor = WorkflowSpeechProcessor(config)
        self.logger = logging.getLogger('WorkflowInterface')
    
    def process_from_video_converter(self, audio_input: Union[str, Path], 
                                   workflow_id: Optional[str] = None) -> Dict[str, Any]:
        """
        从视频转音频环节接收输入并处理
        
        Args:
            audio_input: 音频文件或目录路径
            workflow_id: 工作流ID，用于追踪
            
        Returns:
            标准化的输出结果
        """
        workflow_id = workflow_id or f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.logger.info(f"开始处理工作流 {workflow_id} 的音频输入")
        
        # 处理音频
        result = process_audio_for_workflow(audio_input, config=self.config)
        
        # 标准化输出格式
        standardized_output = self._standardize_output(result, workflow_id)
        
        # 保存工作流状态
        self._save_workflow_state(standardized_output, workflow_id)
        
        return standardized_output
    
    def prepare_for_ppt_generator(self, speech_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        为PPT生成环节准备数据
        
        Args:
            speech_result: 语音转文字结果
            
        Returns:
            适合PPT生成的数据格式
        """
        if speech_result.get('status') == 'completed':
            # 批量处理结果
            processed_files = speech_result.get('processed_files', [])
            ppt_data = {
                'workflow_step': 'speech2text',
                'content_type': 'transcription',
                'total_slides': len(processed_files),
                'slides': [],
                'metadata': {
                    'total_words': speech_result.get('summary', {}).get('total_words', 0),
                    'total_characters': speech_result.get('summary', {}).get('total_characters', 0),
                    'processing_time': speech_result.get('total_processing_time', 0),
                    'timestamp': speech_result.get('timestamp', '')
                }
            }
            
            for i, file_result in enumerate(processed_files):
                if file_result.get('status') == 'success':
                    slide_data = {
                        'slide_number': i + 1,
                        'title': f"音频转写 {i + 1}: {file_result.get('file_name', 'Unknown')}",
                        'content': file_result.get('transcription', ''),
                        'source_file': file_result.get('file_name', ''),
                        'word_count': file_result.get('metadata', {}).get('word_count', 0),
                        'duration': file_result.get('metadata', {}).get('audio_duration', 0),
                        'suggested_layout': self._suggest_slide_layout(file_result.get('transcription', ''))
                    }
                    ppt_data['slides'].append(slide_data)
            
            return ppt_data
        
        elif speech_result.get('status') == 'success':
            # 单个文件结果
            return {
                'workflow_step': 'speech2text',
                'content_type': 'transcription',
                'total_slides': 1,
                'slides': [{
                    'slide_number': 1,
                    'title': f"音频转写: {speech_result.get('file_name', 'Unknown')}",
                    'content': speech_result.get('transcription', ''),
                    'source_file': speech_result.get('file_name', ''),
                    'word_count': speech_result.get('metadata', {}).get('word_count', 0),
                    'duration': speech_result.get('metadata', {}).get('audio_duration', 0),
                    'suggested_layout': self._suggest_slide_layout(speech_result.get('transcription', ''))
                }],
                'metadata': {
                    'total_words': speech_result.get('metadata', {}).get('word_count', 0),
                    'total_characters': speech_result.get('metadata', {}).get('character_count', 0),
                    'processing_time': speech_result.get('processing_time', 0),
                    'timestamp': speech_result.get('timestamp', '')
                }
            }
        
        else:
            return {
                'workflow_step': 'speech2text',
                'content_type': 'error',
                'error': speech_result.get('error', 'Unknown error'),
                'status': 'failed'
            }
    
    def _standardize_output(self, result: Dict[str, Any], workflow_id: str) -> Dict[str, Any]:
        """标准化输出格式"""
        standardized = {
            'workflow_id': workflow_id,
            'step_name': 'speech2text',
            'step_version': '1.0',
            'input_type': 'audio',
            'output_type': 'text',
            'timestamp': datetime.now().isoformat(),
            'status': result.get('status', 'unknown'),
            'result': result
        }
        
        # 添加输出文件路径
        if 'output_directory' in result:
            standardized['output_files'] = {
                'directory': result['output_directory'],
                'detailed_results': f"{result['output_directory']}/speech2text_results_*.json",
                'text_content': f"{result['output_directory']}/speech2text_content_*.txt",
                'workflow_metadata': f"{result['output_directory']}/workflow_metadata_*.json"
            }
        
        return standardized
    
    def _save_workflow_state(self, output: Dict[str, Any], workflow_id: str):
        """保存工作流状态"""
        state_dir = Path('workflow_states')
        state_dir.mkdir(exist_ok=True)
        
        state_file = state_dir / f'{workflow_id}_speech2text.json'
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"工作流状态已保存: {state_file}")
    
    def _suggest_slide_layout(self, text: str) -> str:
        """根据文本内容建议PPT布局"""
        word_count = len(text.split())
        
        if word_count < 50:
            return 'title_and_content'  # 标题+内容
        elif word_count < 200:
            return 'content_with_bullets'  # 带项目符号的内容
        else:
            return 'content_split'  # 分页内容
    
    def get_workflow_info(self) -> Dict[str, Any]:
        """获取工作流信息"""
        return {
            'step_name': 'speech2text',
            'step_description': '语音转文字处理',
            'input_formats': ['mp3', 'wav', 'm4a', 'flac', 'aac', 'ogg'],
            'output_formats': ['json', 'txt'],
            'dependencies': ['funasr', 'torch'],
            'version': '1.0',
            'processor_status': self.processor.get_workflow_status()
        }


# 便捷函数，用于快速集成
def quick_speech2text(audio_input: Union[str, Path], 
                     output_dir: Optional[Union[str, Path]] = None,
                     workflow_id: Optional[str] = None) -> Dict[str, Any]:
    """
    快速语音转文字处理，返回适合PPT生成的格式
    
    Args:
        audio_input: 音频输入路径
        output_dir: 输出目录
        workflow_id: 工作流ID
        
    Returns:
        适合PPT生成的数据格式
    """
    interface = WorkflowInterface()
    
    # 处理音频
    result = interface.process_from_video_converter(audio_input, workflow_id)
    
    # 准备PPT数据
    ppt_data = interface.prepare_for_ppt_generator(result['result'])
    
    return ppt_data


def batch_speech2text_for_ppt(audio_dir: Union[str, Path],
                            output_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    批量处理音频文件，专门为PPT生成准备数据
    
    Args:
        audio_dir: 音频文件目录
        output_dir: 输出目录
        
    Returns:
        PPT生成所需的数据格式
    """
    interface = WorkflowInterface()
    
    # 处理音频目录
    result = interface.process_from_video_converter(audio_dir)
    
    # 准备PPT数据
    ppt_data = interface.prepare_for_ppt_generator(result['result'])
    
    return ppt_data


if __name__ == '__main__':
    # 测试示例
    import argparse
    
    parser = argparse.ArgumentParser(description='工作流语音转文字接口')
    parser.add_argument('input', help='输入音频文件或目录')
    parser.add_argument('-o', '--output', help='输出目录')
    parser.add_argument('--workflow-id', help='工作流ID')
    parser.add_argument('--ppt-format', action='store_true', help='输出PPT格式数据')
    
    args = parser.parse_args()
    
    if args.ppt_format:
        # 输出PPT格式数据
        result = quick_speech2text(args.input, args.output, args.workflow_id)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        # 标准处理
        interface = WorkflowInterface()
        result = interface.process_from_video_converter(args.input, args.workflow_id)
        print(f"处理完成，状态: {result['status']}")
        print(f"工作流ID: {result['workflow_id']}") 