#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工作流集成示例
展示如何在视频转音频 -> 语音转文字 -> 文字转PPT 工作流中使用
"""

import json
from pathlib import Path
from workflow_interface import WorkflowInterface, quick_speech2text, batch_speech2text_for_ppt


def example_single_file_workflow():
    """示例：单个文件的工作流处理"""
    print("=== 单个文件工作流示例 ===")
    
    # 假设这是从视频转音频环节得到的音频文件
    audio_file = "../datasets/mp3/test2.mp3"
    
    # 使用快速接口处理
    ppt_data = quick_speech2text(
        audio_input=audio_file,
        output_dir="../output/workflow_single",
        workflow_id="example_single_001"
    )
    
    print(f"处理完成，生成了 {ppt_data['total_slides']} 个PPT页面")
    print(f"总字数: {ppt_data['metadata']['total_words']}")
    
    # 这里可以将 ppt_data 传递给下游的PPT生成环节
    return ppt_data


def example_batch_workflow():
    """示例：批量文件的工作流处理"""
    print("\n=== 批量文件工作流示例 ===")
    
    # 假设这是从视频转音频环节得到的音频目录
    audio_dir = "../datasets/mp3"
    
    # 批量处理
    ppt_data = batch_speech2text_for_ppt(
        audio_dir=audio_dir,
        output_dir="../output/workflow_batch"
    )
    
    print(f"批量处理完成，生成了 {ppt_data['total_slides']} 个PPT页面")
    print(f"总字数: {ppt_data['metadata']['total_words']}")
    
    # 显示每个页面的信息
    for slide in ppt_data['slides']:
        print(f"页面 {slide['slide_number']}: {slide['title']}")
        print(f"  字数: {slide['word_count']}, 建议布局: {slide['suggested_layout']}")
    
    return ppt_data


def example_workflow_integration():
    """示例：完整的工作流集成"""
    print("\n=== 完整工作流集成示例 ===")
    
    # 创建工作流接口
    interface = WorkflowInterface()
    
    # 获取工作流信息
    workflow_info = interface.get_workflow_info()
    print(f"工作流步骤: {workflow_info['step_name']}")
    print(f"支持格式: {workflow_info['input_formats']}")
    
    # 模拟从上游接收音频数据
    audio_input = "../datasets/mp3"
    
    # 处理音频（模拟从视频转音频环节接收）
    result = interface.process_from_video_converter(
        audio_input=audio_input,
        workflow_id="integration_example_001"
    )
    
    print(f"工作流ID: {result['workflow_id']}")
    print(f"处理状态: {result['status']}")
    
    # 为下游PPT生成准备数据
    ppt_data = interface.prepare_for_ppt_generator(result['result'])
    
    # 模拟传递给下游PPT生成环节
    print(f"准备传递给PPT生成环节的数据:")
    print(f"  - 总页面数: {ppt_data['total_slides']}")
    print(f"  - 总字数: {ppt_data['metadata']['total_words']}")
    print(f"  - 处理时间: {ppt_data['metadata']['processing_time']:.2f}秒")
    
    return result, ppt_data


def example_ppt_data_format():
    """示例：展示PPT数据格式"""
    print("\n=== PPT数据格式示例 ===")
    
    # 处理单个文件并获取PPT格式数据
    ppt_data = quick_speech2text(
        audio_input="../datasets/mp3/test2.mp3",
        workflow_id="format_example_001"
    )
    
    # 展示数据结构
    print("PPT数据格式:")
    print(json.dumps(ppt_data, ensure_ascii=False, indent=2))
    
    return ppt_data


def simulate_upstream_downstream_integration():
    """模拟与上下游环节的集成"""
    print("\n=== 上下游集成模拟 ===")
    
    # 模拟上游：视频转音频环节的输出
    upstream_output = {
        'step_name': 'video2audio',
        'output_directory': '../datasets/mp3',
        'audio_files': ['test.mp3', 'test2.mp3'],
        'workflow_id': 'upstream_001'
    }
    
    print(f"接收到上游数据: {upstream_output['step_name']}")
    print(f"音频文件: {upstream_output['audio_files']}")
    
    # 当前环节：语音转文字
    interface = WorkflowInterface()
    speech_result = interface.process_from_video_converter(
        audio_input=upstream_output['output_directory'],
        workflow_id=upstream_output['workflow_id']
    )
    
    print(f"语音转文字完成: {speech_result['status']}")
    
    # 准备传递给下游：PPT生成
    ppt_data = interface.prepare_for_ppt_generator(speech_result['result'])
    
    # 模拟下游接收数据
    downstream_input = {
        'step_name': 'text2ppt',
        'input_data': ppt_data,
        'workflow_id': speech_result['workflow_id']
    }
    
    print(f"传递给下游: {downstream_input['step_name']}")
    print(f"PPT页面数: {downstream_input['input_data']['total_slides']}")
    
    return speech_result, ppt_data


if __name__ == '__main__':
    # 运行所有示例
    try:
        # 单个文件示例
        single_result = example_single_file_workflow()
        
        # 批量处理示例
        batch_result = example_batch_workflow()
        
        # 完整集成示例
        integration_result, ppt_data = example_workflow_integration()
        
        # PPT数据格式示例
        format_result = example_ppt_data_format()
        
        # 上下游集成模拟
        upstream_result, downstream_data = simulate_upstream_downstream_integration()
        
        print("\n=== 所有示例执行完成 ===")
        print("工作流集成模块已准备就绪，可以与上下游环节对接")
        
    except Exception as e:
        print(f"示例执行出错: {e}")
        print("请检查音频文件路径和模型配置") 