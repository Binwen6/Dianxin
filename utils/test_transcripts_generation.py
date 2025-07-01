#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试transcripts.json生成功能
验证speech2text和speech2text_api两个目录下的transcripts.json生成
"""

import json
import os
from pathlib import Path
import sys

def test_speech2text_api():
    """测试speech2text_api目录下的transcripts.json生成"""
    print("=== 测试 speech2text_api 目录 ===")
    
    try:
        # 导入speech2text_api模块
        sys.path.append('speech2text_api')
        from workflow_interface import quick_speech2text
        
        # 测试单个文件处理
        print("测试单个文件处理...")
        result = quick_speech2text(
            audio_input="../datasets/mp3/test2.mp3",
            output_dir="../output/test_speech2text_api",
            workflow_id="test_api_001"
        )
        
        # 检查transcripts.json是否生成
        transcripts_file = Path("../output/test_speech2text_api/transcripts.json")
        if transcripts_file.exists():
            print(f"✅ transcripts.json 生成成功: {transcripts_file}")
            
            # 读取并验证格式
            with open(transcripts_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 验证基本结构
            assert 'metadata' in data, "缺少metadata字段"
            assert 'transcripts' in data, "缺少transcripts字段"
            assert len(data['transcripts']) > 0, "transcripts数组为空"
            
            # 验证时间戳信息
            transcript = data['transcripts'][0]
            assert 'segments' in transcript, "缺少segments字段"
            assert len(transcript['segments']) > 0, "segments数组为空"
            
            print(f"✅ 格式验证通过")
            print(f"   文件数: {data['metadata']['total_files']}")
            print(f"   成功数: {data['metadata']['successful_transcriptions']}")
            print(f"   段落数: {len(transcript['segments'])}")
            
            # 显示时间戳示例
            print("   时间戳示例:")
            for i, segment in enumerate(transcript['segments'][:3]):
                print(f"     [{segment['start']:.2f}s - {segment['end']:.2f}s] {segment['text'][:30]}...")
            
        else:
            print("❌ transcripts.json 生成失败")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False
    
    return True

def test_speech2text():
    """测试speech2text目录下的transcripts.json生成"""
    print("\n=== 测试 speech2text 目录 ===")
    
    try:
        # 导入speech2text模块
        sys.path.append('speech2text')
        from whisper import WhisperTranscriber
        
        # 创建输出目录
        output_dir = Path("../output/test_speech2text")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化转录器（启用时间戳）
        print("初始化Whisper转录器...")
        transcriber = WhisperTranscriber(return_timestamps=True)
        
        # 测试单个文件处理
        print("测试单个文件处理...")
        result = transcriber.transcribe_file("../datasets/mp3/test2.mp3")
        
        # 保存结果
        results = [result]
        transcriber.save_results(results, str(output_dir))
        
        # 检查transcripts.json是否生成
        transcripts_file = output_dir / "transcripts.json"
        if transcripts_file.exists():
            print(f"✅ transcripts.json 生成成功: {transcripts_file}")
            
            # 读取并验证格式
            with open(transcripts_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 验证基本结构
            assert 'metadata' in data, "缺少metadata字段"
            assert 'transcripts' in data, "缺少transcripts字段"
            assert len(data['transcripts']) > 0, "transcripts数组为空"
            
            # 验证时间戳信息
            transcript = data['transcripts'][0]
            assert 'segments' in transcript, "缺少segments字段"
            
            print(f"✅ 格式验证通过")
            print(f"   文件数: {data['metadata']['total_files']}")
            print(f"   成功数: {data['metadata']['successful_transcriptions']}")
            print(f"   时间戳功能: {data['metadata']['return_timestamps']}")
            
            # 显示时间戳示例
            if transcript['segments']:
                print("   时间戳示例:")
                for i, segment in enumerate(transcript['segments'][:3]):
                    print(f"     [{segment['start']:.2f}s - {segment['end']:.2f}s] {segment['text'][:30]}...")
            else:
                print("   注意: 未生成时间戳信息（可能需要更长的音频文件）")
            
        else:
            print("❌ transcripts.json 生成失败")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False
    
    return True

def compare_formats():
    """比较两个目录生成的transcripts.json格式"""
    print("\n=== 格式比较 ===")
    
    try:
        # 读取两个文件
        api_file = Path("../output/test_speech2text_api/transcripts.json")
        whisper_file = Path("../output/test_speech2text/transcripts.json")
        
        if not api_file.exists() or not whisper_file.exists():
            print("❌ 无法比较，文件不存在")
            return
        
        with open(api_file, 'r', encoding='utf-8') as f:
            api_data = json.load(f)
        
        with open(whisper_file, 'r', encoding='utf-8') as f:
            whisper_data = json.load(f)
        
        print("✅ 两个文件都成功生成")
        print("\n格式对比:")
        print(f"  speech2text_api: {len(api_data['transcripts'])} 个转录")
        print(f"  speech2text: {len(whisper_data['transcripts'])} 个转录")
        
        # 比较字段
        api_fields = set(api_data['transcripts'][0].keys()) if api_data['transcripts'] else set()
        whisper_fields = set(whisper_data['transcripts'][0].keys()) if whisper_data['transcripts'] else set()
        
        print(f"\n字段对比:")
        print(f"  speech2text_api 独有: {api_fields - whisper_fields}")
        print(f"  speech2text 独有: {whisper_fields - api_fields}")
        print(f"  共同字段: {api_fields & whisper_fields}")
        
        # 检查时间戳字段
        api_has_timestamps = any('segments' in t and t['segments'] for t in api_data['transcripts'])
        whisper_has_timestamps = any('segments' in t and t['segments'] for t in whisper_data['transcripts'])
        
        print(f"\n时间戳支持:")
        print(f"  speech2text_api: {'✅' if api_has_timestamps else '❌'}")
        print(f"  speech2text: {'✅' if whisper_has_timestamps else '❌'}")
        
    except Exception as e:
        print(f"❌ 比较失败: {e}")

def main():
    """主测试函数"""
    print("开始测试 transcripts.json 生成功能...")
    
    # 创建输出目录
    Path("../output").mkdir(exist_ok=True)
    
    # 测试两个目录
    api_success = test_speech2text_api()
    whisper_success = test_speech2text()
    
    # 比较格式
    if api_success and whisper_success:
        compare_formats()
    
    # 总结
    print("\n=== 测试总结 ===")
    print(f"speech2text_api: {'✅ 通过' if api_success else '❌ 失败'}")
    print(f"speech2text: {'✅ 通过' if whisper_success else '❌ 失败'}")
    
    if api_success and whisper_success:
        print("\n🎉 所有测试通过！两个目录都成功生成 transcripts.json")
        print("📁 输出文件位置:")
        print("   - speech2text_api: ../output/test_speech2text_api/transcripts.json")
        print("   - speech2text: ../output/test_speech2text/transcripts.json")
    else:
        print("\n⚠️  部分测试失败，请检查错误信息")

if __name__ == "__main__":
    main() 