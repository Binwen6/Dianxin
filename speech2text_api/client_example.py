#!/usr/bin/env python3
"""
Speech to Text API 客户端示例
演示如何使用API进行语音转文字
"""

import requests
import json
import sys
from pathlib import Path

def transcribe_audio(file_path: str, api_key: str, user_id: str = "default_user"):
    """
    使用API转录音频文件
    
    Args:
        file_path: 音频文件路径
        api_key: API密钥
        user_id: 用户标识
        
    Returns:
        转录结果字典
    """
    url = "http://localhost:8000/audio-to-text"
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    
    try:
        with open(file_path, "rb") as f:
            files = {"file": f}
            data = {"user": user_id}
            
            print(f"正在转录文件: {file_path}")
            response = requests.post(url, headers=headers, files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                print("✅ 转录成功!")
                print(f"📝 转录文本: {result['text']}")
                print(f"⏱️  处理时间: {result.get('processing_time', 'N/A')}秒")
                print(f"📊 文件大小: {result.get('file_size', 'N/A')} 字节")
                print(f"🤖 使用模型: {result.get('model_used', 'N/A')}")
                return result
            else:
                print(f"❌ 转录失败: {response.status_code}")
                print(f"错误信息: {response.text}")
                return None
                
    except Exception as e:
        print(f"❌ 请求失败: {e}")
        return None

def main():
    """主函数"""
    print("🎵 Speech to Text API 客户端示例")
    print("=" * 50)
    
    # 配置
    api_key = "your-secret-api-key-here"  # 请修改为你的实际API密钥
    
    # 查找测试音频文件
    project_root = Path(__file__).parent.parent
    test_files = []
    
    for ext in ['.mp3', '.m4a', '.wav']:
        test_files.extend(list(project_root.glob(f"datasets/{ext[1:]}/*{ext}")))
    
    if not test_files:
        print("❌ 未找到测试音频文件")
        print("请确保在 datasets/ 目录下有音频文件")
        return
    
    # 选择第一个测试文件
    test_file = test_files[0]
    print(f"📁 使用测试文件: {test_file}")
    
    # 执行转录
    result = transcribe_audio(str(test_file), api_key, "example_user")
    
    if result:
        # 保存结果到文件
        output_file = Path(__file__).parent / "transcription_result.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"💾 结果已保存到: {output_file}")

if __name__ == "__main__":
    main() 