#!/usr/bin/env python3
"""
API测试脚本
用于测试语音转文字API的功能
"""

import requests
import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def test_api_health():
    """测试API健康检查"""
    try:
        response = requests.get("http://localhost:8000/")
        print(f"健康检查状态码: {response.status_code}")
        print(f"响应内容: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"健康检查失败: {e}")
        return False

def test_audio_to_text():
    """测试音频转文字功能"""
    # 查找测试音频文件
    test_files = []
    for ext in ['.mp3', '.m4a', '.wav']:
        test_files.extend(list(project_root.glob(f"datasets/{ext[1:]}/*{ext}")))
    
    if not test_files:
        print("未找到测试音频文件")
        return False
    
    test_file = test_files[0]
    print(f"使用测试文件: {test_file}")
    
    url = "http://localhost:8000/audio-to-text"
    headers = {
        "Authorization": "Bearer your-secret-api-key-here"
    }
    
    try:
        with open(test_file, "rb") as f:
            files = {"file": f}
            data = {"user": "test_user"}
            
            print("发送请求...")
            response = requests.post(url, headers=headers, files=files, data=data)
            
            print(f"状态码: {response.status_code}")
            print(f"响应头: {dict(response.headers)}")
            
            if response.status_code == 200:
                result = response.json()
                print("转录成功!")
                print(f"转录文本: {result['text']}")
                print(f"处理时间: {result.get('processing_time', 'N/A')}秒")
                return True
            else:
                print(f"请求失败: {response.text}")
                return False
                
    except Exception as e:
        print(f"测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=== Speech to Text API 测试 ===")
    
    # 测试健康检查
    print("\n1. 测试健康检查...")
    if not test_api_health():
        print("健康检查失败，请确保API服务正在运行")
        return
    
    # 测试音频转文字
    print("\n2. 测试音频转文字...")
    if test_audio_to_text():
        print("\n✅ 所有测试通过!")
    else:
        print("\n❌ 测试失败")

if __name__ == "__main__":
    main() 