#!/usr/bin/env python3
"""
传统ASR系统使用示例
"""

import os
import numpy as np
import librosa
from traditional_asr import TraditionalASR
from audio_preprocessor import AudioPreprocessor
from train_and_recognize import TraditionalASRSystem

def create_sample_audio():
    """创建示例音频数据（用于演示）"""
    print("创建示例音频数据...")
    
    # 创建示例目录结构
    os.makedirs("sample_data/train/hello", exist_ok=True)
    os.makedirs("sample_data/train/world", exist_ok=True)
    os.makedirs("sample_data/test", exist_ok=True)
    
    # 生成简单的正弦波作为示例音频
    sample_rate = 16000
    duration = 1.0  # 1秒
    
    # 生成不同频率的音频作为不同标签
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # "hello" 标签 - 使用较低频率
    hello_audio = 0.3 * np.sin(2 * np.pi * 440 * t)  # A4音符
    librosa.output.write_wav("sample_data/train/hello/sample1.wav", hello_audio, sample_rate)
    librosa.output.write_wav("sample_data/train/hello/sample2.wav", hello_audio * 0.8, sample_rate)
    
    # "world" 标签 - 使用较高频率
    world_audio = 0.3 * np.sin(2 * np.pi * 880 * t)  # A5音符
    librosa.output.write_wav("sample_data/train/world/sample1.wav", world_audio, sample_rate)
    librosa.output.write_wav("sample_data/train/world/sample2.wav", world_audio * 0.9, sample_rate)
    
    # 测试音频
    test_audio = 0.3 * np.sin(2 * np.pi * 440 * t)  # 类似hello
    librosa.output.write_wav("sample_data/test/test1.wav", test_audio, sample_rate)
    
    print("示例音频数据创建完成")

def demo_basic_usage():
    """演示基本使用"""
    print("\n=== 基本使用演示 ===")
    
    # 创建ASR系统
    asr_system = TraditionalASRSystem()
    
    # 训练模型
    print("训练模型...")
    success = asr_system.train_from_directory("sample_data/train")
    if success:
        print("模型训练成功！")
        
        # 保存模型
        asr_system.save_model("trained_model.pkl")
        print("模型已保存到 trained_model.pkl")
        
        # 测试识别
        print("测试识别...")
        result = asr_system.recognize_audio_file("sample_data/test/test1.wav")
        print(f"识别结果: {result}")
    else:
        print("模型训练失败")

def demo_audio_preprocessing():
    """演示音频预处理功能"""
    print("\n=== 音频预处理演示 ===")
    
    preprocessor = AudioPreprocessor()
    
    # 加载音频
    audio = preprocessor.load_audio("sample_data/train/hello/sample1.wav")
    if len(audio) > 0:
        print(f"原始音频长度: {len(audio)} 采样点")
        
        # 预处理
        processed_audio = preprocessor.preprocess_audio(audio)
        print(f"预处理后音频长度: {len(processed_audio)} 采样点")
        
        # 提取频谱特征
        spectral_features = preprocessor.extract_spectral_features(processed_audio)
        print(f"频谱特征形状: {spectral_features.shape}")
        
        # VAD检测
        speech_segments = preprocessor.voice_activity_detection(processed_audio)
        print(f"检测到 {len(speech_segments)} 个语音段")

def demo_feature_extraction():
    """演示特征提取"""
    print("\n=== 特征提取演示 ===")
    
    asr = TraditionalASR()
    preprocessor = AudioPreprocessor()
    
    # 加载音频
    audio = preprocessor.load_audio("sample_data/train/hello/sample1.wav")
    if len(audio) > 0:
        # 提取MFCC特征
        mfcc_features = asr.extract_mfcc_features(audio)
        print(f"MFCC特征形状: {mfcc_features.shape}")
        print(f"MFCC特征维度: {mfcc_features.shape[1]}")
        
        # 计算DTW距离（与自身比较）
        distance = asr.dtw_distance(mfcc_features, mfcc_features)
        print(f"DTW距离（自比较）: {distance}")

def demo_model_persistence():
    """演示模型持久化"""
    print("\n=== 模型持久化演示 ===")
    
    # 创建并训练模型
    asr_system = TraditionalASRSystem()
    success = asr_system.train_from_directory("sample_data/train")
    
    if success:
        # 保存模型
        asr_system.save_model("demo_model.pkl")
        print("模型已保存")
        
        # 创建新的ASR系统并加载模型
        new_asr_system = TraditionalASRSystem()
        new_asr_system.load_model("demo_model.pkl")
        print("模型已加载")
        
        # 测试加载的模型
        result = new_asr_system.recognize_audio_file("sample_data/test/test1.wav")
        print(f"使用加载模型识别结果: {result}")

def demo_batch_processing():
    """演示批量处理"""
    print("\n=== 批量处理演示 ===")
    
    asr_system = TraditionalASRSystem()
    
    # 加载已训练的模型
    if os.path.exists("trained_model.pkl"):
        asr_system.load_model("trained_model.pkl")
        
        # 批量识别
        results = asr_system.batch_recognize("sample_data/test", "batch_results.json")
        print(f"批量识别完成，结果已保存到 batch_results.json")
        print("识别结果:")
        for file, result in results.items():
            print(f"  {file}: {result}")
    else:
        print("请先训练模型")

def cleanup():
    """清理临时文件"""
    print("\n=== 清理临时文件 ===")
    
    files_to_remove = [
        "trained_model.pkl",
        "demo_model.pkl", 
        "batch_results.json"
    ]
    
    for file in files_to_remove:
        if os.path.exists(file):
            os.remove(file)
            print(f"已删除: {file}")
    
    # 删除示例数据目录
    if os.path.exists("sample_data"):
        import shutil
        shutil.rmtree("sample_data")
        print("已删除示例数据目录")

def main():
    """主函数"""
    print("传统ASR系统使用示例")
    print("=" * 50)
    
    try:
        # 创建示例数据
        create_sample_audio()
        
        # 演示各种功能
        demo_basic_usage()
        demo_audio_preprocessing()
        demo_feature_extraction()
        demo_model_persistence()
        demo_batch_processing()
        
        print("\n所有演示完成！")
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
    
    finally:
        # 询问是否清理
        response = input("\n是否清理临时文件？(y/n): ")
        if response.lower() == 'y':
            cleanup()

if __name__ == "__main__":
    main() 