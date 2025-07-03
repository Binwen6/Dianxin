#!/usr/bin/env python3
"""
传统音频处理技术的语音转文本训练和识别脚本
"""

import os
import numpy as np
import librosa
import json
import logging
from typing import Dict, List, Tuple
from traditional_asr import TraditionalASR
from audio_preprocessor import AudioPreprocessor

class TraditionalASRSystem:
    """传统ASR系统主类"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.asr = TraditionalASR(sample_rate=sample_rate)
        self.preprocessor = AudioPreprocessor(sample_rate=sample_rate)
        
        # 配置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def train_from_directory(self, training_dir: str, labels_file: str = None) -> bool:
        """
        从目录训练模型
        
        Args:
            training_dir: 训练数据目录，包含按标签组织的音频文件
            labels_file: 标签映射文件（可选）
        """
        try:
            if not os.path.exists(training_dir):
                self.logger.error(f"训练目录不存在: {training_dir}")
                return False
            
            # 加载标签映射
            label_mapping = {}
            if labels_file and os.path.exists(labels_file):
                with open(labels_file, 'r', encoding='utf-8') as f:
                    label_mapping = json.load(f)
            
            # 遍历训练目录
            for label_dir in os.listdir(training_dir):
                label_path = os.path.join(training_dir, label_dir)
                if not os.path.isdir(label_path):
                    continue
                
                # 获取标签名称
                label = label_mapping.get(label_dir, label_dir)
                self.logger.info(f"训练标签: {label}")
                
                # 收集该标签的所有音频文件
                audio_files = []
                for file in os.listdir(label_path):
                    if file.lower().endswith(('.wav', '.mp3', '.m4a', '.flac')):
                        audio_files.append(os.path.join(label_path, file))
                
                if not audio_files:
                    self.logger.warning(f"标签 {label} 没有找到音频文件")
                    continue
                
                # 提取特征
                all_features = []
                for audio_file in audio_files:
                    try:
                        # 加载音频
                        audio = self.preprocessor.load_audio(audio_file)
                        if len(audio) == 0:
                            continue
                        
                        # 预处理
                        audio = self.preprocessor.preprocess_audio(audio)
                        
                        # 提取MFCC特征
                        features = self.asr.extract_mfcc_features(audio)
                        if len(features) > 0:
                            all_features.append(features)
                        
                    except Exception as e:
                        self.logger.error(f"处理音频文件失败 {audio_file}: {e}")
                        continue
                
                if all_features:
                    # 合并特征
                    combined_features = np.vstack(all_features)
                    
                    # 训练GMM模型
                    success = self.asr.train_gmm_model(combined_features, label)
                    if success:
                        self.logger.info(f"标签 {label} 训练成功，特征数量: {len(combined_features)}")
                    else:
                        self.logger.error(f"标签 {label} 训练失败")
            
            return len(self.asr.gmm_models) > 0
            
        except Exception as e:
            self.logger.error(f"训练过程失败: {e}")
            return False
    
    def recognize_audio_file(self, audio_file: str) -> str:
        """
        识别单个音频文件
        
        Args:
            audio_file: 音频文件路径
            
        Returns:
            识别结果文本
        """
        try:
            # 加载音频
            audio = self.preprocessor.load_audio(audio_file)
            if len(audio) == 0:
                return ""
            
            # 预处理
            audio = self.preprocessor.preprocess_audio(audio)
            
            # 识别
            result = self.asr.recognize_speech(audio)
            
            return result
            
        except Exception as e:
            self.logger.error(f"音频识别失败: {e}")
            return ""
    
    def batch_recognize(self, audio_dir: str, output_file: str = None) -> Dict[str, str]:
        """
        批量识别音频文件
        
        Args:
            audio_dir: 音频文件目录
            output_file: 输出结果文件（可选）
            
        Returns:
            识别结果字典 {文件名: 识别结果}
        """
        try:
            results = {}
            
            # 获取所有音频文件
            audio_files = []
            for file in os.listdir(audio_dir):
                if file.lower().endswith(('.wav', '.mp3', '.m4a', '.flac')):
                    audio_files.append(file)
            
            self.logger.info(f"找到 {len(audio_files)} 个音频文件")
            
            # 批量识别
            for i, audio_file in enumerate(audio_files):
                self.logger.info(f"处理文件 {i+1}/{len(audio_files)}: {audio_file}")
                
                file_path = os.path.join(audio_dir, audio_file)
                result = self.recognize_audio_file(file_path)
                results[audio_file] = result
            
            # 保存结果
            if output_file:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                self.logger.info(f"结果已保存到: {output_file}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"批量识别失败: {e}")
            return {}
    
    def save_model(self, model_path: str):
        """保存模型"""
        try:
            self.asr.save_model(model_path)
            self.logger.info(f"模型已保存到: {model_path}")
        except Exception as e:
            self.logger.error(f"模型保存失败: {e}")
    
    def load_model(self, model_path: str):
        """加载模型"""
        try:
            self.asr.load_model(model_path)
            self.logger.info(f"模型已从 {model_path} 加载")
        except Exception as e:
            self.logger.error(f"模型加载失败: {e}")
    
    def evaluate_model(self, test_dir: str, ground_truth_file: str = None) -> Dict:
        """
        评估模型性能
        
        Args:
            test_dir: 测试数据目录
            ground_truth_file: 真实标签文件（可选）
            
        Returns:
            评估结果
        """
        try:
            results = {}
            correct = 0
            total = 0
            
            # 加载真实标签
            ground_truth = {}
            if ground_truth_file and os.path.exists(ground_truth_file):
                with open(ground_truth_file, 'r', encoding='utf-8') as f:
                    ground_truth = json.load(f)
            
            # 测试每个文件
            for file in os.listdir(test_dir):
                if not file.lower().endswith(('.wav', '.mp3', '.m4a', '.flac')):
                    continue
                
                file_path = os.path.join(test_dir, file)
                predicted = self.recognize_audio_file(file_path)
                
                if file in ground_truth:
                    true_label = ground_truth[file]
                    is_correct = predicted.lower() == true_label.lower()
                    correct += int(is_correct)
                    total += 1
                    
                    results[file] = {
                        'predicted': predicted,
                        'true': true_label,
                        'correct': is_correct
                    }
                else:
                    results[file] = {
                        'predicted': predicted,
                        'true': 'unknown',
                        'correct': None
                    }
            
            # 计算准确率
            accuracy = correct / total if total > 0 else 0
            
            evaluation_result = {
                'accuracy': accuracy,
                'correct': correct,
                'total': total,
                'results': results
            }
            
            self.logger.info(f"评估结果 - 准确率: {accuracy:.2%} ({correct}/{total})")
            
            return evaluation_result
            
        except Exception as e:
            self.logger.error(f"模型评估失败: {e}")
            return {}

def main():
    """主函数 - 演示使用"""
    import argparse
    
    parser = argparse.ArgumentParser(description='传统ASR系统')
    parser.add_argument('--mode', choices=['train', 'recognize', 'batch', 'evaluate'], 
                       required=True, help='运行模式')
    parser.add_argument('--input', required=True, help='输入路径')
    parser.add_argument('--output', help='输出路径')
    parser.add_argument('--model', help='模型路径')
    parser.add_argument('--labels', help='标签映射文件')
    parser.add_argument('--ground_truth', help='真实标签文件（用于评估）')
    
    args = parser.parse_args()
    
    # 创建ASR系统
    asr_system = TraditionalASRSystem()
    
    if args.mode == 'train':
        # 训练模式
        if args.model:
            asr_system.load_model(args.model)
        
        success = asr_system.train_from_directory(args.input, args.labels)
        if success:
            if args.output:
                asr_system.save_model(args.output)
            print("训练完成")
        else:
            print("训练失败")
    
    elif args.mode == 'recognize':
        # 单文件识别模式
        if args.model:
            asr_system.load_model(args.model)
        
        result = asr_system.recognize_audio_file(args.input)
        print(f"识别结果: {result}")
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(result)
    
    elif args.mode == 'batch':
        # 批量识别模式
        if args.model:
            asr_system.load_model(args.model)
        
        results = asr_system.batch_recognize(args.input, args.output)
        print(f"批量识别完成，处理了 {len(results)} 个文件")
    
    elif args.mode == 'evaluate':
        # 评估模式
        if args.model:
            asr_system.load_model(args.model)
        
        evaluation = asr_system.evaluate_model(args.input, args.ground_truth)
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(evaluation, f, ensure_ascii=False, indent=2)
        print(f"评估完成，准确率: {evaluation.get('accuracy', 0):.2%}")

if __name__ == "__main__":
    main() 