import numpy as np
import librosa
import scipy.signal as signal
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import pickle
import os
from typing import List, Tuple, Dict, Optional
import logging

class TraditionalASR:
    """
    传统音频处理技术的语音转文本系统
    使用MFCC特征提取、DTW动态时间规整、HMM隐马尔可夫模型等技术
    """
    
    def __init__(self, sample_rate: int = 16000, n_mfcc: int = 13, 
                 frame_length: int = 2048, hop_length: int = 512):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.frame_length = frame_length
        self.hop_length = hop_length
        
        # 声学模型组件
        self.gmm_models = {}  # 高斯混合模型
        self.hmm_models = {}  # 隐马尔可夫模型
        self.scaler = StandardScaler()
        
        # 语言模型
        self.language_model = {}
        self.vocabulary = set()
        
        # 配置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def extract_mfcc_features(self, audio: np.ndarray) -> np.ndarray:
        """
        提取MFCC特征
        """
        try:
            # 预加重
            pre_emphasis = 0.97
            emphasized_audio = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])
            
            # 分帧
            frames = librosa.util.frame(emphasized_audio, 
                                      frame_length=self.frame_length, 
                                      hop_length=self.hop_length)
            
            # 加窗
            window = np.hanning(self.frame_length)
            frames = frames * window[:, np.newaxis]
            
            # 计算功率谱
            mag_frames = np.absolute(np.fft.rfft(frames, axis=0))
            pow_frames = (1.0 / self.frame_length) * (mag_frames ** 2)
            
            # 梅尔滤波器组
            mel_basis = librosa.filters.mel(sr=self.sample_rate, 
                                          n_fft=self.frame_length, 
                                          n_mels=26)
            mel_pow = np.dot(mel_basis, pow_frames)
            
            # 对数变换
            log_mel_pow = np.log(mel_pow + 1e-10)
            
            # DCT变换得到MFCC
            mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, 
                                       n_mfcc=self.n_mfcc, 
                                       n_fft=self.frame_length, 
                                       hop_length=self.hop_length)
            
            # 添加一阶和二阶差分
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
            
            # 组合特征
            features = np.vstack([mfcc, mfcc_delta, mfcc_delta2])
            
            return features.T  # 转置为 (时间帧, 特征维度)
            
        except Exception as e:
            self.logger.error(f"MFCC特征提取失败: {e}")
            return np.array([])
    
    def dtw_distance(self, seq1: np.ndarray, seq2: np.ndarray) -> float:
        """
        动态时间规整(DTW)距离计算
        """
        n, m = len(seq1), len(seq2)
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0
        
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = np.linalg.norm(seq1[i-1] - seq2[j-1])
                dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j],    # 插入
                                            dtw_matrix[i, j-1],    # 删除
                                            dtw_matrix[i-1, j-1])  # 匹配
        
        return dtw_matrix[n, m]
    
    def train_gmm_model(self, features: np.ndarray, label: str, 
                       n_components: int = 8) -> bool:
        """
        训练高斯混合模型
        """
        try:
            if len(features) < n_components:
                self.logger.warning(f"特征数量不足，无法训练GMM模型: {label}")
                return False
            
            # 标准化特征
            features_scaled = self.scaler.fit_transform(features)
            
            # 训练GMM
            gmm = GaussianMixture(n_components=n_components, 
                                 covariance_type='diag', 
                                 random_state=42)
            gmm.fit(features_scaled)
            
            self.gmm_models[label] = gmm
            self.logger.info(f"GMM模型训练完成: {label}")
            return True
            
        except Exception as e:
            self.logger.error(f"GMM模型训练失败: {e}")
            return False
    
    def simple_hmm(self, features: np.ndarray, n_states: int = 5) -> Dict:
        """
        简单的隐马尔可夫模型实现
        """
        try:
            n_frames, n_features = features.shape
            
            # 初始化参数
            pi = np.ones(n_states) / n_states  # 初始状态概率
            A = np.eye(n_states) * 0.8 + np.eye(n_states, k=1) * 0.2  # 转移矩阵
            A[-1, -1] = 1.0  # 最后一个状态自环
            
            # 计算每个状态的发射概率（使用GMM）
            B = np.zeros((n_states, n_frames))
            for i in range(n_states):
                # 简单的状态发射概率（基于特征距离）
                state_features = features[i * n_frames // n_states:(i + 1) * n_frames // n_states]
                if len(state_features) > 0:
                    mean_feature = np.mean(state_features, axis=0)
                    for j in range(n_frames):
                        B[i, j] = np.exp(-np.linalg.norm(features[j] - mean_feature))
            
            # 归一化发射概率
            B = B / (B.sum(axis=0, keepdims=True) + 1e-10)
            
            return {
                'pi': pi,
                'A': A,
                'B': B,
                'n_states': n_states
            }
            
        except Exception as e:
            self.logger.error(f"HMM模型创建失败: {e}")
            return {}
    
    def viterbi_decode(self, hmm_model: Dict, features: np.ndarray) -> List[int]:
        """
        Viterbi算法进行解码
        """
        try:
            pi = hmm_model['pi']
            A = hmm_model['A']
            B = hmm_model['B']
            n_states = hmm_model['n_states']
            n_frames = len(features)
            
            # 动态规划表
            delta = np.zeros((n_states, n_frames))
            psi = np.zeros((n_states, n_frames), dtype=int)
            
            # 初始化
            delta[:, 0] = np.log(pi + 1e-10) + np.log(B[:, 0] + 1e-10)
            
            # 前向递推
            for t in range(1, n_frames):
                for j in range(n_states):
                    temp = delta[:, t-1] + np.log(A[:, j] + 1e-10)
                    delta[j, t] = np.max(temp) + np.log(B[j, t] + 1e-10)
                    psi[j, t] = np.argmax(temp)
            
            # 回溯
            path = np.zeros(n_frames, dtype=int)
            path[-1] = np.argmax(delta[:, -1])
            
            for t in range(n_frames-2, -1, -1):
                path[t] = psi[path[t+1], t+1]
            
            return path.tolist()
            
        except Exception as e:
            self.logger.error(f"Viterbi解码失败: {e}")
            return []
    
    def build_language_model(self, text_corpus: List[str], n_gram: int = 2):
        """
        构建N-gram语言模型
        """
        try:
            for text in text_corpus:
                words = text.lower().split()
                self.vocabulary.update(words)
                
                # 构建N-gram
                for i in range(len(words) - n_gram + 1):
                    ngram = tuple(words[i:i+n_gram])
                    if ngram not in self.language_model:
                        self.language_model[ngram] = 0
                    self.language_model[ngram] += 1
            
            self.logger.info(f"语言模型构建完成，词汇量: {len(self.vocabulary)}")
            
        except Exception as e:
            self.logger.error(f"语言模型构建失败: {e}")
    
    def recognize_speech(self, audio: np.ndarray) -> str:
        """
        主要的语音识别函数
        """
        try:
            # 1. 特征提取
            features = self.extract_mfcc_features(audio)
            if len(features) == 0:
                return ""
            
            # 2. 声学模型匹配
            best_match = None
            best_score = float('inf')
            
            for label, gmm in self.gmm_models.items():
                # 计算似然度
                features_scaled = self.scaler.transform(features)
                score = -gmm.score(features_scaled)  # 负对数似然
                
                if score < best_score:
                    best_score = score
                    best_match = label
            
            # 3. HMM解码（如果有HMM模型）
            if best_match and best_match in self.hmm_models:
                hmm_model = self.hmm_models[best_match]
                state_sequence = self.viterbi_decode(hmm_model, features)
                
                # 基于状态序列进行后处理
                # 这里可以添加更复杂的后处理逻辑
            
            # 4. 语言模型校正
            if best_match:
                # 简单的语言模型校正
                corrected_text = self.apply_language_model_correction(best_match)
                return corrected_text
            
            return best_match if best_match else ""
            
        except Exception as e:
            self.logger.error(f"语音识别失败: {e}")
            return ""
    
    def apply_language_model_correction(self, text: str) -> str:
        """
        使用语言模型进行文本校正
        """
        try:
            words = text.lower().split()
            corrected_words = []
            
            for i, word in enumerate(words):
                # 检查当前词是否在词汇表中
                if word in self.vocabulary:
                    corrected_words.append(word)
                else:
                    # 寻找最相似的词
                    best_word = word
                    best_score = 0
                    
                    for vocab_word in self.vocabulary:
                        # 简单的编辑距离
                        score = self.edit_distance_similarity(word, vocab_word)
                        if score > best_score:
                            best_score = score
                            best_word = vocab_word
                    
                    corrected_words.append(best_word)
            
            return " ".join(corrected_words)
            
        except Exception as e:
            self.logger.error(f"语言模型校正失败: {e}")
            return text
    
    def edit_distance_similarity(self, word1: str, word2: str) -> float:
        """
        计算两个词的编辑距离相似度
        """
        try:
            m, n = len(word1), len(word2)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            
            for i in range(m + 1):
                dp[i][0] = i
            for j in range(n + 1):
                dp[0][j] = j
            
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if word1[i-1] == word2[j-1]:
                        dp[i][j] = dp[i-1][j-1]
                    else:
                        dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
            
            # 转换为相似度分数
            max_len = max(m, n)
            similarity = 1 - dp[m][n] / max_len
            return max(0, similarity)
            
        except Exception as e:
            return 0.0
    
    def save_model(self, filepath: str):
        """
        保存模型到文件
        """
        try:
            model_data = {
                'gmm_models': self.gmm_models,
                'hmm_models': self.hmm_models,
                'scaler': self.scaler,
                'language_model': self.language_model,
                'vocabulary': self.vocabulary,
                'sample_rate': self.sample_rate,
                'n_mfcc': self.n_mfcc,
                'frame_length': self.frame_length,
                'hop_length': self.hop_length
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            self.logger.info(f"模型已保存到: {filepath}")
            
        except Exception as e:
            self.logger.error(f"模型保存失败: {e}")
    
    def load_model(self, filepath: str):
        """
        从文件加载模型
        """
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.gmm_models = model_data['gmm_models']
            self.hmm_models = model_data['hmm_models']
            self.scaler = model_data['scaler']
            self.language_model = model_data['language_model']
            self.vocabulary = model_data['vocabulary']
            self.sample_rate = model_data['sample_rate']
            self.n_mfcc = model_data['n_mfcc']
            self.frame_length = model_data['frame_length']
            self.hop_length = model_data['hop_length']
            
            self.logger.info(f"模型已从 {filepath} 加载")
            
        except Exception as e:
            self.logger.error(f"模型加载失败: {e}") 