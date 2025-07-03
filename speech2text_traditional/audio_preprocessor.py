import numpy as np
import librosa
import scipy.signal as signal
from typing import Tuple, List
import logging

class AudioPreprocessor:
    """音频预处理类，包含降噪、端点检测、分帧等功能"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.logger = logging.getLogger(__name__)
    
    def load_audio(self, file_path: str) -> np.ndarray:
        """加载音频文件"""
        try:
            audio, sr = librosa.load(file_path, sr=self.sample_rate)
            return audio
        except Exception as e:
            self.logger.error(f"音频加载失败: {e}")
            return np.array([])
    
    def preemphasis(self, audio: np.ndarray, coef: float = 0.97) -> np.ndarray:
        """预加重滤波"""
        return np.append(audio[0], audio[1:] - coef * audio[:-1])
    
    def spectral_subtraction(self, audio: np.ndarray, noise_frames: int = 10) -> np.ndarray:
        """谱减法降噪"""
        try:
            # 计算噪声谱
            noise_spectrum = np.mean(np.abs(np.fft.fft(audio[:noise_frames * 512])), axis=0)
            
            # 分帧处理
            frame_length = 512
            hop_length = 256
            frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
            
            # 应用谱减法
            processed_frames = []
            for frame in frames.T:
                spectrum = np.abs(np.fft.fft(frame))
                # 谱减法
                cleaned_spectrum = np.maximum(spectrum - 0.5 * noise_spectrum, 0.1 * spectrum)
                # 保持相位
                phase = np.angle(np.fft.fft(frame))
                cleaned_frame = np.real(np.fft.ifft(cleaned_spectrum * np.exp(1j * phase)))
                processed_frames.append(cleaned_frame)
            
            return np.concatenate(processed_frames)
        except Exception as e:
            self.logger.error(f"谱减法降噪失败: {e}")
            return audio
    
    def wiener_filter(self, audio: np.ndarray, noise_frames: int = 10) -> np.ndarray:
        """维纳滤波降噪"""
        try:
            # 估计噪声功率谱
            noise_power = np.mean(np.abs(np.fft.fft(audio[:noise_frames * 512])) ** 2)
            
            # 分帧处理
            frame_length = 512
            hop_length = 256
            frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
            
            processed_frames = []
            for frame in frames.T:
                spectrum = np.fft.fft(frame)
                signal_power = np.abs(spectrum) ** 2
                
                # 维纳滤波器
                wiener_gain = signal_power / (signal_power + noise_power)
                filtered_spectrum = spectrum * wiener_gain
                
                filtered_frame = np.real(np.fft.ifft(filtered_spectrum))
                processed_frames.append(filtered_frame)
            
            return np.concatenate(processed_frames)
        except Exception as e:
            self.logger.error(f"维纳滤波失败: {e}")
            return audio
    
    def voice_activity_detection(self, audio: np.ndarray, 
                                frame_length: int = 512, 
                                hop_length: int = 256,
                                threshold: float = 0.01) -> List[Tuple[int, int]]:
        """语音活动检测(VAD)"""
        try:
            # 计算短时能量
            frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
            energy = np.sum(frames ** 2, axis=0)
            
            # 计算过零率
            zero_crossings = np.sum(np.diff(np.sign(frames), axis=0) != 0, axis=0)
            
            # 组合特征
            combined_feature = energy * (1 + 0.1 * zero_crossings)
            
            # 自适应阈值
            threshold = np.mean(combined_feature) + 2 * np.std(combined_feature)
            
            # 检测语音段
            speech_segments = []
            in_speech = False
            start_frame = 0
            
            for i, feature in enumerate(combined_feature):
                if feature > threshold and not in_speech:
                    start_frame = i
                    in_speech = True
                elif feature <= threshold and in_speech:
                    end_frame = i
                    if end_frame - start_frame > 5:  # 最小语音段长度
                        start_sample = start_frame * hop_length
                        end_sample = end_frame * hop_length
                        speech_segments.append((start_sample, end_sample))
                    in_speech = False
            
            # 处理最后一个语音段
            if in_speech and len(combined_feature) - start_frame > 5:
                start_sample = start_frame * hop_length
                end_sample = len(audio)
                speech_segments.append((start_sample, end_sample))
            
            return speech_segments
        except Exception as e:
            self.logger.error(f"VAD检测失败: {e}")
            return [(0, len(audio))]
    
    def normalize_audio(self, audio: np.ndarray, target_db: float = -20) -> np.ndarray:
        """音频归一化"""
        try:
            # 计算RMS
            rms = np.sqrt(np.mean(audio ** 2))
            if rms > 0:
                # 计算目标RMS
                target_rms = 10 ** (target_db / 20)
                # 归一化
                normalized_audio = audio * (target_rms / rms)
                return normalized_audio
            return audio
        except Exception as e:
            self.logger.error(f"音频归一化失败: {e}")
            return audio
    
    def bandpass_filter(self, audio: np.ndarray, low_freq: float = 80, 
                       high_freq: float = 8000) -> np.ndarray:
        """带通滤波"""
        try:
            nyquist = self.sample_rate / 2
            low = low_freq / nyquist
            high = high_freq / nyquist
            
            # 设计巴特沃斯滤波器
            b, a = signal.butter(4, [low, high], btype='band')
            filtered_audio = signal.filtfilt(b, a, audio)
            
            return filtered_audio
        except Exception as e:
            self.logger.error(f"带通滤波失败: {e}")
            return audio
    
    def preprocess_audio(self, audio: np.ndarray, 
                        apply_vad: bool = True,
                        apply_noise_reduction: bool = True) -> np.ndarray:
        """完整的音频预处理流程"""
        try:
            # 1. 预加重
            audio = self.preemphasis(audio)
            
            # 2. 带通滤波
            audio = self.bandpass_filter(audio)
            
            # 3. 降噪
            if apply_noise_reduction:
                audio = self.spectral_subtraction(audio)
            
            # 4. 语音活动检测
            if apply_vad:
                speech_segments = self.voice_activity_detection(audio)
                if speech_segments:
                    # 只保留语音段
                    start, end = speech_segments[0]
                    audio = audio[start:end]
            
            # 5. 归一化
            audio = self.normalize_audio(audio)
            
            return audio
        except Exception as e:
            self.logger.error(f"音频预处理失败: {e}")
            return audio
    
    def extract_spectral_features(self, audio: np.ndarray) -> np.ndarray:
        """提取频谱特征"""
        try:
            # 短时傅里叶变换
            stft = librosa.stft(audio, n_fft=512, hop_length=256)
            
            # 功率谱
            power_spectrum = np.abs(stft) ** 2
            
            # 梅尔频谱
            mel_spectrum = librosa.feature.melspectrogram(y=audio, sr=self.sample_rate)
            
            # 频谱质心
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)
            
            # 频谱带宽
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate)
            
            # 频谱对比度
            spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=self.sample_rate)
            
            # 组合特征
            features = np.vstack([
                mel_spectrum,
                spectral_centroid,
                spectral_bandwidth,
                spectral_contrast
            ])
            
            return features
        except Exception as e:
            self.logger.error(f"频谱特征提取失败: {e}")
            return np.array([]) 