# 传统音频处理技术语音转文本系统

这是一个基于传统音频处理技术的语音转文本系统，不使用深度学习模型，而是采用经典的信号处理和模式识别方法。

## 技术特点

### 核心算法
- **MFCC特征提取** - 梅尔频率倒谱系数，用于音频特征表示
- **DTW动态时间规整** - 用于时间序列模式匹配
- **GMM高斯混合模型** - 用于声学建模
- **HMM隐马尔可夫模型** - 用于序列建模
- **VAD语音活动检测** - 自动检测语音段
- **谱减法降噪** - 传统降噪技术
- **维纳滤波** - 自适应降噪

### 音频预处理
- 预加重滤波
- 带通滤波（80Hz-8kHz）
- 音频归一化
- 语音活动检测
- 多种降噪算法

## 文件结构

```
speech2text_audio/
├── traditional_asr.py          # 核心ASR类
├── audio_preprocessor.py       # 音频预处理模块
├── train_and_recognize.py      # 训练和识别主脚本
├── example_usage.py           # 使用示例
├── requirements.txt           # 依赖包列表
└── README.md                 # 说明文档
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 基本使用

```python
from train_and_recognize import TraditionalASRSystem

# 创建ASR系统
asr_system = TraditionalASRSystem()

# 训练模型
success = asr_system.train_from_directory("training_data")

# 识别音频
result = asr_system.recognize_audio_file("audio.wav")
print(f"识别结果: {result}")
```

### 2. 命令行使用

#### 训练模型
```bash
python train_and_recognize.py --mode train --input training_data --output model.pkl
```

#### 单文件识别
```bash
python train_and_recognize.py --mode recognize --input audio.wav --model model.pkl
```

#### 批量识别
```bash
python train_and_recognize.py --mode batch --input audio_directory --model model.pkl --output results.json
```

#### 模型评估
```bash
python train_and_recognize.py --mode evaluate --input test_data --model model.pkl --ground_truth labels.json
```

### 3. 训练数据格式

训练数据应按以下目录结构组织：

```
training_data/
├── label1/
│   ├── audio1.wav
│   ├── audio2.wav
│   └── ...
├── label2/
│   ├── audio1.wav
│   ├── audio2.wav
│   └── ...
└── ...
```

每个子目录名即为标签名，包含该标签的音频文件。

### 4. 运行示例

```bash
# 运行完整示例
python example_usage.py
```

## 核心功能

### TraditionalASR类

主要方法：
- `extract_mfcc_features()` - 提取MFCC特征
- `dtw_distance()` - 计算DTW距离
- `train_gmm_model()` - 训练GMM模型
- `recognize_speech()` - 语音识别
- `save_model()` / `load_model()` - 模型持久化

### AudioPreprocessor类

主要方法：
- `load_audio()` - 加载音频文件
- `preprocess_audio()` - 完整预处理流程
- `voice_activity_detection()` - 语音活动检测
- `spectral_subtraction()` - 谱减法降噪
- `wiener_filter()` - 维纳滤波降噪
- `extract_spectral_features()` - 提取频谱特征

### TraditionalASRSystem类

主要方法：
- `train_from_directory()` - 从目录训练模型
- `recognize_audio_file()` - 单文件识别
- `batch_recognize()` - 批量识别
- `evaluate_model()` - 模型评估

## 技术原理

### 1. MFCC特征提取
1. 预加重滤波
2. 分帧和加窗
3. 短时傅里叶变换
4. 梅尔滤波器组
5. 对数变换
6. 离散余弦变换

### 2. DTW动态时间规整
- 解决时间序列长度不一致问题
- 计算两个序列的最优对齐距离
- 用于模式匹配和相似度计算

### 3. GMM高斯混合模型
- 建模声学特征的分布
- 每个音素/单词对应一个GMM
- 通过最大似然估计训练参数

### 4. 语音活动检测
- 基于短时能量和过零率
- 自适应阈值检测
- 自动分割语音段

### 5. 降噪技术
- **谱减法**: 估计噪声谱并减去
- **维纳滤波**: 基于信噪比的自适应滤波
- **带通滤波**: 去除人耳听不到的频率成分

## 性能特点

### 优势
- 不依赖大量训练数据
- 计算资源需求低
- 可解释性强
- 适合特定领域应用
- 训练速度快

### 局限性
- 识别准确率相对较低
- 对噪声敏感
- 需要手工设计特征
- 泛化能力有限

## 适用场景

- 特定词汇识别
- 命令词识别
- 有限词汇量应用
- 资源受限环境
- 教学和研究用途

## 扩展功能

### 可添加的功能
1. **N-gram语言模型** - 提高文本预测准确性
2. **声学模型优化** - 改进GMM/HMM参数
3. **特征工程** - 添加更多音频特征
4. **后处理** - 文本校正和优化
5. **实时处理** - 流式音频识别

### 性能优化
1. **并行处理** - 多线程/多进程
2. **缓存机制** - 避免重复计算
3. **模型压缩** - 减少存储空间
4. **GPU加速** - 利用GPU计算

## 注意事项

1. 音频格式支持：WAV, MP3, M4A, FLAC
2. 采样率：默认16kHz，可调整
3. 训练数据：每个标签至少需要几个样本
4. 内存使用：大文件可能需要较多内存
5. 计算时间：特征提取和训练可能需要一些时间

## 故障排除

### 常见问题
1. **音频加载失败** - 检查文件格式和路径
2. **训练失败** - 确保有足够的训练数据
3. **识别结果为空** - 检查音频质量和预处理
4. **内存不足** - 减少批处理大小或音频长度

### 调试建议
1. 启用详细日志输出
2. 检查音频文件完整性
3. 验证特征提取结果
4. 测试单个组件功能

## 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 贡献

欢迎提交Issue和Pull Request来改进这个项目。

## 联系方式

如有问题或建议，请通过GitHub Issues联系。 