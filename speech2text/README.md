# Whisper 语音转录器

这是一个基于 OpenAI Whisper 模型的语音转录工具，支持批量处理音频文件并将结果结构化保存。**新增时间戳功能，支持词级别和段落级别的时间戳输出。**

## 功能特性

- 🎯 **批量处理**: 支持批量转录目录中的所有音频文件
- 📁 **结构化输出**: 将结果保存为JSON、文本和摘要文件
- 🔧 **多格式支持**: 支持 MP3, WAV, M4A, FLAC, OGG, AAC 等格式
- 📊 **详细日志**: 提供详细的处理日志和错误信息
- ⚡ **GPU加速**: 自动检测并使用GPU加速（如果可用）
- 🎛️ **灵活配置**: 支持自定义模型和参数
- ⏰ **时间戳支持**: 支持词级别和段落级别的时间戳输出
- 📺 **字幕生成**: 自动生成SRT格式字幕文件

## 安装依赖

```bash
pip install torch transformers datasets
```

## 使用方法

### 基本使用

```python
from whisper import WhisperTranscriber

# 初始化转录器（不启用时间戳）
transcriber = WhisperTranscriber()

# 批量转录datasets目录中的音频文件
results = transcriber.transcribe_directory("datasets", "output")
```

### 带时间戳的转录

```python
# 初始化转录器（启用时间戳）
transcriber = WhisperTranscriber(return_timestamps=True)

# 转录单个文件
result = transcriber.transcribe_file("datasets/test.mp3")
print(result["transcription"])

# 访问时间戳信息
if 'segment_timestamps' in result:
    for segment in result['segment_timestamps']:
        print(f"[{segment['start']:.2f}s - {segment['end']:.2f}s] {segment['text']}")

if 'word_timestamps' in result:
    for word_info in result['word_timestamps']:
        print(f"[{word_info['start']:.2f}s - {word_info['end']:.2f}s] {word_info['word']}")
```

### 使用不同模型

```python
# 使用更小的模型（速度更快）
transcriber = WhisperTranscriber("openai/whisper-base", return_timestamps=True)

# 使用更大的模型（精度更高）
transcriber = WhisperTranscriber("openai/whisper-large-v3", return_timestamps=True)
```

## 输出文件

脚本会在 `output` 目录下生成以下文件：

### 1. JSON结果文件 (`transcription_results_YYYYMMDD_HHMMSS.json`)
包含所有转录结果的详细JSON数据：

```json
[
  {
    "file_path": "datasets/test.mp3",
    "file_name": "test.mp3",
    "file_size_bytes": 270000,
    "transcription": "转录的文本内容",
    "confidence": 0.95,
    "timestamp": "2024-01-01T12:00:00",
    "model_used": "openai/whisper-large-v3-turbo",
    "device_used": "cuda:0",
    "return_timestamps": true,
    "segment_timestamps": [
      {
        "text": "这是第一段话",
        "start": 0.0,
        "end": 2.5,
        "duration": 2.5
      }
    ],
    "word_timestamps": [
      {
        "word": "这是",
        "start": 0.0,
        "end": 0.8
      }
    ]
  }
]
```

### 2. 文本文件 (`transcription_text_YYYYMMDD_HHMMSS.txt`)
人类可读的转录结果文本文件，包含时间戳信息：

```
转录结果 - 2024-01-01 12:00:00
==================================================

文件 1: test.mp3
路径: datasets/test.mp3
时间: 2024-01-01T12:00:00
转录内容:
他用吧就是要抓好但是经济的就是我们未来可能要去做本地部署对面现在我这要是他也不说那需要需要网络他

时间戳信息:
[0.00s - 2.50s] 他用吧就是要抓好但是经济的就是我们未来可能要去做本地部署对面现在我这要是他也不说那需要需要网络他

------------------------------
```

### 3. SRT字幕文件 (`transcription_subtitles_YYYYMMDD_HHMMSS.srt`)
标准字幕格式，可用于视频播放器：

```
# test.mp3
1
00:00:00,000 --> 00:00:02,500
他用吧就是要抓好但是经济的就是我们未来可能要去做本地部署对面现在我这要是他也不说那需要需要网络他

```

### 4. 摘要文件 (`transcription_summary_YYYYMMDD_HHMMSS.json`)
处理统计信息：

```json
{
  "total_files": 2,
  "successful_transcriptions": 2,
  "failed_transcriptions": 0,
  "timestamp": "2024-01-01T12:00:00",
  "model_used": "openai/whisper-large-v3-turbo",
  "device_used": "cuda:0",
  "return_timestamps": true
}
```

## 时间戳功能详解

### 段落级别时间戳
- 每个音频段落的时间范围
- 包含开始时间、结束时间和持续时间
- 适用于字幕生成和音频分段分析

### 词级别时间戳
- 每个词的精确时间范围
- 通过平均分配段落时间计算得出
- 适用于精确的音频文本对齐

### 时间戳格式
- **秒格式**: 浮点数，精确到小数点后2位
- **SRT格式**: HH:MM:SS,mmm 格式，用于字幕文件

## 支持的音频格式

- MP3 (.mp3)
- WAV (.wav)
- M4A (.m4a)
- FLAC (.flac)
- OGG (.ogg)
- AAC (.aac)

## 可用模型

- `openai/whisper-tiny` - 最快，精度较低
- `openai/whisper-base` - 快速，精度中等
- `openai/whisper-small` - 平衡速度和精度
- `openai/whisper-medium` - 较高精度
- `openai/whisper-large-v3` - 最高精度
- `openai/whisper-large-v3-turbo` - 推荐模型（默认）

## 运行示例

```bash
# 运行主脚本（启用时间戳）
python speech2text/whisper.py

# 运行示例脚本
python speech2text/example_usage.py
```

## 高级用法

### 自定义音频分块长度
```python
# 对于长音频文件，可以调整分块长度
result = transcriber.transcribe_file("long_audio.mp3", chunk_length_s=60)
```

### 手动生成SRT字幕
```python
# 获取时间戳信息后手动生成字幕
if 'segment_timestamps' in result:
    for segment in result['segment_timestamps']:
        start_time = transcriber._seconds_to_srt_time(segment['start'])
        end_time = transcriber._seconds_to_srt_time(segment['end'])
        print(f"{start_time} --> {end_time}")
        print(f"{segment['text']}\n")
```

## 错误处理

脚本包含完善的错误处理机制：

- 自动跳过不支持的音频格式
- 记录处理失败的详细信息
- 继续处理其他文件即使某个文件失败
- 提供详细的错误日志

## 性能优化

- 自动检测并使用GPU加速
- 支持半精度浮点数计算（GPU模式）
- 低内存使用模式
- 批量处理减少模型加载时间
- 音频分块处理长文件

## 注意事项

1. 首次运行时会下载模型文件，需要网络连接
2. GPU模式需要安装CUDA版本的PyTorch
3. 大模型需要更多内存和计算资源
4. 转录质量取决于音频质量和模型选择
5. 时间戳功能会增加处理时间，但提供更丰富的输出信息
6. 词级别时间戳是通过平均分配计算的，可能不够精确
7. SRT字幕文件符合标准格式，可在大多数视频播放器中使用

## 应用场景

- **字幕生成**: 自动为视频生成字幕文件
- **音频分析**: 分析音频中的语音内容和时间分布
- **会议记录**: 记录会议内容并标记时间点
- **语音搜索**: 在音频中搜索特定内容的时间位置
- **教育应用**: 为教学视频添加字幕和索引 