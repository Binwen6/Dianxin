# Dianxin - 音频处理工具集

这是一个功能完整的音频处理工具集，包含M4A批量转换和Whisper语音转录功能。

## 🎵 功能模块

### 1. M4A批量转换为MP3
- 批量转换M4A文件为MP3格式
- 多线程并发处理
- 支持递归搜索子目录

### 2. Whisper语音转录
- 基于OpenAI Whisper模型的语音转录
- 支持批量处理多种音频格式
- 结构化输出结果（JSON、文本、摘要）
- 命令行工具和Python API

---

# M4A批量转换为MP3脚本

这是一个功能完整的Python脚本，用于批量将M4A音频文件转换为MP3格式。

## 功能特性

- 🎵 批量转换M4A文件为MP3格式
- 📁 支持递归搜索子目录
- ⚡ 多线程并发转换，提高效率
- 📊 实时进度显示
- 🔧 可配置MP3质量
- 📝 详细的日志记录
- 🛡️ 完善的错误处理

## 系统要求

### 必需软件
- **Python 3.7+**
- **FFmpeg** - 用于音频转换

### Python依赖
- `tqdm` - 进度条显示

## 安装步骤

1. **安装FFmpeg**
   ```bash
   # macOS (使用Homebrew)
   brew install ffmpeg
   
   # Ubuntu/Debian
   sudo apt update
   sudo apt install ffmpeg
   
   # Windows
   # 下载并安装: https://ffmpeg.org/download.html
   ```

2. **安装Python依赖**
   ```bash
   pip install -r requirements.txt
   ```

## 使用方法

### 基本用法

```bash
# 转换datasets目录下的所有m4a文件
python utils/m4a_to_mp3.py datasets/

# 转换当前目录下的所有m4a文件
python utils/m4a_to_mp3.py .
```

### 高级用法

```bash
# 指定输出目录
python utils/m4a_to_mp3.py datasets/ -o output/

# 设置MP3质量 (128k, 192k, 320k)
python utils/m4a_to_mp3.py datasets/ -q 320k

# 使用8个并发线程
python utils/m4a_to_mp3.py datasets/ -w 8

# 显示详细日志
python utils/m4a_to_mp3.py datasets/ -v

# 组合使用
python utils/m4a_to_mp3.py datasets/ -o output/ -q 320k -w 8 -v
```

### 命令行参数

| 参数 | 简写 | 说明 | 默认值 |
|------|------|------|--------|
| `input_dir` | - | 包含m4a文件的输入目录 | 必需 |
| `--output` | `-o` | 输出目录 | 与输入目录相同 |
| `--quality` | `-q` | MP3质量 (128k, 192k, 320k) | 192k |
| `--workers` | `-w` | 最大并发工作线程数 | 4 |
| `--verbose` | `-v` | 显示详细日志 | False |

## 输出说明

### 文件输出
- 转换后的MP3文件会保持原始的目录结构
- 文件名保持不变，仅扩展名从`.m4a`变为`.mp3`

### 日志输出
- 控制台实时显示转换进度
- 详细日志保存到`m4a_conversion.log`文件
- 包含成功/失败统计信息

### 示例输出
```
2024-01-01 12:00:00 - INFO - FFmpeg 已安装
2024-01-01 12:00:00 - INFO - 找到 5 个m4a文件
2024-01-01 12:00:00 - INFO - 开始批量转换 5 个文件...
转换进度: 100%|██████████| 5/5 [00:30<00:00,  6.00s/file, 成功=5, 失败=0]
2024-01-01 12:00:30 - INFO - 转换完成! 成功: 5, 失败: 0

转换统计:
总文件数: 5
成功转换: 5
转换失败: 0
```

## 注意事项

1. **FFmpeg依赖**: 脚本需要FFmpeg才能工作，请确保已正确安装
2. **文件权限**: 确保对输入和输出目录有读写权限
3. **磁盘空间**: 转换过程需要足够的磁盘空间存储MP3文件
4. **并发限制**: 根据系统性能调整并发线程数，避免过度占用系统资源

## 故障排除

### 常见问题

1. **FFmpeg未找到**
   ```
   错误: FFmpeg 未安装或不在PATH中
   解决: 安装FFmpeg并确保在系统PATH中
   ```

2. **权限错误**
   ```
   错误: Permission denied
   解决: 检查文件和目录权限
   ```

3. **磁盘空间不足**
   ```
   错误: No space left on device
   解决: 清理磁盘空间或更改输出目录
   ```

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request来改进这个脚本！

---

# Whisper语音转录功能

## 功能特性

- 🎯 **批量处理**: 支持批量转录目录中的所有音频文件
- 📁 **结构化输出**: 将结果保存为JSON、文本和摘要文件
- 🔧 **多格式支持**: 支持 MP3, WAV, M4A, FLAC, OGG, AAC 等格式
- 📊 **详细日志**: 提供详细的处理日志和错误信息
- ⚡ **GPU加速**: 自动检测并使用GPU加速（如果可用）
- 🎛️ **灵活配置**: 支持自定义模型和参数

## 安装依赖

```bash
pip install torch transformers datasets
```

## 使用方法

### 命令行工具

```bash
# 转录单个文件
python speech2text/cli.py datasets/test.mp3 --single

# 批量转录目录
python speech2text/cli.py datasets/ -o output/

# 使用不同模型
python speech2text/cli.py datasets/ -m openai/whisper-base

# 查看帮助
python speech2text/cli.py --help
```

### Python API

```python
from speech2text.whisper import WhisperTranscriber

# 初始化转录器
transcriber = WhisperTranscriber()

# 批量转录
results = transcriber.transcribe_directory("datasets", "output")

# 单个文件转录
result = transcriber.transcribe_file("datasets/test.mp3")
print(result["transcription"])
```

## 输出文件

脚本会在 `output` 目录下生成以下文件：

- `transcription_results_YYYYMMDD_HHMMSS.json` - 详细JSON结果
- `transcription_text_YYYYMMDD_HHMMSS.txt` - 人类可读的文本文件
- `transcription_summary_YYYYMMDD_HHMMSS.json` - 处理统计摘要

## 可用模型

- `openai/whisper-tiny` - 最快，精度较低
- `openai/whisper-base` - 快速，精度中等
- `openai/whisper-small` - 平衡速度和精度
- `openai/whisper-medium` - 较高精度
- `openai/whisper-large-v3` - 最高精度
- `openai/whisper-large-v3-turbo` - 推荐模型（默认）

## 示例输出

### JSON结果文件
```json
[
  {
    "file_path": "datasets/test.mp3",
    "file_name": "test.mp3",
    "file_size_bytes": 276758,
    "transcription": "转录的文本内容",
    "confidence": 0.95,
    "timestamp": "2024-01-01T12:00:00",
    "model_used": "openai/whisper-large-v3-turbo",
    "device_used": "cuda:0"
  }
]
```

### 摘要文件
```json
{
  "total_files": 2,
  "successful_transcriptions": 2,
  "failed_transcriptions": 0,
  "timestamp": "2024-01-01T12:00:00",
  "model_used": "openai/whisper-large-v3-turbo",
  "device_used": "cuda:0"
}
```

## 注意事项

1. 首次运行时会下载模型文件，需要网络连接
2. GPU模式需要安装CUDA版本的PyTorch
3. 大模型需要更多内存和计算资源
4. 转录质量取决于音频质量和模型选择

更多详细信息请查看 `speech2text/README.md`。 