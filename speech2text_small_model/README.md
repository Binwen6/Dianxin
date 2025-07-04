# SenseVoice 工作流语音转文字模块

这是一个专门为工作流设计的语音转文字模块，作为视频转音频 -> 语音转文字 -> 文字转PPT 工作流的中间环节。

## 功能特点

- 🎯 **工作流集成**: 专门设计用于与上下游环节对接
- 📁 **批量处理**: 支持目录级音频文件批量转写
- 📊 **标准化输出**: 提供标准化的JSON和TXT格式输出
- 🕐 **时间戳支持**: 生成包含时间戳的transcripts.json格式
- 🎨 **PPT就绪**: 输出格式专门为PPT生成优化
- 📝 **详细日志**: 完整的处理日志和状态追踪
- ⚙️ **灵活配置**: 支持多种配置参数和自定义设置

## 文件结构

```
speech2text_api/
├── workflow_processor.py      # 核心工作流处理器
├── workflow_interface.py      # 工作流集成接口
├── workflow_example.py        # 使用示例
├── batch_processor.py         # 批处理脚本（通用版本）
├── SenseVoice.py             # 原始脚本
├── config.json               # 配置文件模板
├── transcripts_format_example.json  # transcripts.json格式示例
└── README.md                 # 本文档
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 基本使用

#### 单个文件处理
```python
from workflow_interface import quick_speech2text

# 处理单个音频文件
ppt_data = quick_speech2text(
    audio_input="path/to/audio.mp3",
    output_dir="output",
    workflow_id="workflow_001"
)

print(f"生成了 {ppt_data['total_slides']} 个PPT页面")
```

#### 批量处理
```python
from workflow_interface import batch_speech2text_for_ppt

# 批量处理音频目录
ppt_data = batch_speech2text_for_ppt(
    audio_dir="path/to/audio/directory",
    output_dir="output"
)

print(f"批量处理完成，总字数: {ppt_data['metadata']['total_words']}")
```

### 3. 工作流集成

```python
from workflow_interface import WorkflowInterface

# 创建工作流接口
interface = WorkflowInterface()

# 从上游接收音频数据
result = interface.process_from_video_converter(
    audio_input="audio_directory",
    workflow_id="workflow_001"
)

# 为下游PPT生成准备数据
ppt_data = interface.prepare_for_ppt_generator(result['result'])

# 传递给下游环节
# downstream_module.process(ppt_data)
```

## 输出格式

### 1. transcripts.json（主要输出格式）

这是模块的主要输出格式，包含完整的时间戳信息：

```json
{
  "metadata": {
    "total_files": 2,
    "successful_transcriptions": 2,
    "failed_transcriptions": 0,
    "total_words": 150,
    "total_characters": 300,
    "total_audio_duration": 45.2,
    "total_processing_time": 5.5,
    "average_processing_time": 2.75,
    "timestamp": "2024-01-15T10:30:45.123456",
    "model_used": "iic/SenseVoiceSmall",
    "device_used": "cuda:0",
    "language": "auto",
    "workflow_step": "speech2text"
  },
  "transcripts": [
    {
      "file_name": "audio1.mp3",
      "file_path": "/path/to/audio1.mp3",
      "file_size_bytes": 10210000,
      "timestamp": "2024-01-15T10:30:42.123456",
      "model_used": "iic/SenseVoiceSmall",
      "device_used": "cuda:0",
      "language": "auto",
      "status": "success",
      "transcription": "这是音频文件的转录内容。",
      "processing_time": 2.5,
      "word_count": 25,
      "character_count": 50,
      "audio_duration": 20.5,
      "segments": [
        {
          "text": "这是音频文件的转录内容。",
          "start": 0.0,
          "end": 8.2,
          "duration": 8.2
        }
      ],
      "error": null
    }
  ]
}
```

### 2. 详细结果 (JSON)
```json
{
  "workflow_id": "workflow_001",
  "step_name": "speech2text",
  "status": "completed",
  "processed_files": [
    {
      "file_name": "audio1.mp3",
      "transcription": "转写的文本内容...",
      "processing_time": 2.5,
      "metadata": {
        "word_count": 150,
        "character_count": 300,
        "audio_duration": 45.2
      }
    }
  ],
  "summary": {
    "total_words": 150,
    "total_characters": 300,
    "total_audio_duration": 45.2
  }
}
```

### 3. PPT就绪格式
```json
{
  "workflow_step": "speech2text",
  "content_type": "transcription",
  "total_slides": 1,
  "slides": [
    {
      "slide_number": 1,
      "title": "音频转写: audio1.mp3",
      "content": "转写的文本内容...",
      "word_count": 150,
      "suggested_layout": "content_with_bullets"
    }
  ],
  "metadata": {
    "total_words": 150,
    "processing_time": 2.5
  }
}
```

## 时间戳功能

### SenseVoice API (speech2text_api)
- 基于音频时长和文本长度估算时间戳
- 按句子分割并分配时间
- 支持中文、英文等多种语言的句子分割

### Whisper (speech2text)
- 使用Whisper原生的时间戳功能
- 提供精确的词级别和段落级别时间戳
- 支持SRT字幕文件生成

## 配置参数

### 模型配置
```json
{
  "model_dir": "iic/SenseVoiceSmall",
  "device": "cuda:0",
  "language": "auto",
  "batch_size_s": 60,
  "merge_vad": true,
  "merge_length_s": 15
}
```

### 工作流配置
```json
{
  "output_format": "json",
  "include_timestamps": true,
  "segment_by_speaker": false,
  "min_segment_length": 1.0,
  "max_segment_length": 30.0
}
```

## 命令行使用

### 基本命令
```bash
# 处理单个文件
python workflow_processor.py audio.mp3 -o output

# 处理目录
python workflow_processor.py audio_directory -o output

# 输出PPT格式数据
python workflow_interface.py audio.mp3 --ppt-format

# 默认启用并行处理
python workflow_processor.py audio_directory -o output

# 禁用并行处理
python workflow_processor.py audio_directory -o output --no-parallel

# 指定工作进程数
python workflow_processor.py audio_directory -o output --max-workers 4

# 启用通义千问内容纠错
python workflow_processor.py input_audio.mp3 --enable-correction --dashscope-api-key YOUR_API_KEY

```

### 参数说明
- `input`: 输入音频文件或目录
- `-o, --output`: 输出目录
- `--device`: 设备 (cuda:0, cpu)
- `--language`: 语言 (auto, zn, en, yue, ja, ko)
- `--workflow-id`: 工作流ID
- `--ppt-format`: 输出PPT格式数据

## 输出文件说明

处理完成后，会在输出目录生成以下文件：

1. **transcripts.json** - 主要输出文件，包含时间戳信息
2. **speech2text_results_*.json** - 详细处理结果
3. **speech2text_content_*.txt** - 纯文本内容
4. **workflow_metadata_*.json** - 工作流元数据
5. **processing_stats_*.json** - 处理统计信息

## 与上下游环节集成

### 上游集成（视频转音频）
```python
# 假设从视频转音频环节接收数据
upstream_output = {
    'step_name': 'video2audio',
    'output_directory': 'audio_files',
    'audio_files': ['video1.mp3', 'video2.mp3'],
    'workflow_id': 'workflow_001'
}

# 处理音频
interface = WorkflowInterface()
result = interface.process_from_video_converter(
    audio_input=upstream_output['output_directory'],
    workflow_id=upstream_output['workflow_id']
)
```

### 下游集成（文字转PPT）
```python
# 准备PPT数据
ppt_data = interface.prepare_for_ppt_generator(result['result'])

# 传递给PPT生成环节
downstream_input = {
    'step_name': 'text2ppt',
    'input_data': ppt_data,
    'workflow_id': result['workflow_id']
}

# ppt_generator.process(downstream_input)
```

## 错误处理

模块包含完整的错误处理机制：

- 文件不存在或格式不支持
- 模型加载失败
- 处理过程中的异常
- 输出保存失败

所有错误都会记录在日志中，并在结果中标记状态。

## 性能优化

- 模型单例模式，避免重复加载
- 批量处理优化
- 内存使用优化
- 处理时间统计

## 扩展功能

### 自定义配置
```python
config = {
    'model_dir': 'custom/model/path',
    'device': 'cpu',
    'language': 'en',
    'custom_parameters': 'value'
}

interface = WorkflowInterface(config)
```

### 自定义输出格式
```python
# 可以扩展 _save_batch_results 方法
# 添加自定义输出格式
```

## 故障排除

### 常见问题

1. **模型加载失败**
   - 检查模型路径是否正确
   - 确认设备配置（GPU/CPU）
   - 检查依赖包版本

2. **音频文件无法处理**
   - 确认文件格式支持
   - 检查文件是否损坏
   - 验证文件路径

3. **输出文件保存失败**
   - 检查输出目录权限
   - 确认磁盘空间
   - 验证文件路径

### 日志查看
```bash
# 查看处理日志
tail -f logs/sensevoice_batch_*.log
```

## 版本信息

- 版本: 1.0
- 依赖: funasr>=0.10.0, torch>=2.0.0
- 支持格式: mp3, wav, m4a, flac, aac, ogg
- 支持语言: 中文, 英文, 粤语, 日语, 韩语
- 时间戳支持: ✅

## 许可证

本项目遵循项目整体许可证。 