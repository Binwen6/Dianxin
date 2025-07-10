# SenseVoice 智能语音转文字处理器 v2.0

> 🚀 **下一代语音转文字工作流处理器** - 支持音频/视频处理、GPU并行计算、智能分割和AI内容纠错

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![SenseVoice](https://img.shields.io/badge/SenseVoice-Small-green.svg)](https://github.com/alibaba-damo-academy/FunASR)
[![GPU](https://img.shields.io/badge/GPU-CUDA%20支持-orange.svg)](https://developer.nvidia.com/cuda-zone)
[![License](https://img.shields.io/badge/License-MIT-red.svg)](LICENSE)

## 🌟 核心特性

### 🎯 **智能处理能力**
- **🎬 视频音频一体化**: 自动从视频文件提取音频并转写
- **🧠 智能分割技术**: 长音频自动分割，最优化并行处理策略
- **🔧 AI内容纠错**: 集成通义千问API，智能纠错和优化转写结果
- **⚡ GPU并行计算**: 支持多进程GPU并行处理，大幅提升处理速度

### 🚀 **高性能架构**
- **📊 多层次并行**: 文件级 + 分段级双重并行处理
- **🎛️ 智能负载均衡**: 根据音频时长自动调整处理策略
- **💾 内存优化**: 智能模型加载和资源管理
- **📈 性能监控**: 实时处理效率分析和优化建议

### 🎨 **工作流集成**
- **🔗 标准化接口**: 完美适配视频转音频 → 语音转文字 → 文字转PPT工作流
- **📋 多格式输出**: JSON、TXT、工作流元数据等多种格式
- **🕐 时间戳支持**: 精确的段落和句子级时间戳
- **📁 批量处理**: 目录级批量处理，支持混合音频/视频文件

## 📦 安装依赖

### 核心依赖
```bash
pip install -r requirements.txt
```

### 系统依赖 (ffmpeg)
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Windows
# 从 https://ffmpeg.org/download.html 下载并添加到PATH
```

### GPU支持 (可选)
```bash
# CUDA 11.8+
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 🚀 快速开始

### 基础用法

```python
from video_to_text import WorkflowSpeechProcessor, process_audio_for_workflow

# 方法1: 简单处理
result = process_audio_for_workflow(
    audio_input="path/to/audio.mp3",  # 支持音频或视频文件
    output_dir="output",
    config={
        'device': 'cuda:0',
        'enable_dashscope_correction': True,
        'parallel_processing': True
    }
)

# 方法2: 高级配置
processor = WorkflowSpeechProcessor({
    'device': 'cuda:0',
    'parallel_processing': True,
    'enable_smart_segmentation': True,
    'enable_dashscope_correction': True,
    'max_workers': 4
})

result = processor.process_single_audio("video.mp4")
print(f"转写结果: {result['transcription']}")
```

### 命令行使用

```bash
# 🎵 处理音频文件
python video_to_text.py audio.mp3 -o output --enable-correction

# 🎬 处理视频文件  
python video_to_text.py video.mp4 -o output --device cuda:0

# 📁 批量处理目录
python video_to_text.py /path/to/media/directory -o output --parallel

# 🔧 高级配置
python video_to_text.py input.mp4 \
    --device cuda:0 \
    --enable-correction \
    --max-workers 4 \
    --video-audio-quality 2 \
    --dashscope-model qwen-turbo

# 📊 查看配置信息
python video_to_text.py --show-config
```

## ⚙️ 配置参数详解

### 🎛️ 基础配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `device` | `auto` | 处理设备 (`cuda:0`, `cpu`, `auto`) |
| `language` | `auto` | 语言设置 (`auto`, `zh`, `en`, `ja`, `ko`) |
| `model_dir` | `iic/SenseVoiceSmall` | SenseVoice模型路径 |

### ⚡ 并行处理配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `parallel_processing` | `True` | 启用并行处理 |
| `max_workers` | `4` | 最大工作进程数 |
| `enable_smart_segmentation` | `True` | 智能分割长音频 |
| `processing_time_ratio` | `1/17` | 处理时间比率 (研究优化结果) |

### 🧠 AI内容纠错配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `enable_dashscope_correction` | `False` | 启用通义千问纠错 |
| `dashscope_model` | `qwen-turbo` | 使用的通义千问模型 |
| `dashscope_temperature` | `0.1` | 温度参数 (0.0-2.0) |
| `dashscope_max_tokens` | `10000` | 最大token数 |

### 🎬 视频处理配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `video_audio_quality` | `4` | 音频提取质量 (0-9, 0最高质量) |
| `cleanup_extracted_audio` | `False` | 处理后清理临时音频文件 |

## 🔧 高级功能

### 🧠 智能分割算法

系统采用基于研究优化的智能分割算法：

```python
# 最优分割段数计算公式
optimal_segments = sqrt(audio_duration * processing_time_ratio / 7)

# 自动激活条件
if optimal_segments > 2 and audio_duration > 600:
    use_smart_segmentation = True
```

**智能分割优势:**
- 📊 根据音频时长自动计算最优分割策略
- 🔄 分段重叠处理，确保内容完整性
- ⚖️ 平衡并行效率与模型加载开销

### 🚀 GPU并行处理

支持多种并行处理模式：

1. **文件级并行**: 多个音频文件同时处理
2. **分段级并行**: 单个长音频分割后并行处理
3. **混合并行**: 同时支持文件级和分段级并行

```python
# GPU并行配置示例
config = {
    'device': 'cuda:0',
    'parallel_processing': True,
    'max_workers': 4,  # 根据显存大小调整
    'enable_smart_segmentation': True
}
```

### 🔍 AI内容纠错

集成阿里云通义千问API，提供智能内容纠错：

```python
# 启用AI纠错
config = {
    'enable_dashscope_correction': True,
    'dashscope_api_key': 'your_api_key',
    'dashscope_model': 'qwen-turbo',
    'dashscope_temperature': 0.1
}
```

**纠错功能:**
- ✅ 语法错误修正
- ✅ 标点符号优化
- ✅ 同音字纠错
- ✅ 语义连贯性提升

## 📋 输出格式

### 📄 标准transcripts.json格式

```json
{
  "metadata": {
    "total_files": 1,
    "successful_transcriptions": 1,
    "total_words": 150,
    "total_characters": 300,
    "total_audio_duration": 45.2,
    "processing_time": 5.5,
    "model_used": "iic/SenseVoiceSmall",
    "device_used": "cuda:0",
    "workflow_step": "speech2text",
    "timestamp": "2024-01-15T10:30:45"
  },
  "transcripts": [
    {
      "file_name": "audio.mp3",
      "status": "success",
      "transcription": "转写的完整文本内容...",
      "processing_time": 5.5,
      "word_count": 150,
      "character_count": 300,
      "audio_duration": 45.2,
      "segments": [
        {
          "text": "这是第一段内容",
          "start": 0.0,
          "end": 8.2,
          "duration": 8.2
        }
      ],
      "metadata": {
        "model_used": "iic/SenseVoiceSmall",
        "device": "cuda:0",
        "input_type": "video",
        "extraction_successful": true,
        "segmentation_info": {
          "total_segments": 3,
          "successful_segments": 3,
          "segmentation_method": "smart_segmentation"
        },
        "dashscope_correction": {
          "enabled": true,
          "correction_time": 1.2,
          "text_changed": true
        }
      }
    }
  ]
}
```

### 📁 输出文件结构

```
output/
├── audio_name/                    # 每个音频的独立目录
│   ├── transcripts.json           # 标准格式转写结果
│   ├── audio_name_result.json     # 详细处理结果
│   └── audio_name_content.txt     # 纯文本内容
├── all_transcripts.json           # 所有文件的汇总结果
├── all_content_20240115_103045.txt # 汇总文本内容
├── batch_results_20240115_103045.json # 批处理结果
└── workflow_metadata_20240115_103045.json # 工作流元数据
```

## 🎯 支持的格式

### 🎵 音频格式
`.mp3`, `.wav`, `.m4a`, `.flac`, `.aac`, `.ogg`, `.wma`

### 🎬 视频格式  
`.mp4`, `.avi`, `.mkv`, `.mov`, `.wmv`, `.flv`, `.webm`, `.m4v`, `.3gp`, `.3g2`, `.asf`, `.rm`, `.rmvb`, `.vob`, `.ts`, `.mts`, `.m2ts`, `.f4v`, `.divx`, `.xvid`, `.ogv`

## 🔧 故障排除

### 🚨 常见问题

#### GPU内存不足
```bash
# 解决方案1: 减少并发数
python video_to_text.py input.mp4 --max-workers 2

# 解决方案2: 使用CPU模式
python video_to_text.py input.mp4 --device cpu

# 解决方案3: 禁用并行处理
python video_to_text.py input.mp4 --no-parallel
```

#### 模型加载失败
```bash
# 检查模型路径
python video_to_text.py --show-config

# 手动下载模型
# 模型会自动从HuggingFace下载到缓存目录
```

#### ffmpeg未安装
```bash
# 检查ffmpeg状态
python video_to_text.py --show-config

# 安装ffmpeg (见安装依赖部分)
```

### 📊 性能优化建议

1. **GPU模式下的最佳实践:**
   - 单GPU: `max_workers=2-4`
   - 多GPU: 配置不同的`cuda:0`, `cuda:1`设备

2. **长音频处理优化:**
   - 启用智能分割: `--enable-smart-seg`
   - 调整分割参数: `min_segment_duration=30`

3. **批量处理优化:**
   - 小文件(<4个): 自动使用串行处理
   - 大文件批量: 启用并行处理

## 🔗 工作流集成

### 📥 上游集成 (视频转音频)

```python
# 接收来自视频处理模块的数据
upstream_data = {
    'step_name': 'video2audio',
    'output_directory': 'extracted_audios/',
    'video_files': ['video1.mp4', 'video2.mp4'],
    'workflow_id': 'workflow_001'
}

# 处理音频转写
result = process_audio_for_workflow(
    audio_input=upstream_data['output_directory'],
    workflow_id=upstream_data['workflow_id']
)
```

### 📤 下游集成 (文字转PPT)

```python
# 为PPT生成模块准备数据
def prepare_for_ppt_generator(transcripts_data):
    return {
        'workflow_step': 'text2ppt',
        'content_type': 'transcription',
        'slides': [
            {
                'slide_number': i+1,
                'title': f"音频转写: {transcript['file_name']}",
                'content': transcript['transcription'],
                'word_count': transcript['word_count']
            }
            for i, transcript in enumerate(transcripts_data['transcripts'])
        ],
        'metadata': transcripts_data['metadata']
    }

# 传递给PPT生成模块
ppt_data = prepare_for_ppt_generator(result)
```

## 📈 性能基准

### ⚡ 处理速度

| 音频时长 | CPU模式 | GPU模式 | GPU并行模式 |
|----------|---------|---------|-------------|
| 1分钟 | ~6s | ~3s | ~2s |
| 10分钟 | ~60s | ~35s | ~12s |
| 60分钟 | ~600s | ~350s | ~80s |

### 💾 资源占用

| 模式 | GPU显存 | 系统内存 | CPU使用率 |
|------|---------|----------|-----------|
| 单进程 | ~2GB | ~1GB | 50% |
| 并行(4进程) | ~6GB | ~3GB | 90% |

## 🤝 贡献指南

### 🐛 问题报告

发现问题请通过以下方式报告:
1. 详细描述问题现象
2. 提供复现步骤和示例文件
3. 包含系统环境信息 (`--show-config`)

### 🚀 功能建议

欢迎提出新功能建议:
- 新的语音模型支持
- 更多输出格式
- 性能优化方案

## 📚 API文档

### 核心类: WorkflowSpeechProcessor

```python
class WorkflowSpeechProcessor:
    def __init__(self, config: Optional[Dict[str, Any]] = None)
    def process_single_audio(self, audio_path: Union[str, Path]) -> Dict[str, Any]
    def process_audio_directory(self, audio_dir: Union[str, Path]) -> Dict[str, Any]
    def get_workflow_status(self) -> Dict[str, Any]
```

### 便捷函数

```python
def process_audio_for_workflow(
    audio_input: Union[str, Path], 
    output_dir: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]
```

## 📜 更新日志

### v2.0.0 (2024-01-15)
- ✨ 新增视频文件直接处理支持
- ⚡ 实现GPU并行处理架构
- 🧠 集成通义千问AI内容纠错
- 📊 智能分割算法优化
- 🔧 全面重构配置系统

### v1.x.x
- 基础音频转写功能
- 批量处理支持
- 时间戳生成

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 🙏 致谢

- [FunASR](https://github.com/alibaba-damo-academy/FunASR) - SenseVoice模型支持
- [阿里云通义千问](https://dashscope.aliyun.com/) - AI内容纠错服务
- [FFmpeg](https://ffmpeg.org/) - 音视频处理支持

---

**💡 提示**: 如果你觉得这个项目有用，请给我们一个 ⭐ Star！这对我们非常重要！

**📞 联系我们**: 如有任何问题，欢迎提交 Issue 或 Pull Request。 