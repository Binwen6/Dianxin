# Speech to Text API

基于FastAPI的语音转文字API服务，支持多种音频格式。

## 功能特性

- 🎵 支持多种音频格式：mp3, mp4, mpeg, mpga, m4a, wav, webm
- 🔐 API密钥认证
- 📁 文件大小限制：15MB
- ⚡ 基于Whisper模型的快速转录
- 📊 详细的处理信息和时间戳
- 🚀 异步处理，支持并发请求

## 快速开始

### 1. 安装依赖

```bash
cd speech2text_api
pip install -r requirements.txt
```

### 2. 配置环境

复制环境配置文件并修改：

```bash
cp env.example .env
```

编辑 `.env` 文件，设置你的API密钥：

```env
API_KEY=your-secret-api-key-here
```

### 3. 启动服务

```bash
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

或者直接运行：

```bash
python app.py
```

### 4. 访问API文档

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API使用

### 认证

所有API请求都需要在Authorization header中包含API密钥：

```
Authorization: Bearer your-secret-api-key-here
```

### 语音转文字

**端点：** `POST /audio-to-text`

**请求格式：** `multipart/form-data`

**参数：**
- `file` (必需): 音频文件
- `user` (必需): 用户标识

**示例请求：**

```bash
curl --request POST \
  --url http://localhost:8000/audio-to-text \
  --header 'Authorization: Bearer your-secret-api-key-here' \
  --header 'Content-Type: multipart/form-data' \
  --form 'file=@/path/to/your/audio.mp3' \
  --form 'user=user123'
```

**响应示例：**

```json
{
  "text": "这是转换后的文字内容",
  "user": "user123",
  "file_name": "audio.mp3",
  "file_size": 1024000,
  "model_used": "openai/whisper-large-v3-turbo",
  "processing_time": 2.5,
  "timestamp": "2024-01-01T12:00:00"
}
```

### Python客户端示例

```python
import requests

url = "http://localhost:8000/audio-to-text"
headers = {
    "Authorization": "Bearer your-secret-api-key-here"
}

with open("audio.mp3", "rb") as f:
    files = {"file": f}
    data = {"user": "user123"}
    
    response = requests.post(url, headers=headers, files=files, data=data)
    result = response.json()
    print(result["text"])
```

## 配置选项

### 环境变量

| 变量名 | 默认值 | 描述 |
|--------|--------|------|
| `HOST` | `0.0.0.0` | 服务监听地址 |
| `PORT` | `8000` | 服务端口 |
| `DEBUG` | `false` | 调试模式 |
| `API_KEY` | `your-secret-api-key-here` | API密钥 |
| `WHISPER_MODEL` | `openai/whisper-large-v3-turbo` | Whisper模型ID |
| `RETURN_TIMESTAMPS` | `false` | 是否返回时间戳 |
| `CHUNK_LENGTH_S` | `30` | 音频分块长度（秒） |
| `LOG_LEVEL` | `INFO` | 日志级别 |

### 支持的音频格式

- MP3 (.mp3)
- MP4 (.mp4)
- MPEG (.mpeg)
- MPGA (.mpga)
- M4A (.m4a)
- WAV (.wav)
- WebM (.webm)

## 错误处理

API会返回标准的HTTP状态码和错误信息：

- `400 Bad Request`: 请求参数错误（如文件格式不支持、文件过大）
- `401 Unauthorized`: API密钥无效
- `500 Internal Server Error`: 服务器内部错误

## 部署

### Docker部署

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 生产环境建议

1. 使用HTTPS
2. 设置强密码的API密钥
3. 配置反向代理（如Nginx）
4. 使用进程管理器（如PM2或systemd）
5. 配置日志轮转
6. 监控服务状态

## 开发

### 项目结构

```
speech2text_api/
├── app.py                 # 主应用文件
├── core/                  # 核心模块
│   ├── config.py         # 配置管理
│   └── auth.py           # 认证模块
├── models/               # 数据模型
│   ├── request_models.py # 请求模型
│   └── response_models.py # 响应模型
├── services/             # 业务服务
│   └── transcription_service.py # 转录服务
├── requirements.txt      # 依赖列表
├── env.example          # 环境配置示例
└── README.md            # 项目文档
```

### 运行测试

```bash
# 安装测试依赖
pip install pytest httpx

# 运行测试
pytest
```

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request！ 