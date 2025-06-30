#!/usr/bin/env python3
"""
Speech to Text API 启动脚本
"""

import uvicorn
import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def main():
    """启动API服务器"""
    print("🚀 启动 Speech to Text API 服务器...")
    
    # 检查环境文件
    env_file = Path(__file__).parent / ".env"
    if not env_file.exists():
        print("⚠️  未找到 .env 文件，使用默认配置")
        print("💡 建议复制 env.example 为 .env 并配置API密钥")
    
    # 启动服务器
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main() 