#!/bin/bash
# M4A快速转换脚本
# 使用方法: ./quick_convert.sh [输入目录] [输出目录]

# 设置默认值
INPUT_DIR=${1:-"datasets/mp3"}
OUTPUT_DIR=${2:-"converted"}

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🎵 M4A批量转换工具${NC}"
echo -e "${BLUE}==================${NC}"

# 检查Python是否安装
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python3 未安装${NC}"
    exit 1
fi

# 检查FFmpeg是否安装
if ! command -v ffmpeg &> /dev/null; then
    echo -e "${RED}❌ FFmpeg 未安装${NC}"
    echo -e "${YELLOW}请安装FFmpeg:${NC}"
    echo -e "  macOS: brew install ffmpeg"
    echo -e "  Ubuntu: sudo apt install ffmpeg"
    exit 1
fi

# 检查输入目录是否存在
if [ ! -d "$INPUT_DIR" ]; then
    echo -e "${RED}❌ 输入目录不存在: $INPUT_DIR${NC}"
    exit 1
fi

# 检查是否有m4a文件
M4A_COUNT=$(find "$INPUT_DIR" -name "*.m4a" | wc -l)
if [ "$M4A_COUNT" -eq 0 ]; then
    echo -e "${YELLOW}⚠️  在 $INPUT_DIR 中未找到m4a文件${NC}"
    exit 1
fi

echo -e "${GREEN}✅ 找到 $M4A_COUNT 个m4a文件${NC}"
echo -e "${BLUE}📁 输入目录: $INPUT_DIR${NC}"
echo -e "${BLUE}📁 输出目录: $OUTPUT_DIR${NC}"

# 询问用户是否继续
read -p "是否开始转换? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}取消转换${NC}"
    exit 0
fi

# 执行转换
echo -e "${BLUE}🔄 开始转换...${NC}"
python3 utils/m4a_to_mp3.py "$INPUT_DIR" -o "$OUTPUT_DIR" -q 192k -w 4

# 检查转换结果
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ 转换完成!${NC}"
    echo -e "${BLUE}📁 输出目录: $OUTPUT_DIR${NC}"
    
    # 显示转换后的文件
    MP3_COUNT=$(find "$OUTPUT_DIR" -name "*.mp3" | wc -l)
    echo -e "${GREEN}📊 成功转换 $MP3_COUNT 个MP3文件${NC}"
else
    echo -e "${RED}❌ 转换失败${NC}"
    exit 1
fi 