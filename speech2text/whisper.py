import torch
import os
import json
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WhisperTranscriber:
    def __init__(self, model_id: str = "openai/whisper-large-v3-turbo", return_timestamps: bool = False):
        """
        初始化Whisper转录器
        
        Args:
            model_id: 使用的模型ID
            return_timestamps: 是否返回时间戳信息
        """
        self.model_id = model_id
        self.return_timestamps = return_timestamps
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        logger.info(f"使用设备: {self.device}")
        logger.info(f"加载模型: {model_id}")
        logger.info(f"时间戳功能: {'启用' if return_timestamps else '禁用'}")
        
        # 加载模型
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, 
            torch_dtype=self.torch_dtype, 
            low_cpu_mem_usage=True, 
            use_safetensors=True
        )
        self.model.to(self.device)
        
        # 加载处理器
        self.processor = AutoProcessor.from_pretrained(model_id)
        
        # 创建pipeline
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device,
            return_timestamps=return_timestamps,
        )
        
        logger.info("模型加载完成")
    
    def transcribe_file(self, audio_path: str, chunk_length_s: int = 30) -> Dict:
        """
        转录单个音频文件
        
        Args:
            audio_path: 音频文件路径
            chunk_length_s: 音频分块长度（秒），用于长音频处理
            
        Returns:
            包含转录结果的字典
        """
        try:
            logger.info(f"开始转录文件: {audio_path}")
            
            # 获取文件信息
            file_path = Path(audio_path)
            file_size = file_path.stat().st_size if file_path.exists() else 0
            
            # 执行转录
            if self.return_timestamps:
                # 启用时间戳的转录
                result = self.pipe(
                    audio_path,
                    chunk_length_s=chunk_length_s,
                    stride_length_s=1,
                    return_timestamps=True
                )
            else:
                # 普通转录
                result = self.pipe(audio_path)
            
            # 构建结果字典
            transcription_result = {
                "file_path": str(audio_path),
                "file_name": file_path.name,
                "file_size_bytes": file_size,
                "transcription": result["text"],
                "confidence": result.get("confidence", None),
                "timestamp": datetime.datetime.now().isoformat(),
                "model_used": self.model_id,
                "device_used": self.device,
                "processing_time": None,  # 可以添加计时功能
                "return_timestamps": self.return_timestamps
            }
            
            # 如果启用了时间戳，添加时间戳信息
            if self.return_timestamps and "chunks" in result:
                transcription_result["chunks"] = result["chunks"]
                
                # 生成词级别时间戳
                # word_timestamps = self._extract_word_timestamps(result)
                # transcription_result["word_timestamps"] = word_timestamps
                
                # 生成段落级别时间戳
                segment_timestamps = self._extract_segment_timestamps(result)
                transcription_result["segment_timestamps"] = segment_timestamps
            
            logger.info(f"转录完成: {audio_path}")
            return transcription_result
            
        except Exception as e:
            logger.error(f"转录文件 {audio_path} 时出错: {str(e)}")
            return {
                "file_path": str(audio_path),
                "file_name": Path(audio_path).name,
                "error": str(e),
                "timestamp": datetime.datetime.now().isoformat(),
                "model_used": self.model_id,
                "device_used": self.device,
                "return_timestamps": self.return_timestamps
            }
    
    def _extract_word_timestamps(self, result: Dict) -> List[Dict]:
        """
        从转录结果中提取词级别时间戳
        
        Args:
            result: 转录结果
            
        Returns:
            词级别时间戳列表
        """
        word_timestamps = []
        
        if "chunks" in result:
            for chunk in result["chunks"]:
                if "timestamp" in chunk:
                    start_time, end_time = chunk["timestamp"]
                    text = chunk["text"].strip()
                    
                    # 简单的词分割（可以进一步优化）
                    words = text.split()
                    if words:
                        # 平均分配时间给每个词
                        time_per_word = (end_time - start_time) / len(words)
                        for i, word in enumerate(words):
                            word_start = start_time + i * time_per_word
                            word_end = word_start + time_per_word
                            word_timestamps.append({
                                "word": word,
                                "start": round(word_start, 2),
                                "end": round(word_end, 2)
                            })
        
        return word_timestamps
    
    def _extract_segment_timestamps(self, result: Dict) -> List[Dict]:
        """
        从转录结果中提取段落级别时间戳
        
        Args:
            result: 转录结果
            
        Returns:
            段落级别时间戳列表
        """
        segment_timestamps = []
        
        if "chunks" in result:
            for chunk in result["chunks"]:
                if "timestamp" in chunk:
                    start_time, end_time = chunk["timestamp"]
                    text = chunk["text"].strip()
                    
                    segment_timestamps.append({
                        "text": text,
                        "start": round(start_time, 2),
                        "end": round(end_time, 2),
                        "duration": round(end_time - start_time, 2)
                    })
        
        return segment_timestamps
    
    def transcribe_directory(self, input_dir: str, output_dir: str = "output") -> List[Dict]:
        """
        批量转录目录中的音频文件
        
        Args:
            input_dir: 输入目录路径
            output_dir: 输出目录路径
            
        Returns:
            所有转录结果的列表
        """
        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 支持的音频格式
        audio_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac'}
        
        # 获取所有音频文件
        input_path = Path(input_dir)
        audio_files = []
        
        for ext in audio_extensions:
            audio_files.extend(input_path.glob(f"*{ext}"))
            audio_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        if not audio_files:
            logger.warning(f"在目录 {input_dir} 中未找到音频文件")
            return []
        
        logger.info(f"找到 {len(audio_files)} 个音频文件")
        
        # 转录所有文件
        results = []
        for audio_file in audio_files:
            result = self.transcribe_file(str(audio_file))
            results.append(result)
        
        # 保存结果
        self.save_results(results, output_dir)
        
        return results
    
    def save_results(self, results: List[Dict], output_dir: str):
        """
        保存转录结果到文件
        
        Args:
            results: 转录结果列表
            output_dir: 输出目录
        """
        output_path = Path(output_dir)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存为JSON文件
        json_file = output_path / f"transcription_results_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 保存为transcripts.json格式（包含时间戳）
        transcripts_file = output_path / "transcripts.json"
        transcripts_data = self._format_transcripts_json(results)
        with open(transcripts_file, 'w', encoding='utf-8') as f:
            json.dump(transcripts_data, f, ensure_ascii=False, indent=2)
        
        # 保存为文本文件
        txt_file = output_path / f"transcription_text_{timestamp}.txt"
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(f"转录结果 - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n\n")
            
            for i, result in enumerate(results, 1):
                f.write(f"文件 {i}: {result.get('file_name', 'Unknown')}\n")
                f.write(f"路径: {result.get('file_path', 'Unknown')}\n")
                f.write(f"时间: {result.get('timestamp', 'Unknown')}\n")
                
                if 'error' in result:
                    f.write(f"错误: {result['error']}\n")
                else:
                    f.write(f"转录内容:\n{result.get('transcription', '')}\n")
                    
                    # 如果启用了时间戳，添加时间戳信息
                    if result.get('return_timestamps', False) and 'segment_timestamps' in result:
                        f.write(f"\n时间戳信息:\n")
                        for segment in result['segment_timestamps']:
                            f.write(f"[{segment['start']:.2f}s - {segment['end']:.2f}s] {segment['text']}\n")
                
                f.write("\n" + "-" * 30 + "\n\n")
        
        # 保存摘要
        summary_file = output_path / f"transcription_summary_{timestamp}.json"
        summary = {
            "total_files": len(results),
            "successful_transcriptions": len([r for r in results if 'error' not in r]),
            "failed_transcriptions": len([r for r in results if 'error' in r]),
            "timestamp": datetime.datetime.now().isoformat(),
            "model_used": self.model_id,
            "device_used": self.device,
            "return_timestamps": self.return_timestamps
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        # 如果启用了时间戳，保存SRT字幕文件
        if self.return_timestamps:
            srt_file = output_path / f"transcription_subtitles_{timestamp}.srt"
            self._save_srt_file(results, srt_file)
            logger.info(f"字幕文件: {srt_file}")
        
        logger.info(f"结果已保存到 {output_dir}")
        logger.info(f"JSON文件: {json_file}")
        logger.info(f"transcripts.json: {transcripts_file}")
        logger.info(f"文本文件: {txt_file}")
        logger.info(f"摘要文件: {summary_file}")
    
    def _format_transcripts_json(self, results: List[Dict]) -> Dict:
        """
        格式化transcripts.json数据
        
        Args:
            results: 转录结果列表
            
        Returns:
            格式化的transcripts.json数据
        """
        transcripts_data = {
            "metadata": {
                "total_files": len(results),
                "successful_transcriptions": len([r for r in results if 'error' not in r]),
                "failed_transcriptions": len([r for r in results if 'error' in r]),
                "timestamp": datetime.datetime.now().isoformat(),
                "model_used": self.model_id,
                "device_used": self.device,
                "return_timestamps": self.return_timestamps
            },
            "transcripts": []
        }
        
        for result in results:
            transcript_entry = {
                "file_name": result.get('file_name', 'Unknown'),
                "file_path": result.get('file_path', 'Unknown'),
                "file_size_bytes": result.get('file_size_bytes', 0),
                "timestamp": result.get('timestamp', ''),
                "model_used": result.get('model_used', self.model_id),
                "device_used": result.get('device_used', self.device)
            }
            
            if 'error' in result:
                transcript_entry["status"] = "error"
                transcript_entry["error"] = result['error']
                transcript_entry["transcription"] = ""
                transcript_entry["segments"] = []
            else:
                transcript_entry["status"] = "success"
                transcript_entry["transcription"] = result.get('transcription', '')
                transcript_entry["confidence"] = result.get('confidence', None)
                transcript_entry["processing_time"] = result.get('processing_time', None)
                
                # 添加时间戳信息
                if result.get('return_timestamps', False) and 'segment_timestamps' in result:
                    transcript_entry["segments"] = result['segment_timestamps']
                else:
                    transcript_entry["segments"] = []
            
            transcripts_data["transcripts"].append(transcript_entry)
        
        return transcripts_data
    
    def _save_srt_file(self, results: List[Dict], srt_file: Path):
        """
        保存SRT字幕文件
        
        Args:
            results: 转录结果列表
            srt_file: SRT文件路径
        """
        with open(srt_file, 'w', encoding='utf-8') as f:
            subtitle_index = 1
            
            for result in results:
                if 'error' not in result and 'segment_timestamps' in result:
                    f.write(f"# {result.get('file_name', 'Unknown')}\n")
                    
                    for segment in result['segment_timestamps']:
                        # 转换时间格式为SRT格式 (HH:MM:SS,mmm)
                        start_time = self._seconds_to_srt_time(segment['start'])
                        end_time = self._seconds_to_srt_time(segment['end'])
                        
                        f.write(f"{subtitle_index}\n")
                        f.write(f"{start_time} --> {end_time}\n")
                        f.write(f"{segment['text']}\n\n")
                        
                        subtitle_index += 1
    
    def _seconds_to_srt_time(self, seconds: float) -> str:
        """
        将秒数转换为SRT时间格式
        
        Args:
            seconds: 秒数
            
        Returns:
            SRT格式的时间字符串 (HH:MM:SS,mmm)
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"


def main():
    """主函数"""
    # 初始化转录器（启用时间戳）
    transcriber = WhisperTranscriber(return_timestamps=True)
    
    # 设置输入和输出目录
    input_directory = "datasets"
    output_directory = "output"
    
    # 批量转录
    results = transcriber.transcribe_directory(input_directory, output_directory)
    
    # 打印摘要
    successful = len([r for r in results if 'error' not in r])
    failed = len([r for r in results if 'error' in r])
    
    print(f"\n转录完成!")
    print(f"总文件数: {len(results)}")
    print(f"成功: {successful}")
    print(f"失败: {failed}")
    print(f"时间戳功能: {'启用' if transcriber.return_timestamps else '禁用'}")
    
    if results:
        print(f"\n结果已保存到 {output_directory} 目录")
        
        # 显示时间戳示例
        for result in results:
            if 'error' not in result and 'segment_timestamps' in result:
                print(f"\n文件: {result['file_name']}")
                print("时间戳示例:")
                for segment in result['segment_timestamps'][:3]:  # 只显示前3个段落
                    print(f"  [{segment['start']:.2f}s - {segment['end']:.2f}s] {segment['text']}")
                break


if __name__ == "__main__":
    main()
