from pathlib import Path
import whisper
import json
from datetime import datetime
from typing import Tuple, Optional

class AudioProcessor:
    def __init__(self, output_dir: str):
        """初始化音頻處理器"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加載 Whisper 模型
        print("正在加載 Whisper 模型...")
        self.model = whisper.load_model("base")
    
    def transcribe(self, audio_path: str) -> Tuple[Optional[str], Optional[Path]]:
        """
        轉錄音頻文件
        
        Args:
            audio_path: 音頻文件路徑
            
        Returns:
            tuple: (轉錄內容, 輸出文件路徑)
        """
        try:
            # 轉錄音頻
            print(f"正在轉錄音頻：{audio_path}")
            result = self.model.transcribe(audio_path)
            
            # 生成輸出文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"transcript_{timestamp}.txt"
            
            # 保存轉錄結果
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(result["text"])
            
            print(f"轉錄完成，結果已保存到：{output_path}")
            return result["text"], output_path
            
        except Exception as e:
            print(f"轉錄失敗：{str(e)}")
            return None, None 