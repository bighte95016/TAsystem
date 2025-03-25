from pathlib import Path
import argparse
from modules.audio_processor import AudioProcessor
from modules.text_processor import TextProcessor
from modules.vector_store import VectorStore
from modules.llm_processor import LLMProcessor
from modules.voice_qa import VoiceQA

class AudioQASystem:
    def __init__(self, output_dir: str, vector_store_dir: str, use_local_tts: bool = True):
        """初始化音頻問答系統"""
        self.audio_processor = AudioProcessor(output_dir)
        self.text_processor = TextProcessor()
        self.vector_store = VectorStore(vector_store_dir)
        self.llm_processor = LLMProcessor()
        self.voice_qa = VoiceQA(use_local_tts=use_local_tts)
        
        # 創建問答鏈
        self.qa_chain = self.llm_processor.create_qa_chain(
            self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            )
        )
    
    def process_audio(self, audio_path: str) -> bool:
        """處理音頻文件"""
        # 轉錄音頻
        content, output_path = self.audio_processor.transcribe(audio_path)
        if not content or not output_path:
            return False
            
        # 清理文本
        cleaned_content = self.text_processor.clean_text(content)
        
        # 分塊處理
        chunks = self.text_processor.split_text(cleaned_content)
        
        # 添加每個塊到向量存儲
        success = True
        for chunk in chunks:
            metadata = {
                "source": str(output_path),
                "timestamp": output_path.stem.split("_")[-1]
            }
            if not self.vector_store.add_content(chunk, metadata):
                success = False
                break
        
        return success
    
    def answer_question(self, question: str) -> str:
        """回答問題"""
        try:
            result = self.qa_chain({"query": question})
            return result["result"]
        except Exception as e:
            print(f"問答失敗：{str(e)}")
            return "抱歉，我無法回答這個問題。"
            
    def voice_qa_loop(self):
        """語音問答循環"""
        print("\n歡迎使用語音問答系統！")
        print("你可以用語音問問題，輸入 'q' 退出程序。")
        
        while True:
            try:
                # 錄音並獲取答案
                answer = self.voice_qa.ask_question(self)
                
                # 詢問是否繼續
                choice = input("\n要繼續問問題嗎？(y/n): ").strip().lower()
                if choice != 'y':
                    break
                    
            except KeyboardInterrupt:
                print("\n程序已停止")
                break
            except Exception as e:
                print(f"發生錯誤：{str(e)}")
                break

def main():
    parser = argparse.ArgumentParser(description="音頻問答系統")
    parser.add_argument("--audio-path", required=True, help="音頻文件路徑")
    parser.add_argument("--output-dir", default="transcribed_data", help="輸出目錄")
    parser.add_argument("--vector-store-dir", default="vector_store", help="向量存儲目錄")
    parser.add_argument("--use-online-tts", action="store_true", help="使用在線TTS服務(gTTS)而非本地TTS")
    args = parser.parse_args()
    
    # 初始化系統
    qa_system = AudioQASystem(args.output_dir, args.vector_store_dir, use_local_tts=not args.use_online_tts)
    
    # 處理音頻
    print(f"正在處理音頻文件：{args.audio_path}")
    if qa_system.process_audio(args.audio_path):
        print("音頻處理完成")
        
        # 開始語音問答循環
        qa_system.voice_qa_loop()
    else:
        print("音頻處理失敗")

if __name__ == "__main__":
    main() 