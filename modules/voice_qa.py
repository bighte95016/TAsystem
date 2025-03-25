import sounddevice as sd
import numpy as np
import wave
import os
from datetime import datetime
from pathlib import Path
import whisper
import tempfile
from gtts import gTTS
import pygame
import pyttsx3

class VoiceQA:
    def __init__(self, output_dir: str = "voice_questions", use_local_tts: bool = True):
        """初始化語音問答系統"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.model = whisper.load_model("base")
        # 初始化 pygame 用於播放音頻
        pygame.mixer.init()
        # 是否使用本地TTS引擎
        self.use_local_tts = use_local_tts
        # 初始化本地TTS引擎
        if self.use_local_tts:
            self.tts_engine = pyttsx3.init()
            # 設置語速 (默認值為200，值越大語速越快)
            self.tts_engine.setProperty('rate', 220)
        
    def record_question(self, duration: int = 5, sample_rate: int = 44100) -> str:
        """錄製語音問題"""
        print(f"\n請開始說話，持續 {duration} 秒...")
        recording = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype=np.int16
        )
        sd.wait()
        print("錄音結束！")
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"question_{timestamp}.wav"
        filepath = self.output_dir / filename
        
        # 保存錄音文件
        with wave.open(str(filepath), 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(recording.tobytes())
            
        return str(filepath)
    
    def transcribe_question(self, audio_path: str) -> str:
        """將語音轉換為文字"""
        try:
            result = self.model.transcribe(audio_path)
            text = result["text"].strip()
            # 使用Whisper提供的語言檢測結果
            detected_language = result.get("language", "")
            if detected_language:
                print(f"Whisper檢測到的語言: {detected_language}")
            return text
        except Exception as e:
            print(f"語音轉換失敗：{str(e)}")
            return ""
            
    def text_to_speech(self, text: str, lang: str = 'zh-tw') -> str:
        """將文字轉換為語音"""
        if self.use_local_tts:
            try:
                # 如果是日文或韓文，直接使用gTTS，因為pyttsx3對這些語言支持不佳
                if lang in ['ja', 'ko']:
                    print(f"檢測到{lang}語言，切換到在線TTS服務...")
                    return self._gtts_to_speech(text, lang)
                    
                # 直接使用本地引擎播放，無需保存文件
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
                return None  # 無需返回文件路徑
            except Exception as e:
                print(f"本地語音生成失敗：{str(e)}")
                print("嘗試使用在線語音服務...")
                # 如果本地TTS失敗，回退到在線服務
                return self._gtts_to_speech(text, lang)
        else:
            # 使用在線gTTS服務
            return self._gtts_to_speech(text, lang)
    
    def _gtts_to_speech(self, text: str, lang: str = 'zh-tw') -> str:
        """使用gTTS將文字轉換為語音"""
        try:
            # 創建臨時文件
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                temp_path = temp_file.name
            
            # 生成語音
            tts = gTTS(text=text, lang=lang)
            tts.save(temp_path)
            
            return temp_path
        except Exception as e:
            print(f"語音生成失敗：{str(e)}")
            return None
    
    def play_audio(self, audio_path: str):
        """播放音頻文件"""
        try:
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
        except Exception as e:
            print(f"音頻播放失敗：{str(e)}")
    
    def detect_language(self, text: str) -> str:
        """嘗試檢測文本語言"""
        # 優先檢測日文特有字符（平假名和片假名）
        if any('\u3040' <= char <= '\u309F' for char in text) or any('\u30A0' <= char <= '\u30FF' for char in text):
            return 'ja'
            
        # 檢測韓文字符（朝鮮文音節、朝鮮文字母）
        if any('\uac00' <= char <= '\ud7a3' for char in text) or any('\u1100' <= char <= '\u11ff' for char in text):
            return 'ko'
            
        # 檢測中文字符（中日韓統一表意文字）
        if any('\u4e00' <= char <= '\u9fff' for char in text):
            # 只有當同時存在漢字和日文特有字符（平假名和片假名）時才判定為日文
            if any('\u3040' <= char <= '\u309F' for char in text) or any('\u30A0' <= char <= '\u30FF' for char in text):
                return 'ja'
            return 'zh-tw'
            
        # 更多語言檢測可以在這裡添加
        return 'en'  # 默認英語
            
    def ask_question(self, qa_system) -> str:
        """錄音並提問"""
        # 錄製問題
        audio_path = self.record_question()
        
        # 轉換為文字
        question = self.transcribe_question(audio_path)
        if not question:
            return "抱歉，我沒有聽清楚你的問題。"
            
        # 檢測問題語言
        question_lang = self.detect_language(question)
        print(f"\n你的問題是：{question}")
        print(f"問題語言檢測結果：{question_lang}")
        
        # 獲取答案
        answer = qa_system.answer_question(question)
        print(f"\n答案：{answer}")
        
        # 檢測答案語言，預設使用問題的語言
        lang = self.detect_language(answer)
        if lang == 'en' and question_lang != 'en':
            # 如果答案被檢測為英文，但問題不是英文，則使用問題的語言
            # 這是為了處理某些可能未包含特定語言特徵的短回答
            lang = question_lang
            print(f"答案語言檢測為英文，但根據問題語言調整為：{lang}")
        else:
            print(f"答案語言檢測結果：{lang}")
        
        # 將答案轉換為語音並播放
        print("正在播放語音回答...")
        if self.use_local_tts and lang not in ['ja', 'ko']:
            # 使用本地TTS直接播放 (對於中文和英文)
            self.text_to_speech(answer, lang)
        else:
            # 使用gTTS播放 (對於日文、韓文或當本地TTS不可用時)
            speech_file = self.text_to_speech(answer, lang)
            if speech_file:
                self.play_audio(speech_file)
                # 刪除臨時文件
                try:
                    os.unlink(speech_file)
                except:
                    pass
        
        return answer 