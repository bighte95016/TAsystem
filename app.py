import streamlit as st
import os
import sys
from pathlib import Path
import tempfile
import time
import sounddevice as sd
import soundfile as sf
import numpy as np
import pygame
from modules.audio_processor import AudioProcessor
from modules.text_processor import TextProcessor
from modules.vector_store import VectorStore
from modules.llm_processor import LLMProcessor
from modules.voice_qa import VoiceQA
import pyttsx3
import subprocess

# 初始化 pygame 音頻
try:
    pygame.mixer.init()
    pygame.init()
    pygame_initialized = True
    print("Pygame 初始化成功")
except Exception as e:
    pygame_initialized = False
    print(f"Pygame 初始化失敗: {str(e)}")

# 設定頁面
st.set_page_config(
    page_title="教學問答系統",
    page_icon="🎙️",
    layout="wide"
)

# 設定目錄路徑
TRANSCRIBED_DATA_DIR = "transcribed_data"
VECTOR_STORE_DIR = "vector_store"
VOICE_QUESTIONS_DIR = "voice_questions"
os.makedirs(TRANSCRIBED_DATA_DIR, exist_ok=True)
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
os.makedirs(VOICE_QUESTIONS_DIR, exist_ok=True)

# 初始化組件
@st.cache_resource
def load_components():
    audio_processor = AudioProcessor(TRANSCRIBED_DATA_DIR)
    text_processor = TextProcessor()
    vector_store = VectorStore(VECTOR_STORE_DIR)
    llm_processor = LLMProcessor()
    voice_qa = VoiceQA(use_local_tts=True)
    
    # 創建問答鏈
    qa_chain = llm_processor.create_qa_chain(
        vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
    )
    
    return {
        "audio_processor": audio_processor,
        "text_processor": text_processor,
        "vector_store": vector_store,
        "llm_processor": llm_processor,
        "voice_qa": voice_qa,
        "qa_chain": qa_chain
    }

# 處理音頻文件
def process_audio_file(components, audio_path):
    # 轉錄音頻
    content, output_path = components["audio_processor"].transcribe(audio_path)
    if not content or not output_path:
        return False, "音頻轉錄失敗"
        
    # 清理文本
    cleaned_content = components["text_processor"].clean_text(content)
    
    # 分塊處理
    chunks = components["text_processor"].split_text(cleaned_content)
    
    # 添加每個塊到向量存儲
    success = True
    chunk_count = 0
    for chunk in chunks:
        metadata = {
            "source": str(output_path),
            "timestamp": output_path.stem.split("_")[-1]
        }
        if components["vector_store"].add_content(chunk, metadata):
            chunk_count += 1
        else:
            success = False
            break
    
    if success:
        return True, f"處理成功! 文本已分為 {chunk_count} 個塊並存儲到向量數據庫"
    else:
        return False, "向量存儲失敗"

# 回答問題
def answer_question(components, question):
    try:
        result = components["qa_chain"]({"query": question})
        return result["result"]
    except Exception as e:
        st.error(f"問答失敗：{str(e)}")
        return "抱歉，我無法回答這個問題。"

# 語音合成並播放
def speak_answer(components, text):
    voice_qa = components["voice_qa"]
    
    # 檢測語言
    lang = voice_qa.detect_language(text)
    st.info(f"檢測到語言: {lang}")
    
    # 顯示處理訊息
    status_msg = st.empty()
    status_msg.info("正在生成語音...")
    
    # 日文和韓文使用在線服務
    if lang in ['ja', 'ko']:
        use_online_tts = True
    else:
        use_online_tts = False
    
    # 判斷是否使用本地 TTS
    if not use_online_tts:
        try:
            status_msg.info("使用本地語音引擎...")
            
            # 創建一個獨立的 Python 腳本來運行 TTS
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py', encoding='utf-8') as script_file:
                script_path = script_file.name
                script_file.write("""# -*- coding: utf-8 -*-
import sys
import pyttsx3
import os

def speak_text(text_file_path):
    try:
        # 讀取文本文件
        with open(text_file_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
            
        engine = pyttsx3.init()
        engine.setProperty('rate', 180)
        engine.setProperty('volume', 0.9)
        
        # 嘗試設置中文語音
        try:
            voices = engine.getProperty('voices')
            for voice in voices:
                if "chinese" in voice.name.lower() or "zh" in voice.id.lower():
                    engine.setProperty('voice', voice.id)
                    break
        except Exception as e:
            print(f"設置語音失敗: {str(e)}")
        
        # 播放語音
        engine.say(text)
        engine.runAndWait()
        
        # 釋放資源
        engine.stop()
        
        print("TTS播放完成")
        return True
    except Exception as e:
        print(f"TTS錯誤: {str(e)}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        text_file_path = sys.argv[1]
        speak_text(text_file_path)
""")
            
            # 創建文本臨時文件
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as text_file:
                text_file_path = text_file.name
                text_file.write(text)
            
            # 執行腳本(使用子進程而非線程)
            status_msg.info("正在播放語音...")
            
            # 使用子進程執行腳本，傳遞文本文件路徑
            process = subprocess.Popen(
                [sys.executable, script_path, text_file_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # 等待進程完成
            stdout, stderr = process.communicate(timeout=30)
            
            # 刪除臨時文件
            try:
                os.unlink(script_path)
                os.unlink(text_file_path)
            except Exception as e:
                print(f"刪除臨時文件失敗: {str(e)}")
            
            # 檢查執行結果
            if process.returncode == 0 and "TTS播放完成" in stdout:
                status_msg.success("本地語音播放完成")
                return
            else:
                st.warning(f"本地TTS未正確執行: {stderr}")
                use_online_tts = True
                
        except Exception as e:
            #st.warning(f"本地語音生成失敗，原因: {str(e)}，將嘗試在線服務...")  暫時關閉20250325
            use_online_tts = True
    
    # 如果需要使用在線 gTTS 服務
    if use_online_tts:
        try:
            status_msg.info("使用在線語音服務...")
            
            # 創建臨時文件
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
                temp_path = temp_file.name
                
            # 生成語音
            from gtts import gTTS
            
            # 對於中文，使用zh-TW標記
            if lang == 'zh-tw':
                tts_lang = 'zh-TW'
            # 對於日語
            elif lang == 'ja':
                tts_lang = 'ja'
            # 對於韓語
            elif lang == 'ko':
                tts_lang = 'ko'
            # 其他語言（包括英語）
            else:
                tts_lang = 'en'
                
            tts = gTTS(text=text, lang=tts_lang, slow=False)
            tts.save(temp_path)
            
            # 顯示音頻播放器
            st.audio(temp_path, format="audio/mp3")
            status_msg.success("語音生成成功！請點擊上方播放按鈕收聽。")
            
            # 刪除臨時文件（延遲刪除）
            import threading
            def delete_file_later():
                import time
                time.sleep(10)  # 等待 10 秒再刪除
                try:
                    os.unlink(temp_path)
                    print(f"臨時文件已刪除: {temp_path}")
                except Exception as e:
                    print(f"刪除臨時文件失敗: {str(e)}")
                    
            threading.Thread(target=delete_file_later).start()
            
        except Exception as e:
            status_msg.error(f"所有語音生成方法均失敗: {str(e)}")
            st.exception(e)

# 錄製音頻
def record_audio(duration=5, fs=16000):
    st.write("正在錄音...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    st.write("錄音完成!")
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    audio_path = os.path.join(VOICE_QUESTIONS_DIR, f"question_{timestamp}.wav")
    sf.write(audio_path, recording, fs)
    
    return audio_path

# 主應用程序
def main():
    st.title("🎙️ 教學問答系統")
    
    # 確保 TTS 環境正確
    ensure_tts_environment()
    
    # 加載組件
    with st.spinner("正在加載模型和組件..."):
        components = load_components()
    
    # 檢查 TTS 設定
    check_tts_status(components)
    
    # 創建選項卡
    tab1, tab2 = st.tabs(["📤 音檔上傳和處理", "❓ 語音問答"])
    
    # 選項卡1：音頻上傳和處理
    with tab1:
        st.header("上傳課程音檔")
        
        uploaded_file = st.file_uploader("選擇音檔文件", type=["mp3", "wav", "m4a", "ogg"], help="上傳音檔文件，支持多種格式")
        
        if uploaded_file is not None:
            # 顯示音頻播放器
            st.audio(uploaded_file, format="audio/wav")
            
            # 處理按鈕
            if st.button("處理音檔", type="primary"):
                with st.spinner("正在處理音檔..."):
                    # 保存上傳的文件到臨時文件
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        audio_path = tmp_file.name
                    
                    # 處理音頻文件
                    success, message = process_audio_file(components, audio_path)
                    
                    # 刪除臨時文件
                    os.unlink(audio_path)
                    
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
    
    # 選項卡2：語音問答
    with tab2:
        st.header("語音問答")
        
        # 問答方式選擇
        qa_method = st.radio("選擇提問方式", ["文字輸入", "語音輸入"], horizontal=True)
        
        if qa_method == "文字輸入":
            question = st.text_input("請輸入您的問題")
            col1, col2 = st.columns(2)
            
            with col1:
                get_answer = st.button("獲取答案", type="primary")
                
            if get_answer and question:
                # 將答案保存到 session_state 中，以便在頁面重新加載時保持
                if 'current_answer' not in st.session_state:
                    st.session_state.current_answer = ""
                    
                with st.spinner("正在思考..."):
                    answer = answer_question(components, question)
                    st.session_state.current_answer = answer
                    
                st.write("### 答案")
                st.write(answer)
                
                # 顯示朗讀按鈕
                with col2:
                    speak_button = st.button("朗讀答案")
                
                if speak_button and st.session_state.current_answer:
                    speak_answer(components, st.session_state.current_answer)
            elif 'current_answer' in st.session_state and st.session_state.current_answer:
                # 如果有保存的答案但沒有點擊獲取按鈕（如朗讀後的頁面重載）
                st.write("### 答案")
                st.write(st.session_state.current_answer)
                
                # 顯示朗讀按鈕
                with col2:
                    speak_button = st.button("朗讀答案")
                
                if speak_button:
                    speak_answer(components, st.session_state.current_answer)
                
        else:  # 語音輸入
            col1, col2 = st.columns(2)
            
            with col1:
                duration = st.slider("錄音時長（秒）", min_value=3, max_value=30, value=5)
                
            with col2:
                record_button = st.button("開始錄音", type="primary")
                
            if record_button:
                # 保存錄音狀態到 session_state
                st.session_state.recording_done = True
                st.session_state.recording_path = record_audio(duration)
                
                # 轉錄問題
                with st.spinner("正在轉錄您的問題..."):
                    content, _ = components["audio_processor"].transcribe(st.session_state.recording_path)
                    st.session_state.question_content = content
                    
                if content:
                    st.write("### 您的問題")
                    st.write(content)
                    
                    # 獲取答案
                    with st.spinner("正在思考..."):
                        answer = answer_question(components, content)
                        st.session_state.voice_answer = answer
                    
                    st.write("### 答案")
                    st.write(answer)
                    
                    # 自動朗讀答案
                    speak_answer(components, answer)
                else:
                    st.error("無法識別您的問題，請重試。")
                    st.session_state.recording_done = False
            
            # 如果已經錄音並有答案，但頁面重新加載
            elif 'recording_done' in st.session_state and st.session_state.recording_done:
                if hasattr(st.session_state, 'question_content') and st.session_state.question_content:
                    st.write("### 您的問題")
                    st.write(st.session_state.question_content)
                
                if hasattr(st.session_state, 'voice_answer') and st.session_state.voice_answer:
                    st.write("### 答案")
                    st.write(st.session_state.voice_answer)
                    
                    # 添加重新播放按鈕
                    if st.button("重新朗讀答案"):
                        speak_answer(components, st.session_state.voice_answer)

# 檢查 TTS 設定
def check_tts_status(components):
    if 'tts_checked' in st.session_state:
        return
    
    import platform
    system = platform.system()
    
    with st.expander("語音合成資訊", expanded=False):
        st.write("### 語音合成系統狀態")
        
        # 檢查本地引擎
        try:
            import pyttsx3
            engine = pyttsx3.init()
            
            try:
                voices = engine.getProperty('voices')
                voice_names = [v.name for v in voices if v.name]
                
                st.success("✅ 本地 TTS 引擎可用")
                st.write(f"平台: {system}")
                st.write(f"語音引擎: pyttsx3 (基於 {engine.proxy._name if hasattr(engine, 'proxy') and hasattr(engine.proxy, '_name') else '未知'})")
                st.write(f"發現 {len(voices)} 個語音")
                
                if voice_names:
                    with st.expander("可用語音列表"):
                        for i, name in enumerate(voice_names):
                            st.write(f"{i+1}. {name}")
                
                # 清理資源
                engine.stop()
                del engine
                
                st.info("系統會優先使用本地 TTS 引擎，僅在特定語言（如日文、韓文）或本地引擎失敗時才使用在線服務")
            except Exception as e:
                st.warning(f"本地 TTS 引擎初始化成功，但無法獲取語音列表: {str(e)}")
                st.info("將使用在線語音服務作為備選")
        except Exception as e:
            st.error(f"❌ 本地 TTS 引擎不可用: {str(e)}")
            st.info("將完全使用在線語音服務")
        
        # 在線服務資訊
        st.write("### 在線 TTS 服務")
        st.success("✅ Google Text-to-Speech (gTTS) 服務可用")
        st.write("支援的語言: 中文、英文、日文、韓文等多種語言")
        st.write("需要網絡連接")
        
    # 標記為已檢查
    st.session_state.tts_checked = True

# 重置 TTS 引擎
def reset_tts_engine():
    """強制重置 pyttsx3 引擎，解決重複使用問題"""
    try:
        import pyttsx3
        import atexit
        
        if 'tts_engine_initialized' not in st.session_state:
            # 清理任何可能存在的 pyttsx3 引擎實例
            try:
                engine = pyttsx3.init()
                engine.stop()
                del engine
            except:
                pass
            
            # 註冊退出時的清理函數
            def cleanup_tts():
                try:
                    engine = pyttsx3.init()
                    engine.stop()
                    del engine
                    print("TTS 引擎已清理")
                except:
                    pass
            
            atexit.register(cleanup_tts)
            
            st.session_state.tts_engine_initialized = True
            print("TTS 引擎重置成功")
    except Exception as e:
        print(f"重置 TTS 引擎失敗: {str(e)}")

# 確保 TTS 環境正確
def ensure_tts_environment():
    """確保 TTS 環境正確設置，適合多次使用"""
    try:
        # 重置 TTS 引擎
        reset_tts_engine()
        
        # 在 Windows 上進行額外檢查
        import platform
        if platform.system() == 'Windows':
            try:
                # 檢查 Windows 的 SAPI 可用性
                import win32com.client
                speaker = win32com.client.Dispatch("SAPI.SpVoice")
                voices = speaker.GetVoices()
                
                # 記錄找到的語音
                print(f"Windows SAPI 找到 {voices.Count} 個語音")
                for i in range(voices.Count):
                    voice = voices.Item(i)
                    print(f"語音 {i+1}: {voice.GetDescription()}")
                
                # 清理資源
                del speaker
                del voices
                
                print("Windows SAPI 正常")
            except Exception as e:
                print(f"Windows SAPI 檢查失敗: {str(e)}")
    except Exception as e:
        print(f"TTS 環境檢查錯誤: {str(e)}")
    
    return True

if __name__ == "__main__":
    main() 