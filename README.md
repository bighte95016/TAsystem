# 語音問答系統

這是一個基於 OpenAI Whisper 的語音問答系統，可以將音頻文件轉換為文字，存儲到向量數據庫中，並支持基於語音的問答功能。

## 安裝步驟

1. 首先確保您已安裝 Python 3.7 或更高版本
2. 安裝所需的依賴包：
   ```bash
   pip install -r requirements.txt
   ```

## 使用方法

### 命令行模式

1. 準備您要轉錄的音頻文件（支持格式：mp3, wav, m4a, ogg 等）
2. 在命令行中運行以下命令：
   ```bash
   python audio_qa_system.py --audio-path <音頻文件路徑>
   ```
   例如：
   ```bash
   python audio_qa_system.py --audio-path my_lecture.mp3
   ```

### Streamlit 界面（推薦）

我們提供了一個基於 Streamlit 的用戶友好界面：

1. 運行以下命令啟動 Streamlit 應用：
   ```bash
   python run_app.py
   ```
   或直接：
   ```bash
   streamlit run app.py
   ```

2. 在瀏覽器中打開顯示的鏈接（通常是 http://localhost:8501）

3. 在 Streamlit 界面中：
   - 在"音頻上傳和處理"選項卡中上傳課程音頻文件
   - 在"語音問答"選項卡中通過文字或語音提問

## 功能特點

- **音頻轉文字**：使用 OpenAI Whisper 將音頻轉換為文字
- **向量存儲**：將轉錄內容分塊並存儲到向量數據庫中
- **語音問答**：支持通過語音提問並獲取語音回答
- **文字問答**：支持通過文字提問並獲取文字回答

## 目錄結構

- `modules/`: 包含系統的核心模塊
- `transcribed_data/`: 存儲轉錄的文本文件
- `vector_store/`: 存儲向量數據庫
- `voice_questions/`: 存儲用戶的語音問題
- `app.py`: Streamlit 應用程序
- `run_app.py`: 啟動 Streamlit 應用的腳本

## 注意事項

- 首次運行時，程序會自動下載 Whisper 模型（大約 1GB）
- 轉錄速度取決於您的電腦性能和音頻文件長度
- 建議使用較短的音頻文件進行測試
- 使用語音問答功能需要麥克風訪問權限 