# 語音轉文字程序

這是一個使用 OpenAI Whisper 的語音轉文字程序，可以將各種音頻文件轉換為文字。

## 安裝步驟

1. 首先確保您已安裝 Python 3.7 或更高版本
2. 安裝所需的依賴包：
   ```bash
   pip install -r requirements.txt
   ```

## 使用方法

1. 準備您要轉錄的音頻文件（支持格式：mp3, wav, m4a, wma 等）
2. 在命令行中運行以下命令：
   ```bash
   python speech_to_text.py <音頻文件路徑>
   ```
   例如：
   ```bash
   python speech_to_text.py my_audio.mp3
   ```

## 輸出

程序會：
1. 在控制台顯示轉錄的文字
2. 自動將轉錄結果保存到一個文本文件中（文件名格式：原音頻文件名_transcription.txt）

## 注意事項

- 首次運行時，程序會自動下載 Whisper 模型（大約 1GB）
- 轉錄速度取決於您的電腦性能和音頻文件長度
- 建議使用較短的音頻文件進行測試 