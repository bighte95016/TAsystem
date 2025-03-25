import subprocess
import sys
import os

def main():
    """運行 Streamlit 應用程序"""
    print("正在啟動音頻問答系統...")
    
    # 檢查 streamlit 是否安裝
    try:
        import streamlit
        print("找到 Streamlit 版本:", streamlit.__version__)
    except ImportError:
        print("未找到 Streamlit，請先安裝依賴：")
        print("pip install -r requirements.txt")
        sys.exit(1)
    
    # 運行 streamlit 應用
    cmd = ["streamlit", "run", "app.py"]
    try:
        print("正在啟動應用...")
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n應用已停止")
    except Exception as e:
        print(f"啟動應用時出錯：{str(e)}")

if __name__ == "__main__":
    main() 