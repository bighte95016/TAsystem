from typing import List, Dict, Any
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter

class TextProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """初始化文本處理器"""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "，", " ", ""]
        )
    
    def split_text(self, text: str) -> List[str]:
        """將文本分割成固定大小的塊"""
        return self.text_splitter.split_text(text)
    
    def clean_text(self, text: str) -> str:
        """清理文本"""
        # 移除多餘的空白字符
        text = re.sub(r'\s+', ' ', text)
        # 移除特殊字符
        text = re.sub(r'[^\w\s\u4e00-\u9fff。，！？、]', '', text)
        return text.strip()
    
    def format_qa_prompt(self, question: str, context: str) -> str:
        """格式化問答提示"""
        return f"""請根據以下內容回答問題。如果無法從內容中找到答案，請說明。

內容：
{context}

問題：{question}

請提供準確、簡潔的答案。"""
    
    def format_search_results(self, results: List[Dict[str, Any]]) -> str:
        """格式化搜索結果"""
        formatted_text = ""
        for i, result in enumerate(results, 1):
            formatted_text += f"來源 {i}:\n{result['content']}\n\n"
        return formatted_text.strip() 