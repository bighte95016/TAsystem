from typing import Optional
import openai
import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.retriever import BaseRetriever

class LLMProcessor:
    def __init__(self):
        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            raise ValueError("未設置 OPENAI_API_KEY 環境變量")
    
    def get_answer(self, prompt: str) -> Optional[str]:
        """使用 LLM 獲取答案"""
        try:
            response = openai.ChatCompletion.create(
                model="gpt4",
                messages=[
                    {"role": "system", "content": "你是一個專業的助手，請根據提供的內容準確回答問題。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"LLM 處理失敗: {str(e)}")
            return None
            
    def create_qa_chain(self, retriever: BaseRetriever) -> RetrievalQA:
        """創建問答鏈"""
        # 創建提示模板
        prompt_template = """親愛的小朋友，我們來玩個有趣的問答遊戲吧！

我來告訴你一個小故事：
{context}

你想問什麼呢？
{question}

請注意：
1. 請用與問題相同的語言來回答
2. 如果問題是中文，請用中文回答
3. 如果問題是英文，請用英文回答
4. 如果問題是日文，請用日文回答
5. 以此類推...

我會用簡單的例子來告訴你答案，就像：
- 如果是在講天氣，我會用"像太陽公公在微笑"這樣的方式
- 如果是在講數字，我會用"像你有5顆糖果"這樣的方式
- 如果是在講時間，我會用"像你放學後要等媽媽來接"這樣的方式

這樣你就能更容易理解啦！"""

        # 創建提示
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        # 創建 LLM
        llm = ChatOpenAI(
            model_name="gpt-4",
            temperature=0.7
        )

        # 創建問答鏈
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )

        return qa_chain 