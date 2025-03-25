from pathlib import Path
import chromadb
from chromadb.config import Settings
import json
from typing import List, Dict, Any, Optional
from langchain.schema import Document
from langchain.vectorstores.base import VectorStore
from langchain.embeddings.base import Embeddings
from langchain.schema.retriever import BaseRetriever
from pydantic import Field

class ChromaRetriever(BaseRetriever):
    vector_store: Any = Field(description="向量存儲實例")
    search_type: str = Field(default="similarity", description="搜索類型")
    search_kwargs: Dict[str, Any] = Field(default_factory=dict, description="搜索參數")
    type: str = Field(default="chroma", description="檢索器類型")
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        k = self.search_kwargs.get("k", 3)
        results = self.vector_store.search(query, n_results=k)
        
        documents = []
        for result in results:
            doc = Document(
                page_content=result['content'],
                metadata=result['metadata']
            )
            documents.append(doc)
        return documents

class VectorStore(VectorStore):
    def __init__(self, persist_directory: str):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # 初始化 ChromaDB 客戶端
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # 創建或獲取集合
        self.collection = self.client.get_or_create_collection(
            name="audio_transcripts",
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_content(self, content: str, metadata: Dict[str, Any]) -> bool:
        """添加內容到向量存儲"""
        try:
            # 生成文檔ID
            doc_id = f"doc_{len(self.collection.get()['ids'])}"
            
            # 添加文檔到集合
            self.collection.add(
                documents=[content],
                metadatas=[metadata],
                ids=[doc_id]
            )
            return True
        except Exception as e:
            print(f"添加內容失敗: {str(e)}")
            return False
    
    def search(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """搜索相似內容"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            # 格式化結果
            formatted_results = []
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None
                })
            
            return formatted_results
        except Exception as e:
            print(f"搜索失敗: {str(e)}")
            return []
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        """獲取所有文檔"""
        try:
            results = self.collection.get()
            documents = []
            for i in range(len(results['ids'])):
                documents.append({
                    'id': results['ids'][i],
                    'content': results['documents'][i],
                    'metadata': results['metadatas'][i]
                })
            return documents
        except Exception as e:
            print(f"獲取文檔失敗: {str(e)}")
            return []
            
    def as_retriever(self, search_type: str = "similarity", search_kwargs: dict = None) -> BaseRetriever:
        """實現 LangChain 的檢索器接口"""
        return ChromaRetriever(
            vector_store=self,
            search_type=search_type,
            search_kwargs=search_kwargs or {}
        )
        
    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> "VectorStore":
        """從文本列表創建向量存儲"""
        persist_directory = kwargs.get("persist_directory", "vector_store")
        instance = cls(persist_directory)
        
        # 添加文本到存儲
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas else {}
            instance.add_content(text, metadata)
            
        return instance
        
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Document]:
        """執行相似度搜索"""
        results = self.search(query, n_results=k)
        return [
            Document(
                page_content=result['content'],
                metadata=result['metadata']
            )
            for result in results
        ] 