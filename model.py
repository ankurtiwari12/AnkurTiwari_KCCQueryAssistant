import requests
import json
from typing import Dict, List, Optional, Tuple
from vector_store import VectorStore, DocumentProcessor
import time
import os

class OllamaModel:
    def __init__(self, model_name: str = "llama2", vector_store_path: str = "vector_store.pkl"):
        self.model_name = model_name
        self.base_url = "http://localhost:11434/api"
        self.vector_store = VectorStore(vector_store_path=vector_store_path)
        self.doc_processor = DocumentProcessor()
        self.is_rag_initialized = False
        # Try to load vector store at initialization
        if os.path.exists(vector_store_path):
            if self.vector_store.load():
                self.is_rag_initialized = True
                print("Vector store loaded during initialization")
        
    def initialize_rag(self, df, text_columns: List[str]):
        """Initialize RAG with documents from dataframe"""
        start_time = time.time()
        
        try:
            # Check if vector store already loaded during init
            if self.is_rag_initialized:
                print("RAG system already initialized")
                return
            
            # Try to load existing vector store
            if os.path.exists(self.vector_store.vector_store_path):
                print("Loading existing vector store...")
                if self.vector_store.load():
                    print("Vector store loaded successfully")
                    self.is_rag_initialized = True
                    print(f"RAG system initialized in {time.time() - start_time:.2f} seconds")
                    return
                else:
                    print("Failed to load vector store, recreating...")
            
            # Process documents and create new vector store
            print("Processing documents...")
            documents = self.doc_processor.process_dataframe(df, text_columns)
            if not documents:
                raise ValueError(f"No valid documents created from columns: {text_columns}")
            
            print(f"Created {len(documents)} document chunks")
            self.vector_store.add_documents(documents, use_quantization=True)
            self.is_rag_initialized = True
            print(f"RAG system initialized in {time.time() - start_time:.2f} seconds")
            
        except Exception as e:
            print(f"Error initializing RAG: {str(e)}")
            self.is_rag_initialized = False
            raise
        
    def generate_response(self, prompt: str, system_prompt: Optional[str] = None, use_rag: bool = True, k: int = 3) -> Tuple[str, Dict]:
        """
        Generate a response using the Ollama model with RAG
        Returns:
            Tuple[str, Dict]: (response, timing_info)
        """
        timing_info = {}
        total_start_time = time.time()
        url = f"{self.base_url}/generate"
        
        if use_rag:
            if not self.is_rag_initialized:
                print("Warning: RAG system is not initialized. Falling back to regular generation.")
                timing_info['rag_retrieval_time'] = 0
            else:
                try:
                    # Get relevant documents
                    rag_start_time = time.time()
                    relevant_docs = self.vector_store.similarity_search(prompt, k=k)
                    
                    # Create context from relevant documents
                    context = "\n\n".join([doc for doc, _ in relevant_docs])
                    print(f"Retrieved {len(relevant_docs)} relevant documents")
                    
                    # Create RAG prompt
                    rag_prompt = f"""Context information is below:
                    {context}

                    Given the context information and no other information, answer the following query:
                    {prompt}
                    """
                    
                    prompt = rag_prompt
                    timing_info['rag_retrieval_time'] = time.time() - rag_start_time
                except Exception as e:
                    print(f"Warning: RAG retrieval failed: {e}. Falling back to regular generation.")
                    timing_info['rag_retrieval_time'] = 0
        
        # Generate response
        generation_start_time = time.time()
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }
        
        if system_prompt:
            payload["system"] = system_prompt
            
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            result = response.json()["response"]
        except requests.exceptions.RequestException as e:
            print(f"Error calling Ollama API: {e}")
            result = ""
            
        timing_info['generation_time'] = time.time() - generation_start_time
        timing_info['total_time'] = time.time() - total_start_time
            
        return result, timing_info
            
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model
        """
        url = f"{self.base_url}/show"
        
        try:
            response = requests.post(url, json={"name": self.model_name})
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error getting model info: {e}")
            return {} 