import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Dict, Tuple
import torch
import pickle
import os

class VectorStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", vector_store_path: str = "vector_store.pkl"):
        self.encoder = SentenceTransformer(model_name)
        self.index = None
        self.documents = []
        self.vector_store_path = vector_store_path
        
    def add_documents(self, documents: List[str], use_quantization: bool = True):
        """Add documents to the vector store with optional quantization"""
        print("Encoding documents...")
        self.documents = documents
        
        # Create embeddings
        embeddings = self.encoder.encode(documents, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')
        
        # Get dimension
        dimension = embeddings.shape[1]
        
        if use_quantization:
            print("Using quantized index...")
            # Use IVF (Inverted File Index) with Product Quantization
            nlist = min(int(len(documents) ** 0.5), 256)  # number of clusters
            quantizer = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIVFPQ(quantizer, dimension, nlist, 8, 8)  # 8 bits per sub-vector
            self.index.train(embeddings)
        else:
            print("Using flat index...")
            self.index = faiss.IndexFlatL2(dimension)
        
        # Add vectors to the index
        self.index.add(embeddings)
        print(f"Added {len(documents)} documents to the vector store")
        
        # Save the vector store
        self.save()
        
    def load(self) -> bool:
        """Load vector store from disk if exists"""
        if os.path.exists(self.vector_store_path):
            print(f"Loading vector store from {self.vector_store_path}")
            with open(self.vector_store_path, 'rb') as f:
                saved_data = pickle.load(f)
                self.index = saved_data['index']
                self.documents = saved_data['documents']
            return True
        return False
    
    def save(self):
        """Save vector store to disk"""
        print(f"Saving vector store to {self.vector_store_path}")
        with open(self.vector_store_path, 'wb') as f:
            pickle.dump({
                'index': self.index,
                'documents': self.documents
            }, f)
        
    def similarity_search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Search for similar documents"""
        if self.index is None:
            raise ValueError("No documents have been added to the vector store")
            
        # Encode query
        query_vector = self.encoder.encode([query])[0].reshape(1, -1).astype('float32')
        
        # Search in FAISS
        distances, indices = self.index.search(query_vector, k)
        
        # Return documents and scores
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.documents):  # Check if index is valid
                results.append((self.documents[idx], float(dist)))
        
        return results

class DocumentProcessor:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def process_dataframe(self, df: pd.DataFrame, text_columns: List[str]) -> List[str]:
        """Process dataframe and create text chunks"""
        documents = []
        
        # Combine specified columns into single text
        for _, row in df.iterrows():
            text = " ".join([str(row[col]) for col in text_columns if col in row])
            if text.strip():  # Only add non-empty chunks
                chunks = self._create_chunks(text)
                documents.extend(chunks)
            
        return documents
    
    def _create_chunks(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = start + self.chunk_size
            chunk = text[start:end].strip()
            if chunk:  # Only add non-empty chunks
                chunks.append(chunk)
            start = end - self.chunk_overlap
            
        return chunks 