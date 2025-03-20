# main.py
# Aplikasi terminal sederhana untuk RAG

import os
import pickle
import faiss
import numpy as np
import google.generativeai as genai
from typing import List, Dict, Any

# Konfigurasi API key
GOOGLE_API_KEY = "AIzaSyBTtRe2IA3o14lxAMOsO83Xhsy2KxSIlMg"  # Ganti dengan API key Anda
genai.configure(api_key=GOOGLE_API_KEY)

class SimpleRAG:
    def __init__(self, vector_store_path: str = "data/vector_store"):
        self.vector_store_path = vector_store_path
        self.embedding_model = "models/text-embedding-004"
        self.generation_model = "gemini-2.0-flash"
        
        # Load vector store
        self.load_vector_store()
        
        # Initialize generation model
        self.model = genai.GenerativeModel(model_name=self.generation_model)
    
    def load_vector_store(self):
        """Load the vector store"""
        if not os.path.exists(f"{self.vector_store_path}.index"):
            raise FileNotFoundError(f"Vector store not found at {self.vector_store_path}.index")
        
        if not os.path.exists(f"{self.vector_store_path}.pkl"):
            raise FileNotFoundError(f"Vector store data not found at {self.vector_store_path}.pkl")
        
        # Load the index
        self.index = faiss.read_index(f"{self.vector_store_path}.index")
        
        # Load the documents
        with open(f"{self.vector_store_path}.pkl", "rb") as f:
            data = pickle.load(f)
            self.documents = data["documents"]
            self.chunks_info = data["chunks_info"]
        
        print(f"Vector store berhasil dimuat: {len(self.chunks_info)} chunks dari {len(self.documents)} dokumen")
    
    def query(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Query the vector store and return the top k relevant chunks"""
        # Create embedding for the query
        query_embedding = genai.embed_content(
            model=self.embedding_model,
            content=query,
            task_type="retrieval_query"
        )
        
        # Convert to numpy array
        query_vector = np.array([query_embedding["embedding"]]).astype('float32')
        
        # Search the index
        distances, indices = self.index.search(query_vector, top_k)
        
        # Gather results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:  # -1 means no result
                chunk_info = self.chunks_info[idx]
                results.append({
                    "id": chunk_info["doc_id"],
                    "chunk_idx": chunk_info["chunk_idx"],
                    "text": chunk_info["text"],
                    "score": float(distances[0][i])
                })
        
        return results
    
    def generate_response(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Generate a response based on the query and retrieved chunks"""
        # Prepare context from chunks
        context = ""
        for i, chunk in enumerate(context_chunks):
            context += f"\nChunk {i+1} (dari {chunk['id']}):\n{chunk['text']}\n"
        
        # Prepare prompt
        prompt = f"""
        Berdasarkan informasi berikut, jawablah pertanyaan pengguna.
        Jika jawabannya tidak ada dalam informasi yang diberikan, katakan bahwa Anda tidak memiliki cukup informasi.
        
        Informasi:
        {context}
        
        Pertanyaan pengguna: {query}
        
        Jawaban:
        """
        
        # Generate response
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error saat menghasilkan respons: {str(e)}"
    
    def run_cli(self):
        """Run the CLI interface"""
        print("=" * 50)
        print("Aplikasi RAG Terminal Sederhana")
        print("=" * 50)
        print("Ketik 'exit' atau 'quit' untuk keluar\n")
        
        while True:
            query = input("\nMasukkan pertanyaan Anda: ")
            
            if query.lower() in ['exit', 'quit']:
                print("\nTerima kasih telah menggunakan aplikasi ini!")
                break
            
            if not query.strip():
                continue
            
            # Retrieve relevant chunks
            print("Mencari informasi relevan...")
            results = self.query(query)
            
            if not results:
                print("Tidak ditemukan informasi yang relevan.")
                continue
            
            # Generate response
            print("Menghasilkan respons...")
            response = self.generate_response(query, results)
            
            # Display response
            print("\n" + "=" * 50)
            print("JAWABAN:")
            print(response)
            print("=" * 50)
            
            # Ask if user wants to see sources
            show_sources = input("\nIngin melihat sumber? (y/n): ").lower()
            if show_sources == 'y':
                print("\nSUMBER INFORMASI:")
                for i, chunk in enumerate(results):
                    print(f"\n--- Sumber {i+1}: {chunk['id']} ---")
                    print(f"Score: {chunk['score']:.4f}")
                    print(chunk['text'][:200] + "...")

if __name__ == "__main__":
    try:
        rag = SimpleRAG()
        rag.run_cli()
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
        print("\nPastikan file vector store (.index dan .pkl) tersedia di folder data/")
        print("Jalankan document_processor.ipynb di Google Colab terlebih dahulu untuk membuat vector store.")
    except Exception as e:
        print(f"Error: {str(e)}")