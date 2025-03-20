# main.py
# Aplikasi terminal sederhana untuk RAG dengan Function Calling (compatible version)

import os
import pickle
import faiss
import numpy as np
import google.generativeai as genai
import json
from typing import List, Dict, Any, Callable, Optional
from dotenv import load_dotenv
import datetime
import webbrowser
import re
import sys

# Try to import rich for better terminal output, but fall back if not available
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown
    console = Console()
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    # Create a simple console replacement
    class SimpleConsole:
        def print(self, *args, **kwargs):
            # Strip any formatting tags if present
            text = str(args[0])
            text = re.sub(r'\[.*?\]', '', text)
            print(text)
        
        def input(self, prompt):
            # Strip any formatting tags
            prompt = re.sub(r'\[.*?\]', '', prompt)
            return input(prompt)
    
    console = SimpleConsole()

# Load environment variables from .env file if it exists
load_dotenv()

# Konfigurasi API key
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")  # Ganti dengan API key Anda
if not GOOGLE_API_KEY:
    console.print("Error: API key tidak ditemukan. Set GOOGLE_API_KEY di file .env")
    sys.exit(1)
    
genai.configure(api_key=GOOGLE_API_KEY)

class SimpleRAG:
    def __init__(self, vector_store_path: str = "data/vector_store"):
        self.vector_store_path = vector_store_path
        self.embedding_model = "models/text-embedding-004"
        self.generation_model = "gemini-2.0-flash"  # atau "gemini-2.0-pro" untuk kualitas lebih tinggi
        
        # Load vector store
        self.load_vector_store()
        
        # Initialize generation model
        self.model = genai.GenerativeModel(
            model_name=self.generation_model,
            generation_config={
                "temperature": 0.2,
                "top_p": 0.8,
                "top_k": 40,
                "max_output_tokens": 2048,
            }
        )
        
        # Register function calls
        self.registered_functions = self._register_functions()
    
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
        
        console.print(f"Vector store berhasil dimuat: {len(self.chunks_info)} chunks dari {len(self.documents)} dokumen")
    
    def _register_functions(self) -> Dict[str, Dict[str, Any]]:
        """Register available functions for the model to call"""
        functions = {
            "search_documents": {
                "function": self.search_documents,
                "description": "Mencari dokumen yang relevan dengan kueri",
                "parameters": ["query", "top_k"]
            },
            "get_current_time": {
                "function": self.get_current_time,
                "description": "Mendapatkan waktu dan tanggal saat ini",
                "parameters": []
            },
            "open_browser": {
                "function": self.open_browser,
                "description": "Membuka browser dengan URL yang ditentukan",
                "parameters": ["url"]
            },
            "summarize_document": {
                "function": self.summarize_document,
                "description": "Meringkas dokumen tertentu",
                "parameters": ["doc_id"]
            },
            "list_available_documents": {
                "function": self.list_available_documents,
                "description": "Menampilkan daftar dokumen yang tersedia",
                "parameters": []
            }
        }
        
        return functions
    
    def search_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
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
    
    def get_current_time(self) -> Dict[str, str]:
        """Get the current time and date"""
        now = datetime.datetime.now()
        return {
            "time": now.strftime("%H:%M:%S"),
            "date": now.strftime("%Y-%m-%d"),
            "day_of_week": now.strftime("%A"),
            "timestamp": now.isoformat()
        }
    
    def open_browser(self, url: str) -> Dict[str, str]:
        """Open a browser with the given URL"""
        try:
            # Add http:// if missing
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
                
            webbrowser.open(url)
            return {"status": "success", "message": f"Browser dibuka dengan URL: {url}"}
        except Exception as e:
            return {"status": "error", "message": f"Gagal membuka browser: {str(e)}"}
    
    def summarize_document(self, doc_id: str) -> Dict[str, Any]:
        """Summarize a specific document"""
        # Find the document
        doc = None
        for document in self.documents:
            if document["id"] == doc_id:
                doc = document
                break
        
        if not doc:
            return {"status": "error", "message": f"Dokumen dengan ID {doc_id} tidak ditemukan"}
        
        # Get the full text
        full_text = doc["text"]
        
        # Summarize using Gemini
        try:
            prompt = f"""
            Ringkaslah dokumen berikut dalam beberapa poin utama:
            
            {full_text[:8000]}  # Batasi panjang untuk mencegah token overflow
            
            Berikan ringkasan yang mencakup poin-poin utama dokumen.
            """
            
            summary_response = self.model.generate_content(prompt)
            
            return {
                "status": "success",
                "doc_id": doc_id,
                "summary": summary_response.text,
                "characters": len(full_text),
                "source": doc["source"]
            }
        except Exception as e:
            return {"status": "error", "message": f"Gagal meringkas dokumen: {str(e)}"}
    
    def list_available_documents(self) -> Dict[str, Any]:
        """List all available documents"""
        doc_list = []
        for doc in self.documents:
            doc_list.append({
                "id": doc["id"],
                "source": doc["source"],
                "size": len(doc["text"]),
                "chunks": doc.get("chunks_count", 0)
            })
        
        return {
            "status": "success",
            "count": len(doc_list),
            "documents": doc_list
        }
    
    def parse_function_call(self, response_text: str) -> Dict[str, Any]:
        """
        Parse function calls from model response text
        Returns a dict with function name and arguments if a function call is detected
        """
        # Look for function call patterns in the response
        function_call_pattern = r'FUNCTION_CALL\[(.*?)\]\((.*?)\)'
        matches = re.findall(function_call_pattern, response_text)
        
        if matches:
            function_name, args_str = matches[0]
            
            # Try to parse args as comma-separated values
            args = {}
            if args_str:
                arg_pairs = args_str.split(',')
                for pair in arg_pairs:
                    if ':' in pair:
                        key, value = pair.split(':', 1)
                        args[key.strip()] = value.strip()
                    else:
                        # Single argument without key
                        args["value"] = pair.strip()
            
            return {
                "detected": True,
                "function": function_name.strip(),
                "args": args
            }
        
        return {"detected": False}
    
    def execute_function(self, function_name: str, args: Dict[str, Any]) -> Any:
        """Execute a function by name with the provided arguments"""
        if function_name not in self.registered_functions:
            return {"error": f"Function {function_name} not found"}
        
        function_info = self.registered_functions[function_name]
        function_to_call = function_info["function"]
        
        # Convert string args to proper types if needed
        processed_args = {}
        
        # Only process parameters that are defined for this function
        for param in function_info["parameters"]:
            if param in args:
                # Try to convert to int if it looks like a number
                if args[param].isdigit():
                    processed_args[param] = int(args[param])
                else:
                    processed_args[param] = args[param]
        
        # Call the function with the processed arguments
        try:
            return function_to_call(**processed_args)
        except Exception as e:
            return {"error": f"Error executing function: {str(e)}"}
    
    def generate_response(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Generate a response based on the query and retrieved chunks"""
        # Prepare context from chunks
        context = ""
        for i, chunk in enumerate(context_chunks):
            context += f"\nChunk {i+1} (dari {chunk['id']}):\n{chunk['text']}\n"
        
        # List available functions for the model
        function_descriptions = "\n".join([
            f"- {name}: {info['description']} (Parameters: {', '.join(info['parameters'])})"
            for name, info in self.registered_functions.items()
        ])
        
        # Prepare prompt with functions
        prompt = f"""
        Berdasarkan informasi berikut, jawablah pertanyaan pengguna.
        Jika jawabannya tidak ada dalam informasi yang diberikan atau kamu perlu informasi tambahan,
        kamu dapat menggunakan salah satu fungsi berikut dengan format FUNCTION_CALL[nama_fungsi](parameter1:nilai1, parameter2:nilai2).
        
        Fungsi yang tersedia:
        {function_descriptions}
        
        Contoh penggunaan:
        FUNCTION_CALL[search_documents](query:artificial intelligence, top_k:3)
        FUNCTION_CALL[get_current_time]()
        FUNCTION_CALL[open_browser](url:google.com)
        
        Informasi:
        {context}
        
        Pertanyaan pengguna: {query}
        
        Jawaban:
        """
        
        # Generate response
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text
            
            # Check if there's a function call in the response
            function_result = self.parse_function_call(response_text)
            
            if function_result["detected"]:
                function_name = function_result["function"]
                args = function_result["args"]
                
                console.print(f"Eksekusi fungsi: {function_name}")
                
                # Execute the function
                result = self.execute_function(function_name, args)
                
                # Format the result for display
                result_str = json.dumps(result, indent=2, ensure_ascii=False)
                
                # Generate a new response with the function result
                follow_up_prompt = f"""
                Kamu telah memanggil fungsi {function_name} untuk pertanyaan: "{query}"
                
                Hasil dari fungsi tersebut adalah:
                {result_str}
                
                Berdasarkan hasil tersebut dan konteks sebelumnya, berikan jawaban final untuk pengguna.
                Jangan disebutkan bahwa kamu menggunakan fungsi, cukup berikan jawaban final saja.
                """
                
                follow_up_response = self.model.generate_content(follow_up_prompt)
                return follow_up_response.text
            else:
                # Just return the original response if no function call
                return response_text
                
        except Exception as e:
            return f"Error saat menghasilkan respons: {str(e)}"
    
    def run_cli(self):
        """Run the CLI interface"""
        if RICH_AVAILABLE:
            console.print(Panel.fit(
                "Aplikasi RAG Terminal dengan Function Calling",
                subtitle="Ketik 'exit' atau 'quit' untuk keluar"
            ))
        else:
            console.print("=" * 50)
            console.print("Aplikasi RAG Terminal dengan Function Calling")
            console.print("=" * 50)
            console.print("Ketik 'exit' atau 'quit' untuk keluar\n")
        
        while True:
            query = console.input("\nMasukkan pertanyaan Anda: ")
            
            if query.lower() in ['exit', 'quit']:
                console.print("\nTerima kasih telah menggunakan aplikasi ini!")
                break
            
            if not query.strip():
                continue
            
            # Retrieve relevant chunks
            console.print("Mencari informasi relevan...")
            results = self.search_documents(query)
            
            if not results:
                console.print("Tidak ditemukan informasi yang relevan.")
                # Still generate a response but without context
                results = []
            
            # Generate response
            console.print("Menghasilkan respons...")
            response = self.generate_response(query, results)
            
            # Display response
            if RICH_AVAILABLE:
                console.print(Panel(
                    Markdown(response),
                    title="JAWABAN",
                    expand=False
                ))
            else:
                console.print("\n" + "=" * 50)
                console.print("JAWABAN:")
                console.print(response)
                console.print("=" * 50)
            
            # Ask if user wants to see sources
            if results:
                show_sources = console.input("\nIngin melihat sumber? (y/n): ").lower()
                if show_sources == 'y':
                    console.print("\nSUMBER INFORMASI:")
                    for i, chunk in enumerate(results):
                        if RICH_AVAILABLE:
                            console.print(Panel(
                                f"{chunk['text'][:300]}...",
                                title=f"Sumber {i+1}: {chunk['id']} (Score: {chunk['score']:.4f})",
                                expand=False
                            ))
                        else:
                            console.print(f"\n--- Sumber {i+1}: {chunk['id']} ---")
                            console.print(f"Score: {chunk['score']:.4f}")
                            console.print(chunk['text'][:200] + "...")

if __name__ == "__main__":
    try:
        rag = SimpleRAG()
        rag.run_cli()
    except FileNotFoundError as e:
        console.print(f"Error: {str(e)}")
        console.print("\nPastikan file vector store (.index dan .pkl) tersedia di folder data/")
        console.print("Jalankan train.py terlebih dahulu untuk membuat vector store.")
    except Exception as e:
        console.print(f"Error: {str(e)}")