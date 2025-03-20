#!/usr/bin/env python3
# train.py - Local version of document processor for vector embeddings

import os
import glob
import zipfile
import pickle
import numpy as np
import faiss
import google.generativeai as genai
from typing import List, Dict, Any
from pypdf import PdfReader
from tqdm import tqdm
import argparse
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

class DocumentProcessor:
    def __init__(self, data_dir: str = "data/sample_docs"):
        self.data_dir = data_dir
        self.documents = []
        self.embeddings = []
        self.chunks_info = []
        self.embedding_model = "models/text-embedding-004"  # Latest embedding model
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
    
    def load_documents(self) -> List[Dict[str, Any]]:
        """Load documents from the data directory"""
        # Look for PDF files in the data directory
        pdf_files = glob.glob(os.path.join(self.data_dir, "*.pdf"))
        
        documents = []
        for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
            document = self._process_pdf(pdf_file)
            documents.append(document)
            
        # Look for text files in the data directory
        txt_files = glob.glob(os.path.join(self.data_dir, "*.txt"))
        
        for txt_file in tqdm(txt_files, desc="Processing TXTs"):
            document = self._process_txt(txt_file)
            documents.append(document)
        
        self.documents = documents
        print(f"Total documents processed: {len(documents)}")
        return documents
    
    def _process_pdf(self, file_path: str) -> Dict[str, Any]:
        """Process a PDF file and extract text"""
        try:
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            
            return {
                "id": os.path.basename(file_path),
                "source": file_path,
                "text": text
            }
        except Exception as e:
            print(f"Error processing PDF {file_path}: {str(e)}")
            return {
                "id": os.path.basename(file_path),
                "source": file_path,
                "text": f"Error processing document: {str(e)}"
            }
    
    def _process_txt(self, file_path: str) -> Dict[str, Any]:
        """Process a text file"""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            
            return {
                "id": os.path.basename(file_path),
                "source": file_path,
                "text": text
            }
        except Exception as e:
            print(f"Error processing text file {file_path}: {str(e)}")
            return {
                "id": os.path.basename(file_path),
                "source": file_path,
                "text": f"Error processing document: {str(e)}"
            }
    
    def _chunk_text(self, text: str, chunk_size: int = 1000) -> List[str]:
        """Split text into chunks of specified size"""
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    
    def create_embeddings(self) -> List[List[float]]:
        """Create embeddings for all documents"""
        if not self.documents:
            self.load_documents()
        
        embeddings = []
        chunks_info = []  # To store chunk information
        
        for doc_idx, doc in enumerate(tqdm(self.documents, desc="Creating embeddings")):
            # Chunk document if it's too large
            chunks = self._chunk_text(doc["text"])
            
            for chunk_idx, chunk in enumerate(chunks):
                try:
                    embedding = genai.embed_content(
                        model=self.embedding_model,
                        content=chunk,
                        task_type="retrieval_document"
                    )
                    embeddings.append(embedding["embedding"])
                    
                    # Save chunk information for better retrieval
                    chunks_info.append({
                        "doc_idx": doc_idx,
                        "doc_id": doc["id"],
                        "chunk_idx": chunk_idx,
                        "text": chunk
                    })
                except Exception as e:
                    print(f"Error on document {doc['id']}, chunk {chunk_idx}: {str(e)}")
                    continue
        
            # For document metadata
            doc["chunks_count"] = len(chunks)
        
        # Save all embeddings as list
        self.embeddings = embeddings
        self.chunks_info = chunks_info
        
        print(f"Total chunks with embeddings: {len(self.embeddings)}")
        return self.embeddings
    
    def save_to_vector_store(self, vector_store_path: str = "data/vector_store"):
        """Save embeddings to a vector store"""
        # Check if we have any embeddings
        if not self.embeddings:
            print("No embeddings found. Creating embeddings first...")
            self.create_embeddings()
            
        # Double-check after creating embeddings
        if not self.embeddings:
            raise ValueError("No documents or embeddings found. Please add documents to the data directory.")
        
        # Create directory for vector store if it doesn't exist
        os.makedirs(os.path.dirname(vector_store_path), exist_ok=True)
        
        # Convert embeddings to numpy array
        embeddings_array = np.array(self.embeddings).astype('float32')
        
        # Create a FAISS index
        dimension = len(self.embeddings[0])
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings_array)
        
        # Save the index
        faiss.write_index(index, f"{vector_store_path}.index")
        
        # Save the documents and chunks info
        with open(f"{vector_store_path}.pkl", "wb") as f:
            data = {
                "documents": self.documents,
                "chunks_info": self.chunks_info
            }
            pickle.dump(data, f)
        
        print(f"Vector store saved to {vector_store_path}.index and {vector_store_path}.pkl")

def extract_zip_files(data_dir):
    """Extract any zip files in the data directory"""
    zip_files = glob.glob(os.path.join(data_dir, "*.zip"))
    for zip_file in zip_files:
        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            print(f"Extracted {zip_file} to {data_dir}")
        except Exception as e:
            print(f"Error extracting {zip_file}: {str(e)}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process documents and create embeddings")
    parser.add_argument("--data-dir", default="data/sample_docs", help="Directory containing documents")
    parser.add_argument("--output", default="data/vector_store", help="Path to save the vector store")
    parser.add_argument("--api-key", help="Google AI API key")
    args = parser.parse_args()
    
    # Set up API key from arguments, environment variable, or prompt
    api_key = args.api_key or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        api_key = input("Enter your Google AI API key: ")
    
    genai.configure(api_key=api_key)
    
    # Extract any zip files in the data directory
    extract_zip_files(args.data_dir)
    
    # Initialize and run document processor
    processor = DocumentProcessor(data_dir=args.data_dir)
    processor.load_documents()
    processor.create_embeddings()
    processor.save_to_vector_store(vector_store_path=args.output)
    
    print("Process completed! Vector store files are available at:")
    print(f" - {args.output}.index")
    print(f" - {args.output}.pkl")

if __name__ == "__main__":
    main()