"""
RAG Engine for Exam/Document Q&A
Uses LangChain + Google Gemini + FAISS for intelligent retrieval
"""

import os
from pathlib import Path
from typing import List, Dict
import logging
from pydantic import SecretStr

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class ExamRAG:
    """
    Retrieval-Augmented Generation engine for exam/document questions
    """
    
    def __init__(self, pdf_directory: str = "../data/pdf", index_path: str = "./faiss_index"):
        """
        Initialize the RAG engine
        
        Args:
            pdf_directory: Path to directory containing PDF files
            index_path: Path to save/load FAISS index
        """
        self.pdf_directory = Path(pdf_directory)
        self.index_path = Path(index_path)
        self.vectorstore = None
        self.qa_chain = None
        
        # Get API key from environment
        self.megallm_api_key = os.getenv("MEGALLM_API_KEY")
        
        if not self.megallm_api_key:
            raise ValueError("MEGALLM_API_KEY must be set in environment")
        
        logger.info(f"Initialized ExamRAG with PDF directory: {self.pdf_directory}")
        logger.info("Using MegaLLM for both LLM and Embeddings")
    
    def load_pdfs(self) -> List:
        """
        Load all PDF files from the specified directory
        
        Returns:
            List of Document objects
        """
        logger.info(f"Loading PDFs from {self.pdf_directory}...")
        
        if not self.pdf_directory.exists():
            raise FileNotFoundError(f"PDF directory not found: {self.pdf_directory}")
        
        pdf_files = list(self.pdf_directory.glob("*.pdf"))
        
        if not pdf_files:
            raise ValueError(f"No PDF files found in {self.pdf_directory}")
        
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        all_documents = []
        for pdf_file in pdf_files:
            try:
                logger.info(f"Loading {pdf_file.name}...")
                loader = PyPDFLoader(str(pdf_file))
                documents = loader.load()
                
                # Add metadata
                for doc in documents:
                    doc.metadata["source_file"] = pdf_file.name
                
                all_documents.extend(documents)
                logger.info(f"  ✓ Loaded {len(documents)} pages from {pdf_file.name}")
            except Exception as e:
                logger.error(f"  ✗ Error loading {pdf_file.name}: {e}")
                continue
        
        logger.info(f"Total documents loaded: {len(all_documents)}")
        return all_documents
    
    def split_documents(self, documents: List) -> List:
        """
        Split documents into chunks for better retrieval
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of chunked Document objects
        """
        logger.info("Splitting documents into chunks...")
        
        # Use RecursiveCharacterTextSplitter for better chunk quality
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks")
        
        return chunks
    
    def build_vectorstore(self, chunks: List):
        """
        Build FAISS vectorstore from document chunks
        
        Args:
            chunks: List of document chunks to embed
        """
        logger.info("Building vector store with FastEmbed (local, no API needed)...")
        
        # Initialize FastEmbed - fast, local embeddings
        embeddings = FastEmbedEmbeddings(
            model_name="BAAI/bge-small-en-v1.5"
        )
        
        # Create FAISS vectorstore
        self.vectorstore = FAISS.from_documents(chunks, embeddings)
        
        # Save index to disk
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.vectorstore.save_local(str(self.index_path))
        logger.info(f"Vector store built and saved to {self.index_path}")
    
    def load_vectorstore(self):
        """
        Load existing FAISS vectorstore from disk
        """
        if not self.index_path.exists():
            raise FileNotFoundError(f"Vector store not found at {self.index_path}")
        
        logger.info(f"Loading vector store from {self.index_path}...")
        
        # Use same embeddings as build_vectorstore
        embeddings = FastEmbedEmbeddings(
            model_name="BAAI/bge-small-en-v1.5"
        )
        
        self.vectorstore = FAISS.load_local(
            str(self.index_path),
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        logger.info("Vector store loaded successfully")
    
    def initialize(self, force_rebuild: bool = False):
        """
        Initialize the RAG system (load or build vectorstore)
        
        Args:
            force_rebuild: If True, rebuild vectorstore from scratch
        """
        logger.info("Initializing RAG engine...")
        
        # Try to load existing index, or build new one
        if not force_rebuild and self.index_path.exists():
            try:
                self.load_vectorstore()
            except Exception as e:
                logger.warning(f"Failed to load existing index: {e}. Building new one...")
                force_rebuild = True
        
        if force_rebuild or not self.index_path.exists():
            # Load and process PDFs
            documents = self.load_pdfs()
            chunks = self.split_documents(documents)
            self.build_vectorstore(chunks)
        
        # Setup QA chain
        self.setup_qa_chain()
        
        logger.info("RAG engine initialized successfully!")
    
    def setup_qa_chain(self):
        """
        Setup the Question-Answering chain with custom prompt
        """
        logger.info("Setting up QA chain with Llama 3.3 70B via MegaLLM...")
        
        # Initialize LLM - Llama 3.3 70B via MegaLLM
        llm = ChatOpenAI(
            model="llama3.3-70b-instruct",  
            api_key=self.megallm_api_key,
            base_url="https://ai.megallm.io/v1",
            temperature=0.3
        )
        
        # Custom prompt template in Vietnamese
        prompt_template = """Bạn là trợ lý AI thông minh, chuyên trả lời câu hỏi dựa trên tài liệu được cung cấp.

Ngữ cảnh từ tài liệu:
{context}

Câu hỏi: {question}

Hướng dẫn trả lời:
- Trả lời bằng tiếng Việt một cách rõ ràng và chính xác
- Dựa vào thông tin trong ngữ cảnh được cung cấp
- Nếu không tìm thấy thông tin trong tài liệu, hãy nói rõ "Tôi không tìm thấy thông tin này trong tài liệu"
- Trả lời ngắn gọn nhưng đầy đủ
- Nếu có thể, trích dẫn nguồn từ tài liệu

Câu trả lời:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create retrieval chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}  # Retrieve top 3 chunks
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        logger.info("QA chain setup complete")
    
    def get_answer(self, query: str) -> Dict[str, any]:
        """
        Get answer for a user query using RAG
        
        Args:
            query: User's question
            
        Returns:
            Dict with 'answer' and 'sources'
        """
        if not self.qa_chain:
            raise RuntimeError("RAG engine not initialized. Call initialize() first.")
        
        logger.info(f"Processing query: {query[:100]}...")
        
        try:
            # Get answer from QA chain
            result = self.qa_chain.invoke({"query": query})
            
            # Extract source information
            sources = []
            for doc in result.get("source_documents", []):
                sources.append({
                    "file": doc.metadata.get("source_file", "Unknown"),
                    "page": doc.metadata.get("page", "Unknown"),
                    "content_preview": doc.page_content[:200] + "..."
                })
            
            response = {
                "answer": result["result"],
                "sources": sources,
                "num_sources": len(sources)
            }
            
            logger.info(f"Answer generated with {len(sources)} sources")
            return response
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise
    
    def get_relevant_chunks(self, query: str, k: int = 3) -> List[Dict]:
        """
        Get relevant document chunks without generating answer
        
        Args:
            query: User's question
            k: Number of chunks to retrieve
            
        Returns:
            List of relevant chunks with metadata
        """
        if not self.vectorstore:
            raise RuntimeError("Vector store not initialized")
        
        docs = self.vectorstore.similarity_search(query, k=k)
        
        chunks = []
        for doc in docs:
            chunks.append({
                "content": doc.page_content,
                "source_file": doc.metadata.get("source_file", "Unknown"),
                "page": doc.metadata.get("page", "Unknown")
            })
        
        return chunks


if __name__ == "__main__":
    # Test the RAG engine
    print("Testing RAG Engine...")
    
    rag = ExamRAG()
    rag.initialize(force_rebuild=True)
    
    # Test query
    test_query = "Làm thế nào để tạo khóa học trên hệ thống?"
    result = rag.get_answer(test_query)
    
    print(f"\nQuery: {test_query}")
    print(f"\nAnswer: {result['answer']}")
    print(f"\nSources: {result['num_sources']} documents")
    for i, source in enumerate(result['sources'], 1):
        print(f"\n  {i}. {source['file']} (Page {source['page']})")
        print(f"     {source['content_preview']}")
