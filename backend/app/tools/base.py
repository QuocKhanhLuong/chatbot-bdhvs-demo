"""
Research Tools for AI Research Assistant.

Implements tools from analyzed repos:
- Web search (Tavily) - from dzhng/deep-research
- ArXiv paper search and analysis - from linhkid/ArxivDigest-extra  
- Python code execution
- Document retrieval (FAISS)
- Query generation for deep research - from NVIDIA aiq-research-assistant
"""

import os
import asyncio
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Optional, List, Dict, Any
import httpx
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.tools import PythonREPLTool
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.documents import Document

from app.config import settings


# =============================================================================
# Web Search Tools
# =============================================================================

def get_search_tool() -> Optional[TavilySearchResults]:
    """
    Initialize Tavily Search tool for web search capabilities.
    Returns None if API key is not configured.
    """
    if not settings.tavily_api_key:
        print("⚠️ Tavily API key not found. Web search will be disabled.")
        return None
    
    os.environ["TAVILY_API_KEY"] = settings.tavily_api_key
    
    return TavilySearchResults(
        max_results=5,
        search_depth="advanced",
        include_answer=True,
        include_raw_content=True,
        name="web_search",
        description=(
            "Search the web for current information. "
            "Use this when you need up-to-date information about AI research, news, "
            "papers, or any topic that requires recent data. "
            "Input should be a search query string."
        )
    )


@tool
async def search_web_async(query: str, num_results: int = 5) -> str:
    """
    Search the web for information using Tavily (async version).
    Use this for deep research when you need to search multiple queries.
    
    Args:
        query: Search query
        num_results: Number of results to return (1-10)
    
    Returns:
        Search results with titles, URLs, and snippets
    """
    if not settings.tavily_api_key:
        return "Web search is not configured. Please set TAVILY_API_KEY."
    
    os.environ["TAVILY_API_KEY"] = settings.tavily_api_key
    
    search_tool = TavilySearchResults(
        max_results=min(num_results, 10),
        search_depth="advanced",
        include_raw_content=True,
        include_answer=True,
    )
    
    results = await asyncio.to_thread(search_tool.invoke, query)
    
    formatted_results = []
    for i, result in enumerate(results, 1):
        content = result.get('content', 'No content')
        if len(content) > 500:
            content = content[:500] + "..."
        formatted_results.append(
            f"[{i}] **{result.get('title', 'No title')}**\n"
            f"URL: {result.get('url', 'N/A')}\n"
            f"Content: {content}\n"
        )
    
    return "\n---\n".join(formatted_results) if formatted_results else "No results found."


# =============================================================================
# ArXiv Search Tools (from ArxivDigest-extra)
# =============================================================================

@tool
async def search_arxiv(
    query: str, 
    max_results: int = 10,
    sort_by: str = "relevance"
) -> str:
    """
    Search ArXiv for academic papers on AI, ML, and related topics.
    
    Args:
        query: Search query (e.g., "large language models", "RAG retrieval", "transformer attention")
        max_results: Maximum number of papers to return (1-50)
        sort_by: Sort order - "relevance", "lastUpdatedDate", "submittedDate"
    
    Returns:
        List of papers with title, authors, abstract, and URL
    """
    # Build ArXiv API URL
    base_url = "http://export.arxiv.org/api/query"
    
    # Construct search query for AI/CS papers
    search_query = f"all:{query}"
    
    params = {
        "search_query": search_query,
        "start": 0,
        "max_results": min(max_results, 50),
        "sortBy": sort_by,
        "sortOrder": "descending"
    }
    
    url = f"{base_url}?{urllib.parse.urlencode(params)}"
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=30.0)
            response.raise_for_status()
        
        # Parse XML response
        root = ET.fromstring(response.text)
        
        # Define namespace
        ns = {
            'atom': 'http://www.w3.org/2005/Atom',
            'arxiv': 'http://arxiv.org/schemas/atom'
        }
        
        papers = []
        for entry in root.findall('atom:entry', ns):
            title_elem = entry.find('atom:title', ns)
            summary_elem = entry.find('atom:summary', ns)
            published_elem = entry.find('atom:published', ns)
            id_elem = entry.find('atom:id', ns)
            
            # Get authors
            authors = []
            for author in entry.findall('atom:author', ns):
                name_elem = author.find('atom:name', ns)
                if name_elem is not None and name_elem.text:
                    authors.append(name_elem.text)
            
            # Get categories
            categories = [
                cat.get('term', '') 
                for cat in entry.findall('atom:category', ns)
            ]
            
            # Extract paper info
            title = title_elem.text.strip().replace('\n', ' ') if title_elem is not None and title_elem.text else "No title"
            abstract = summary_elem.text.strip().replace('\n', ' ')[:400] if summary_elem is not None and summary_elem.text else "No abstract"
            published = published_elem.text[:10] if published_elem is not None and published_elem.text else ""
            arxiv_url = id_elem.text if id_elem is not None and id_elem.text else ""
            
            papers.append({
                "title": title,
                "authors": authors[:5],
                "abstract": abstract,
                "url": arxiv_url,
                "published": published,
                "categories": categories[:3]
            })
        
        # Format results
        formatted = []
        for i, paper in enumerate(papers, 1):
            authors_str = ", ".join(paper["authors"][:3])
            if len(paper["authors"]) > 3:
                authors_str += " et al."
            
            formatted.append(
                f"[{i}] **{paper['title']}**\n"
                f"Authors: {authors_str}\n"
                f"Published: {paper['published']}\n"
                f"Categories: {', '.join(paper['categories'])}\n"
                f"URL: {paper['url']}\n"
                f"Abstract: {paper['abstract']}...\n"
            )
        
        return "\n---\n".join(formatted) if formatted else "No papers found on ArXiv."
        
    except httpx.TimeoutException:
        return "ArXiv search timed out. Please try again."
    except Exception as e:
        return f"Error searching ArXiv: {str(e)}"


async def search_arxiv_structured(
    query: str,
    max_results: int = 10
) -> List[Dict[str, Any]]:
    """
    Search ArXiv and return structured data (for internal use).
    """
    base_url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": min(max_results, 50),
        "sortBy": "relevance",
        "sortOrder": "descending"
    }
    
    url = f"{base_url}?{urllib.parse.urlencode(params)}"
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=30.0)
            response.raise_for_status()
        
        root = ET.fromstring(response.text)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        
        papers = []
        for entry in root.findall('atom:entry', ns):
            title_elem = entry.find('atom:title', ns)
            summary_elem = entry.find('atom:summary', ns)
            published_elem = entry.find('atom:published', ns)
            id_elem = entry.find('atom:id', ns)
            
            authors = []
            for author in entry.findall('atom:author', ns):
                name_elem = author.find('atom:name', ns)
                if name_elem is not None and name_elem.text:
                    authors.append(name_elem.text)
            
            categories = [cat.get('term', '') for cat in entry.findall('atom:category', ns)]
            
            papers.append({
                "id": id_elem.text.split("/")[-1] if id_elem is not None and id_elem.text else "",
                "title": title_elem.text.strip() if title_elem is not None and title_elem.text else "",
                "authors": authors,
                "abstract": summary_elem.text.strip() if summary_elem is not None and summary_elem.text else "",
                "url": id_elem.text if id_elem is not None else "",
                "published": published_elem.text[:10] if published_elem is not None and published_elem.text else "",
                "categories": categories
            })
        
        return papers
        
    except Exception:
        return []


# =============================================================================
# Python REPL Tool
# =============================================================================

def get_python_repl_tool() -> PythonREPLTool:
    """
    Initialize Python REPL tool for code execution.
    ⚠️ WARNING: This executes arbitrary Python code. Use with caution.
    """
    return PythonREPLTool(
        name="python_repl",
        description=(
            "Execute Python code for calculations, data analysis, or any programming task. "
            "Input should be valid Python code. "
            "Use this for: math calculations, data processing, generating code examples, "
            "or when the user asks you to run/execute code. "
            "Always print() the result you want to return."
        )
    )


@tool
def execute_python(code: str) -> str:
    """
    Execute Python code for data analysis, calculations, or processing.
    Use this when you need to perform calculations, analyze data, or generate visualizations.
    
    Args:
        code: Python code to execute. Always use print() to output results.
    
    Returns:
        Output from code execution
    """
    repl = PythonREPLTool()
    try:
        result = repl.run(code)
        return f"✅ Code executed successfully:\n{result}"
    except Exception as e:
        return f"❌ Error executing code: {str(e)}"


# =============================================================================
# Document Retrieval Tool (FAISS)
# =============================================================================

class DocumentRetrieverTool:
    """
    Tool for retrieving relevant documents from local FAISS vector store.
    """
    
    def __init__(self):
        self.vectorstore: Optional[FAISS] = None
        self.embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        self._load_vectorstore()
    
    def _load_vectorstore(self):
        """Load existing FAISS index if available."""
        index_path = settings.faiss_index_path
        
        if os.path.exists(index_path) and os.path.exists(f"{index_path}/index.faiss"):
            try:
                self.vectorstore = FAISS.load_local(
                    index_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                print(f"✅ Loaded FAISS index from {index_path}")
            except Exception as e:
                print(f"⚠️ Failed to load FAISS index: {e}")
                self.vectorstore = None
        else:
            print(f"ℹ️ No FAISS index found at {index_path}. Document retrieval disabled.")
    
    def search(self, query: str, k: int = 4) -> List[Document]:
        """Search for relevant documents."""
        if not self.vectorstore:
            return []
        
        try:
            docs = self.vectorstore.similarity_search(query, k=k)
            return docs
        except Exception as e:
            print(f"⚠️ Error searching documents: {e}")
            return []


# Initialize document retriever
doc_retriever = DocumentRetrieverTool()


@tool
def search_documents(query: str) -> str:
    """
    Search through local PDF documents and knowledge base.
    Use this when the user asks about information that might be in uploaded documents.
    Input: A search query related to the documents.
    Output: Relevant excerpts from the documents.
    """
    docs = doc_retriever.search(query, k=3)
    
    if not docs:
        return "No relevant documents found in the knowledge base."
    
    results = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        content = doc.page_content[:500]  # Limit content length
        results.append(f"[Doc {i}] Source: {source}\n{content}")
    
    return "\n\n---\n\n".join(results)


def get_all_tools() -> list:
    """
    Get all available tools for the agent.
    Only includes tools that are properly configured.
    """
    tools = []
    
    # Web Search
    search_tool = get_search_tool()
    if search_tool:
        tools.append(search_tool)
    
    # Python REPL
    tools.append(get_python_repl_tool())
    
    # Document Retriever (only if vectorstore is loaded)
    if doc_retriever.vectorstore:
        tools.append(search_documents)
    
    return tools
