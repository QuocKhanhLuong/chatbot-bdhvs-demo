#!/usr/bin/env python3
"""
PDF Ingestion Script for RAG Knowledge Base
Extracts text from PDF files and chunks them for better retrieval
"""

import os
import json
import sys
from pathlib import Path

try:
    import PyPDF2
except ImportError:
    print("❌ PyPDF2 not found. Installing...")
    os.system(f"{sys.executable} -m pip install PyPDF2")
    import PyPDF2


def chunk_text(text, max_chunk_size=800):
    """
    Chunk text into smaller pieces for better RAG accuracy
    
    Args:
        text: Full text to chunk
        max_chunk_size: Maximum characters per chunk
        
    Returns:
        List of text chunks
    """
    chunks = []
    paragraphs = text.split('\n\n')
    
    current_chunk = ''
    
    for paragraph in paragraphs:
        # If adding this paragraph would exceed max size, save current chunk
        if len(current_chunk) + len(paragraph) > max_chunk_size and len(current_chunk) > 0:
            chunks.append(current_chunk.strip())
            current_chunk = ''
        
        # If single paragraph is too long, split by sentences
        if len(paragraph) > max_chunk_size:
            sentences = paragraph.replace('! ', '!|').replace('? ', '?|').replace('. ', '.|').split('|')
            for sentence in sentences:
                if len(current_chunk) + len(sentence) > max_chunk_size and len(current_chunk) > 0:
                    chunks.append(current_chunk.strip())
                    current_chunk = ''
                current_chunk += sentence + ' '
        else:
            current_chunk += paragraph + '\n\n'
    
    # Add remaining chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # Filter out very small chunks
    return [chunk for chunk in chunks if len(chunk) > 50]


def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Extracted text as string
    """
    text = ''
    
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Extract text from all pages
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + '\n\n'
                except Exception as e:
                    print(f"  ⚠️  Warning: Could not extract text from page {page_num + 1}: {e}")
                    continue
        
        return text
    except Exception as e:
        raise Exception(f"Failed to read PDF: {e}")


def ingest_pdfs():
    """Main ingestion function"""
    print('🚀 Starting PDF ingestion...\n')
    
    # Setup paths
    script_dir = Path(__file__).parent
    pdf_dir = script_dir.parent / 'data' / 'pdf'
    output_path = script_dir.parent / 'utils' / 'knowledge_base.json'
    
    # Check if PDF directory exists
    if not pdf_dir.exists():
        print(f'❌ PDF directory not found: {pdf_dir}')
        print('Creating data/pdf directory...')
        pdf_dir.mkdir(parents=True, exist_ok=True)
        print('✅ Please place your PDF files in data/pdf/ and run this script again.')
        return
    
    # Get all PDF files
    pdf_files = list(pdf_dir.glob('*.pdf'))
    
    if not pdf_files:
        print('❌ No PDF files found in data/pdf/')
        return
    
    print(f'📚 Found {len(pdf_files)} PDF files:')
    for pdf_file in pdf_files:
        print(f'  - {pdf_file.name}')
    print()
    
    knowledge_base = []
    chunk_id = 1
    
    # Process each PDF
    for pdf_file in pdf_files:
        try:
            print(f'📄 Processing: {pdf_file.name}...')
            
            # Extract text
            text = extract_text_from_pdf(pdf_file)
            print(f'  ✓ Extracted {len(text)} characters')
            
            if not text.strip():
                print(f'  ⚠️  Warning: No text extracted from {pdf_file.name}')
                continue
            
            # Chunk the text
            chunks = chunk_text(text)
            print(f'  ✓ Created {len(chunks)} chunks')
            
            # Add to knowledge base
            for chunk in chunks:
                knowledge_base.append({
                    'id': str(chunk_id),
                    'source': pdf_file.name,
                    'content': chunk
                })
                chunk_id += 1
            
            print(f'  ✅ Successfully processed {pdf_file.name}\n')
            
        except Exception as e:
            print(f'  ❌ Error processing {pdf_file.name}: {e}\n')
            continue
    
    # Save to JSON
    print(f'💾 Saving {len(knowledge_base)} chunks to {output_path}...')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(knowledge_base, f, ensure_ascii=False, indent=2)
    
    print('✅ Ingestion complete!')
    print(f'\n📊 Summary:')
    print(f'  - PDFs processed: {len(pdf_files)}')
    print(f'  - Total chunks: {len(knowledge_base)}')
    print(f'  - Output file: {output_path}')


if __name__ == '__main__':
    try:
        ingest_pdfs()
    except Exception as e:
        print(f'❌ Fatal error: {e}')
        sys.exit(1)
