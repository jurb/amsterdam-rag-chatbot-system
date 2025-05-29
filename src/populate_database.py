import argparse
import os
import re
import shutil
from typing import List

from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

import config as cfg
from helpers.embedding_helpers import OpenAIEmbeddingFunction

class TXTDirectoryLoader:
    """Loader class to load and preprocess TXT files from a directory."""

    def __init__(self, path: str):
        """
        Initialize the TXTDirectoryLoader with the directory path.
        
        Args:
            path (str): Path to the directory containing TXT files.
        """
        self.path = path

    def load(self) -> List[Document]:
        """
        Load and preprocess TXT files from the directory, and convert them to Document objects.
        
        Returns:
            List[Document]: List of Document objects containing the preprocessed text and metadata.
        """
        documents = []
        for root, _, files in os.walk(self.path):
            for file in files:
                if file.endswith(".txt"):
                    file_path = os.path.join(root, file)
                    with open(file_path, "r", encoding="utf-8") as f:
                        text = f.read()
                        # Light preprocessing - preserve markdown structure
                        text = self._preprocess_text(text)
                        metadata = {"source": file_path}
                        documents.append(Document(page_content=text, metadata=metadata))
        return documents

    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text while preserving markdown structure.
        
        Args:
            text (str): Raw text to preprocess.
        
        Returns:
            str: Preprocessed text.
        """
        # Remove excessive blank lines (more than 2 consecutive)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Ensure headers have proper spacing (add blank line after headers if missing)
        text = re.sub(r'(^#{1,6}[^\n]+)(\n)(?=[^\n#])', r'\1\n\n', text, flags=re.MULTILINE)
        
        # Clean up link and image patterns (remove internal newlines)
        text = re.sub(r'\[LINK:[^\]]+\]\([^\)]+\)', lambda m: m.group(0).replace('\n', ' '), text)
        text = re.sub(r'\[IMG:[^\]]+\]', lambda m: m.group(0).replace('\n', ' '), text)
        
        # Clean up markdown link patterns
        text = re.sub(r'\[[^\]]+\]\([^\)]+\)', lambda m: m.group(0).replace('\n', ' '), text)
        
        # Remove trailing whitespace from each line
        lines = text.split('\n')
        lines = [line.rstrip() for line in lines]
        text = '\n'.join(lines)
        
        return text.strip()

def load_documents() -> List[Document]:
    """
    Load documents from different formats (PDF, TXT, HTML) in the data directory.
    
    Returns:
        List[Document]: List of loaded Document objects.
    """
    documents = []

    # Load TXT documents
    txt_loader = TXTDirectoryLoader(cfg.DATA_PATH)
    documents.extend(txt_loader.load())
    
    print(f"Loaded {len(documents)} documents")
    
    # Debug: Show sample of first document
    if documents:
        print("\n--- Sample of first document ---")
        print(f"Source: {documents[0].metadata['source']}")
        print(f"First 500 chars:\n{documents[0].page_content[:500]}")
        print("--- End sample ---\n")

    return documents

def split_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into smaller chunks for processing.
    
    Args:
        documents (List[Document]): List of Document objects to split.
    
    Returns:
        List[Document]: List of split Document objects.
    """
    chunks = []
    
    # Try to use MarkdownTextSplitter for better markdown handling
    try:
        text_splitter = MarkdownTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        print("Using MarkdownTextSplitter")
    except:
        # Fallback to RecursiveCharacterTextSplitter with markdown-aware separators
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
            separators=[
                "\n\n",      # Double newline (paragraph break)
                "\n### ",    # H3 headers
                "\n## ",     # H2 headers  
                "\n# ",      # H1 headers
                "\n#### ",   # H4 headers
                "\n",        # Single newline
                ". ",        # Sentence ending
                " ",         # Space
                ""           # Character
            ]
        )
        print("Using RecursiveCharacterTextSplitter with markdown separators")

    for document in documents:
        text_chunks = text_splitter.split_text(document.page_content)
        for idx, chunk in enumerate(text_chunks):
            chunk_doc = Document(page_content=chunk, metadata=document.metadata.copy())
            chunk_doc.metadata["chunk_index"] = idx
            chunk_doc.metadata["total_chunks"] = len(text_chunks)
            chunks.append(chunk_doc)

    print(f"Split into {len(chunks)} chunks")
    
    # Debug: Show some chunk examples
    if chunks:
        print("\n--- Sample chunks ---")
        for i in range(min(3, len(chunks))):
            print(f"\nChunk {i} (length: {len(chunks[i].page_content)}):")
            print(f"First 200 chars: {chunks[i].page_content[:200]}...")
        print("--- End samples ---\n")

    return chunks

def add_to_chroma(chunks: List[Document]):
    """
    Add or update document chunks in the Chroma database.
    
    Args:
        chunks (List[Document]): List of Document objects (chunks) to add.
    """
    # Initialize the database
    db = Chroma(persist_directory=cfg.CHROMA_PATH, embedding_function=OpenAIEmbeddingFunction())
    
    # Calculate chunk IDs
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Get existing items
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Filter new chunks
    new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]

    if new_chunks:
        print(f"Adding new documents: {len(new_chunks)}")
        
        # Add in batches to avoid overwhelming the system
        batch_size = 100
        for i in tqdm(range(0, len(new_chunks), batch_size), desc="Adding document batches"):
            batch = new_chunks[i:i + batch_size]
            batch_ids = [chunk.metadata["id"] for chunk in batch]
            db.add_documents(batch, ids=batch_ids)
            
        print(f"Successfully added {len(new_chunks)} new chunks to the database")
    else:
        print("No new documents to add")
    
    # Verify by searching for "zonnepanelen"
    print("\n--- Verification search for 'zonnepanelen' ---")
    results = db.similarity_search("zonnepanelen", k=3)
    if results:
        print(f"Found {len(results)} results:")
        for i, result in enumerate(results):
            print(f"\nResult {i+1}:")
            print(f"Source: {result.metadata.get('source', 'Unknown')}")
            print(f"Content preview: {result.page_content[:150]}...")
    else:
        print("No results found for 'zonnepanelen' - check your data!")
    print("--- End verification ---\n")

def calculate_chunk_ids(chunks: List[Document]) -> List[Document]:
    """
    Calculate unique IDs for each chunk based on its source and chunk index.
    
    Args:
        chunks (List[Document]): List of Document objects (chunks) to process.
    
    Returns:
        List[Document]: List of Document objects with assigned IDs.
    """
    for chunk in chunks:
        source = chunk.metadata.get("source")
        chunk_index = chunk.metadata.get("chunk_index", 0)
        # Create a unique ID based on source and chunk index
        chunk_id = f"{source}:chunk_{chunk_index}"
        chunk.metadata["id"] = chunk_id

    return chunks

def clear_database():
    """Clear the existing Chroma database by removing the directory."""
    if os.path.exists(cfg.CHROMA_PATH):
        shutil.rmtree(cfg.CHROMA_PATH)
        print("Database cleared successfully")

def main():
    """Main function to parse arguments, load documents, process them, and manage the database."""
    parser = argparse.ArgumentParser(description="Manage the document database.")
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    parser.add_argument("--debug", action="store_true", help="Enable debug output.")
    args = parser.parse_args()
    
    if args.reset:
        print("Clearing Database...")
        clear_database()

    # Load and process documents
    print("Loading documents...")
    documents = load_documents()
    
    if not documents:
        print("No documents found! Check your data directory.")
        return
    
    print("Splitting documents into chunks...")
    chunks = split_documents(documents)
    
    print("Adding chunks to Chroma database...")
    add_to_chroma(chunks)
    
    print("\nDatabase population complete!")

if __name__ == "__main__":
    main()