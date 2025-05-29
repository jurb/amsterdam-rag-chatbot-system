import argparse
import os
import re
import shutil
import yaml
from typing import List, Dict, Tuple

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
                        
                    # Extract YAML frontmatter and content
                    metadata, content = self._extract_frontmatter(text)
                    
                    # Light preprocessing - preserve markdown structure
                    content = self._preprocess_text(content)
                    
                    # Add source to metadata
                    metadata["source"] = file_path
                    
                    documents.append(Document(page_content=content, metadata=metadata))
        return documents

    def _extract_frontmatter(self, text: str) -> Tuple[Dict, str]:
        """
        Extract YAML frontmatter from text.
        
        Args:
            text (str): Text potentially containing YAML frontmatter.
        
        Returns:
            Tuple[Dict, str]: A tuple of (metadata dict, remaining content).
        """
        # Check if text starts with YAML frontmatter
        if text.startswith('---\n'):
            try:
                # Find the end of frontmatter
                end_match = re.search(r'\n---\n', text[4:])
                if end_match:
                    yaml_text = text[4:end_match.start() + 4]
                    content = text[end_match.end() + 4:].strip()
                    
                    # Parse YAML
                    metadata = yaml.safe_load(yaml_text)
                    if metadata is None:
                        metadata = {}
                    
                    return metadata, content
            except yaml.YAMLError as e:
                print(f"Error parsing YAML frontmatter: {e}")
                # Fall through to return original text
        
        # No frontmatter or parsing error - return empty metadata and original text
        return {}, text

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
            # Create new document with chunk content and copy all metadata
            chunk_doc = Document(page_content=chunk, metadata=document.metadata.copy())
            chunk_doc.metadata["chunk_index"] = idx
            chunk_doc.metadata["total_chunks"] = len(text_chunks)
            chunks.append(chunk_doc)

    print(f"Split into {len(chunks)} chunks")
    
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