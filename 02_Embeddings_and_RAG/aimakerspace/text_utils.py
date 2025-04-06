import os
import fitz  # PyMuPDF
from typing import List, Dict, Any
import warnings # Import warnings

# Langchain related imports
from langchain_openai import OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker

# Check if OPENAI_API_KEY is set and print a warning if not
if "OPENAI_API_KEY" not in os.environ:
    warnings.warn("OPENAI_API_KEY environment variable not set. SemanticTextSplitter may fail.", RuntimeWarning)

# Define a structure for our documents/chunks with metadata
class Document:
    def __init__(self, page_content: str, metadata: Dict[str, Any]):
        self.page_content = page_content
        self.metadata = metadata

    def __repr__(self):
        return f"Document(page_content='{self.page_content[:50]}...', metadata={self.metadata})"

class DocumentLoader:
    def __init__(self, path: str, encoding: str = "utf-8"):
        # Now stores Document objects
        self.documents: List[Document] = [] 
        self.path = path
        self.encoding = encoding

    def load(self):
        if os.path.isdir(self.path):
            self.load_directory()
        elif os.path.isfile(self.path):
            # Process single file
            file_path = self.path
            filename = os.path.basename(file_path)
            metadata = {"source": filename} # Basic metadata
            if file_path.endswith(".txt"):
                self.load_txt_file(file_path, metadata)
            elif file_path.endswith(".pdf"):
                self.load_pdf_file(file_path, metadata)
            else:
                raise ValueError(
                    "Provided file path is not a valid .txt or .pdf file."
                )
        else:
            raise ValueError(
                "Provided path is neither a valid directory nor a file."
            )

    def load_txt_file(self, file_path: str, metadata: Dict[str, Any]):
        with open(file_path, "r", encoding=self.encoding) as f:
            text = f.read()
            self.documents.append(Document(page_content=text, metadata=metadata))

    def load_pdf_file(self, file_path: str, metadata: Dict[str, Any]):
        doc = fitz.open(file_path)
        text = ""
        # Add page number to metadata if desired
        # For simplicity, we combine all pages into one Document for now
        for page_num, page in enumerate(doc):
            text += page.get_text()
            # Could potentially create a Document per page here:
            # page_metadata = {**metadata, "page": page_num + 1}
            # self.documents.append(Document(page_content=page.get_text(), metadata=page_metadata))
        doc.close()
        if text:
            # If combining pages, the metadata refers to the whole file
            self.documents.append(Document(page_content=text, metadata=metadata))

    def load_directory(self):
        for root, _, files in os.walk(self.path):
            for file in files:
                file_path = os.path.join(root, file)
                filename = os.path.basename(file_path)
                metadata = {"source": filename} # Basic metadata for the file
                if file.endswith(".txt"):
                    self.load_txt_file(file_path, metadata)
                elif file.endswith(".pdf"):
                    self.load_pdf_file(file_path, metadata)

    def load_documents(self) -> List[Document]: # Return type updated
        self.load()
        return self.documents


class CharacterTextSplitter:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        assert (
            chunk_size > chunk_overlap
        ), "Chunk size must be greater than chunk overlap"

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, document: Document) -> List[Document]: # Accepts and returns Document
        text = document.page_content
        metadata = document.metadata
        chunks = []
        start_index = 0
        chunk_seq_num = 1
        while start_index < len(text):
            end_index = start_index + self.chunk_size
            chunk_text = text[start_index:end_index]
            chunk_metadata = {**metadata, "chunk_num": chunk_seq_num, "start_char": start_index} # Add chunk metadata
            chunks.append(Document(page_content=chunk_text, metadata=chunk_metadata))
            start_index += self.chunk_size - self.chunk_overlap
            chunk_seq_num += 1
        return chunks

    def split_texts(self, documents: List[Document]) -> List[Document]: # Accepts and returns List[Document]
        chunks = []
        for doc in documents:
            chunks.extend(self.split(doc))
        return chunks


class SemanticTextSplitter:
    def __init__(self, breakpoint_threshold_type="percentile", breakpoint_threshold_amount=0.95):
        # ... (init remains mostly the same, ensure API key check)
        if "OPENAI_API_KEY" not in os.environ:
            raise ValueError("OPENAI_API_KEY environment variable must be set to use SemanticTextSplitter.")
        self.embeddings = OpenAIEmbeddings()
        self.splitter = SemanticChunker(
            self.embeddings,
            breakpoint_threshold_type=breakpoint_threshold_type,
        )
        print(f"Initialized SemanticTextSplitter with threshold type: {breakpoint_threshold_type}")

    def split_texts(self, documents: List[Document]) -> List[Document]: # Accepts List[Document], returns List[Document]
        """Splits a list of documents into semantic chunks, preserving metadata."""
        all_chunked_documents = []
        print(f"Starting semantic splitting on {len(documents)} document(s)...")
        for doc_index, doc in enumerate(documents):
            # Langchain's create_documents works on raw text strings
            langchain_chunks = self.splitter.create_documents([doc.page_content])
            # Create our Document objects, adding original metadata + chunk info
            for chunk_index, lc_chunk in enumerate(langchain_chunks):
                chunk_metadata = {**doc.metadata, "chunk_num": chunk_index + 1} # Add chunk number
                all_chunked_documents.append(Document(page_content=lc_chunk.page_content, metadata=chunk_metadata))
            print(f"  - Document {doc_index+1} split into {len(langchain_chunks)} chunks.")
        
        print(f"Semantic splitting resulted in {len(all_chunked_documents)} total chunks.")
        return all_chunked_documents


if __name__ == "__main__":
    # ... (Setup loader as before)
    loader = DocumentLoader("data/KingLear.txt")
    loader.load()
    
    loaded_documents = loader.documents # Now a list of Document objects

    if not loaded_documents:
        print("No documents loaded. Check the path and file types.")
    else:
        print(f"Loaded {len(loaded_documents)} document(s).")
        print(f"First loaded document: {loaded_documents[0]}")

        print("\n--- CharacterTextSplitter Example ---")
        char_splitter = CharacterTextSplitter()
        char_chunks = char_splitter.split_texts(loaded_documents)
        print(f"Number of character chunks created: {len(char_chunks)}")
        if char_chunks:
            print("First chunk: ", char_chunks[0])
            print("Last chunk: ", char_chunks[-1])
        
        print("\n--- SemanticTextSplitter Example ---")
        try:
            semantic_splitter = SemanticTextSplitter()
            semantic_chunks = semantic_splitter.split_texts(loaded_documents)
            print(f"Number of semantic chunks created: {len(semantic_chunks)}")
            if semantic_chunks:
                print("First chunk: ", semantic_chunks[0])
                print("Last chunk: ", semantic_chunks[-1])
        except ValueError as e:
            print(f"Could not run SemanticTextSplitter example: {e}")
        except Exception as e:
            print(f"An unexpected error occurred during SemanticTextSplitter example: {e}")
