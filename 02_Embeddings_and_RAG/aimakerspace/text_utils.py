import os
import fitz  # PyMuPDF
from typing import List


class DocumentLoader:
    def __init__(self, path: str, encoding: str = "utf-8"):
        self.documents = []
        self.path = path
        self.encoding = encoding

    def load(self):
        if os.path.isdir(self.path):
            self.load_directory()
        elif os.path.isfile(self.path):
            if self.path.endswith(".txt"):
                self.load_txt_file(self.path)
            elif self.path.endswith(".pdf"):
                self.load_pdf_file(self.path)
            else:
                raise ValueError(
                    "Provided file path is not a valid .txt or .pdf file."
                )
        else:
            raise ValueError(
                "Provided path is neither a valid directory nor a file."
            )

    def load_txt_file(self, file_path: str):
        with open(file_path, "r", encoding=self.encoding) as f:
            self.documents.append(f.read())

    def load_pdf_file(self, file_path: str):
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        if text:
            self.documents.append(text)

    def load_directory(self):
        for root, _, files in os.walk(self.path):
            for file in files:
                file_path = os.path.join(root, file)
                if file.endswith(".txt"):
                    self.load_txt_file(file_path)
                elif file.endswith(".pdf"):
                    self.load_pdf_file(file_path)

    def load_documents(self):
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

    def split(self, text: str) -> List[str]:
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunks.append(text[i : i + self.chunk_size])
        return chunks

    def split_texts(self, texts: List[str]) -> List[str]:
        chunks = []
        for text in texts:
            chunks.extend(self.split(text))
        return chunks


if __name__ == "__main__":
    # Example Usage (assuming a directory 'data' with mixed file types)
    # loader = DocumentLoader("data/") # Example for directory
    # or
    loader = DocumentLoader("data/KingLear.txt") # Example for single txt file
    # or
    # loader = DocumentLoader("data/sample.pdf") # Example for single pdf file (if exists)
    loader.load()
    if not loader.documents:
        print("No documents loaded. Check the path and file types.")
    else:
        splitter = CharacterTextSplitter()
        chunks = splitter.split_texts(loader.documents)
        print(f"Number of documents loaded: {len(loader.documents)}")
        print(f"Number of chunks created: {len(chunks)}")
        if chunks:
            print("First chunk:")
            print(chunks[0])
            print("--------")
            if len(chunks) > 1:
                print("Second chunk:")
                print(chunks[1])
                print("--------")
            if len(chunks) > 2:
                 print("Second to last chunk:")
                 print(chunks[-2])
                 print("--------")
            print("Last chunk:")
            print(chunks[-1])
