import argparse
import hashlib
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

import chromadb
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import DocxReader, PDFReader
from llama_index.readers.file.epub.base import BaseReader
from llama_index.readers.file.unstructured.base import UnstructuredReader
from llama_index.vector_stores.chroma import ChromaVectorStore

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.disable(logging.WARNING)

# filehandling
IDX_FILES_LOG = "./indexed_files.json"


def load_indexed_files() -> dict:
    if Path(IDX_FILES_LOG).exists():
        with open(IDX_FILES_LOG, "r") as f:
            return json.load(f)
    return {}


def save_indexed_files(indexed: dict):
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as tmp:
        json.dump(indexed, tmp)

    os.replace(tmp.name, IDX_FILES_LOG)


def file_hash(path: Path) -> str:
    try:
        return hashlib.md5(path.read_bytes()).hexdigest()
    except OSError as e:
        print(f"Skipping unreadable file {path}:{e}")
        return ""


parser = argparse.ArgumentParser(description="Index Document into the vector store")
parser.add_argument(
    "directory",
    nargs="?",
    default="./data",
    help="Directory to indec (default: ./data)",
)
args = parser.parse_args()

DATA_DIR = Path(args.directory).expanduser().resolve()

if not DATA_DIR.exists():
    print(f"Error: Directory '{DATA_DIR}' does not exist.")
    sys.exit(1)

print(f"Indexing from: {DATA_DIR}")


Settings.embed_model = HuggingFaceEmbedding(
    model_name="nomic-ai/nomic-embed-text-v1.5",
    cache_folder="./models",
    device="cpu",
    trust_remote_code=True,
)

Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="vector_collection")
vector_store = ChromaVectorStore(chroma_collection=collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

pdf_reader = PDFReader()
docx_reader = DocxReader()
reader = UnstructuredReader()

extractor_map: dict[str, BaseReader] = {
    ".pdf": pdf_reader,
    ".docx": docx_reader,
    ".html": reader,
    ".txt": reader,
    ".md": reader,
}

idx_files = load_indexed_files()
all_files = [p for p in DATA_DIR.rglob("*") if p.suffix in extractor_map]

current_paths = {str(f) for f in all_files}

deleted_files = [path for path in idx_files if path not in current_paths]

if deleted_files:
    print(f"Found {len(deleted_files)} deleted files. Removing from vector store...")
    for del_file in deleted_files:
        try:
            collection.delete(where={"source": del_file})
            print(f"Deleted vectors for: {del_file}")
        except Exception as e:
            print(f"Error deleting vectors for {del_file}: {e}")

    for del_file in deleted_files:
        idx_files.pop(del_file, None)
    save_indexed_files(idx_files)

# idx_files = {k: v for k, v in idx_files.items() if k in current_paths}

files_hashes = {f: file_hash(f) for f in all_files}

new_files = [
    f
    for f in all_files
    if str(f) not in idx_files or idx_files[str(f)] != files_hashes[f]
]

# indexing
if not new_files:
    print("No new files to index")
else:
    docs = []
    successfully_indexed = []
    for f in new_files:
        try:
            loaded = SimpleDirectoryReader(
                input_files=[f], file_extractor=extractor_map
            ).load_data()

            loaded = [d for d in loaded if d.text and d.text.strip()]

            if not loaded:
                print("No content extracted")
                continue
            for doc in loaded:
                doc.metadata.update(
                    {
                        "source": str(f),
                        "file_name": f.name,
                        "file_type": f.suffix,
                        "file_size": f.stat().st_size,
                    }
                )

            if str(f) in idx_files:
                print(f"File updated. Removing old chunks for: {f}")
                collection.delete(where={"source": str(f)})
            docs.extend(loaded)
            successfully_indexed.append(f)

        except Exception as e:
            print(f"Skipping {f}: {e}")
    if docs:
        index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)
        idx_files.update({str(f): file_hash(f) for f in successfully_indexed})
        save_indexed_files(idx_files)
        print("Indexed")
        print()
        print("To index a new directory, run: ")
        print("     python preprocessing.py ~/path/to/folder")
