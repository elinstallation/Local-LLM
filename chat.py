import sys
import threading
import time

import chromadb
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.base.embeddings.base import similarity
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore


def loading(stop_event):
    i = 0
    while not stop_event.is_set():
        dots = "." * (i % 4)
        sys.stdout.write((f"\rA: Thinking{dots:<3}"))
        sys.stdout.flush()
        time.sleep(0.4)
        i += 1
    sys.stdout.write("\rA: " + " " * 15 + "\rA: ")
    sys.stdout.flush()


Settings.embed_model = HuggingFaceEmbedding(
    model_name="nomic-ai/nomic-embed-text-v1.5",
    cache_folder="./models",
    device="cpu",
    trust_remote_code=True,
)

Settings.llm = Ollama(model="llama3.1:8b", request_timeout=360.0)

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name="vector_collection")
vector_store = ChromaVectorStore(chroma_collection=collection)

index = VectorStoreIndex.from_vector_store(vector_store)

query_engine = index.as_query_engine(streaming=True, similarity_top_k=3)

print("\n" + "=" * 40)
print("Local LLM Ready")
print("Type 'q' to quit")
print("\n" + "=" * 40)

while True:
    user_input = input("\nQ: ")
    if user_input.lower() == "q":
        print("Shutting down...")
        break
    if not user_input.strip():
        continue

    stop_loading = threading.Event()
    loading_thread = threading.Thread(target=loading, args=(stop_loading,))

    try:
        loading_thread.start()
        response = query_engine.query(user_input)
        stop_loading.set()
        loading_thread.join()
        response.print_response_stream()  # type:ignore
        print()
    except Exception as e:
        stop_loading.set()
        loading_thread.join()
        print(f"\nError querying the LLM {e}")
