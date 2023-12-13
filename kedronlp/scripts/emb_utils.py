import chromadb
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores.chroma import Chroma
import numpy as np


class PubMedBert:
    def __init__(self, device):
        self.device = device
        self.model = SentenceTransformer(
            "pritamdeka/S-PubMedBert-MS-MARCO", device=self.device
        )
        self.model.max_seq_length = 512

    def encode(self, doc_batch):
        batch_size = len(doc_batch)
        embeddings = self.model.encode(
            doc_batch, device=self.device, batch_size=batch_size
        )
        return np.stack(embeddings, axis=0).tolist()


class PubMedEmbeddingFunction(chromadb.EmbeddingFunction):
    def __init__(self, model):
        self.model = model

    def __call__(self, input):
        return self.model.encode(input)

    def embed_query(self, query):
        return self.model.encode(query)


def get_langchain_chroma(device, persist_dir="../chroma_store"):
    model = PubMedBert(device=device)
    embed_fn = PubMedEmbeddingFunction(model=model)
    client = chromadb.PersistentClient(path=persist_dir)
    langchain_chroma = Chroma(
        client=client,
        collection_name="pubmed_embeddings",
        embedding_function=embed_fn,
        collection_metadata={"hnsw:space": "cosine"},
    )
    return langchain_chroma
