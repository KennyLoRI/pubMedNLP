import os
from embedding_utils import get_langchain_chroma
from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from time import time

def get_retriever(top_k_params, device):
    vectordb_path = f"../../chroma_store_{top_k_params['granularity']}"
    assert(os.path.isdir(vectordb_path))

    vectordb = get_langchain_chroma(device=device, persist_dir=vectordb_path)

    if top_k_params["retrieval_strategy"] == "ensemble_retrieval":
        print(f"\ninitiating ensemble retriever... (takes long due to inefficient workaround - no chroma bm25 integration yet)")
        #initiate BM25 retriever
        start = time()
        lang_docs = [Document(page_content=doc) for doc in vectordb.get().get("documents", [])]
        bm25_retriever = BM25Retriever.from_documents(lang_docs)
        bm25_retriever.k = top_k_params["top_k"]

        #initiate similarity retriever
        similarity_retriever = vectordb.as_retriever(
            search_type=top_k_params["advanced_dense_retriever"],
            search_kwargs={
                "k": top_k_params["top_k"],
            })

        #initiate ensemble (uses reciprocal rank fusion in the background with default settings)
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, similarity_retriever],
            weights=[0.5, 0.5],
        )
        end = time()
        print(f"total time required to initialize ensemble retriever: {end-start:.2f}s")
        retriever = ensemble_retriever
    else:
        retriever = vectordb

    return retriever

    