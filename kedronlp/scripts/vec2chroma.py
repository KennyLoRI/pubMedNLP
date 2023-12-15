import chromadb
import csv
import sys
import ast
from time import time

# user libs
import scripts_utils

scripts_utils.increase_csv_maxsize()

client = chromadb.PersistentClient(path="../chroma_store")

try:
    client.delete_collection(name="pubmed_embeddings")
except ValueError:
    pass

collection = client.create_collection(
    name="pubmed_embeddings",
    embedding_function=None,
    metadata={"hnsw:space": "cosine"},
)

input_csv = open("../data/01_raw/paragraph_embeddings.csv")
reader = csv.DictReader(input_csv)

ids = []
embeddings = []
batch = []
batch_size = 500
inserted_rows = 0
for row in reader:
    split_doc = row["doc"].split("Paragraph-")
    id = split_doc[0] + "Paragraph-" + split_doc[1][0]

    ids.append(id)
    embeddings.append(ast.literal_eval(row["embedding"]))
    batch.append(row["doc"])

    if len(batch) >= batch_size:
        start = time()
        collection.upsert(
            ids=ids,
            documents=batch,
            embeddings=embeddings,
        )
        inserted_rows += len(batch)
        print(f"rows inserted until now: {inserted_rows} (time for one batch: {time()-start:.4f}s)")
        batch = []
        ids = []
        embeddings = []

if batch:
    collection.upsert(
        ids=ids,
        documents=batch,
        embeddings=embeddings,
    )
    inserted_rows += len(batch)
    batch = []
    ids = []
    embeddings = []

print("done!")
print(f"inserted in total {inserted_rows} documents")

input_csv.close()
