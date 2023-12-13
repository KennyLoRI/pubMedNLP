import chromadb
import csv
import sys
import ast
from time import time

maxInt = sys.maxsize

while True:
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt / 10)

csv.field_size_limit(sys.maxsize)

client = chromadb.PersistentClient(path="../chroma_store")
collection = client.get_or_create_collection(
    name="pubmed_embeddings",
    embedding_function=None,
    metadata={"hnsw:space": "cosine"},
)

input_csv = open("../data/01_raw/doc_embeddings.csv")
reader = csv.DictReader(input_csv)
ids = []
id_lookup = set()
duplicates = 0
embeddings = []
batch = []
batch_size = 1000
inserted_rows = 0
for row in reader:
    id = row["combined_doc"].split("\nAbstract")[0]

    if id in id_lookup:
        duplicates += 1
        continue

    ids.append(id)
    id_lookup.add(id)
    embeddings.append(ast.literal_eval(row["embedding"]))
    batch.append(row["combined_doc"])
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
print(f"skipped {duplicates} duplicate documents")

input_csv.close()
