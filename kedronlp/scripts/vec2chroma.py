import chromadb
import csv
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

id_lookup = set()
duplicate_docs = 0
ids = []
embeddings = []
batch = []
metadatas = []
batch_size = 1000
inserted_rows = 0

start_insertion = time()
for row in reader:
    split_doc = row["doc"].split("Paragraph-")
    id = split_doc[0] + "Paragraph-" + split_doc[1][0]

    if id in id_lookup:
        duplicate_docs += 1
        continue

    id_lookup.add(id)

    metadata = {}
    metadata_chunks = [chunks for chunks in row["doc"].split("\n")][0:-1]
    for chunk in metadata_chunks:
        splitter = chunk.find(":")
        key = chunk[:splitter]
        value = chunk[splitter+2:]
        try:
            value = float(value)
        except ValueError:
            value = value
        metadata[key] = value

    metadatas.append(metadata)
    ids.append(id)
    embeddings.append(ast.literal_eval(row["embedding"]))
    batch.append(row["doc"])

    if len(batch) >= batch_size:
        start = time()
        collection.upsert(
            ids=ids,
            documents=batch,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        inserted_rows += len(batch)
        print(f"rows inserted until now: {inserted_rows} (time for one batch: {time()-start:.4f}s)")
        batch = []
        ids = []
        embeddings = []
        metadatas = []

if batch:
    collection.upsert(
        ids=ids,
        documents=batch,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    inserted_rows += len(batch)
    batch = []
    ids = []
    embeddings = []
    metadatas = []

end_insertion = time()
input_csv.close()

print("done!")
print(f"inserted in total {inserted_rows} documents")
print(f"found {duplicate_docs} duplicate documents")
print(f"insertion duration: {(end_insertion - start_insertion)/60:.2f} min")
