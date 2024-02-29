import chromadb
import csv
import ast
from time import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--granularity",
    type=str,
    required=True,
    help="granularity to add to chroma, can be either 'paragraphs' or 'abstracts'"
)
args = parser.parse_args()

client = chromadb.PersistentClient(path=f"../chroma_store_{args.granularity}")

try:
    client.delete_collection(name="pubmed_embeddings")
except ValueError:
    pass

collection = client.create_collection(
    name="pubmed_embeddings",
    embedding_function=None,
    metadata={"hnsw:space": "cosine"},
)

if args.granularity == "paragraphs":
    input_csv = open("../data/01_raw/paragraph_embeddings.csv", encoding="utf-8")
    doc_key = "doc"
elif args.granularity == "abstracts":
    input_csv = open("../data/01_raw/abstract_metadata_embeddings.csv", encoding="utf-8")
    doc_key = "combined_doc"

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
    id = row[doc_key]  # set id as the document itself, not important

    if id in id_lookup:
        duplicate_docs += 1
        continue

    id_lookup.add(id)

    if args.granularity == "paragraphs":
        metadata_chunks = [chunks for chunks in row[doc_key].split("\n")][0:-1]
    elif args.granularity == "abstracts":
        metadata_chunks = [chunks for chunks in row[doc_key].strip().split("\n")]
        metadata_chunks.pop(-5)

    metadata = {}
    for chunk in metadata_chunks:
        splitter = chunk.find(":")
        key = chunk[:splitter]
        value = chunk[splitter + 2 :]
        try:
            value = float(value)
        except ValueError:
            value = value
        metadata[key] = value

    # preprocess paragraph, leave abstract as is
    if args.granularity == "paragraphs":
        split = row[doc_key].split("Paragraph-")
        new_doc = split[0] + "Abstract" + split[1][1:]
        row[doc_key] = new_doc

    metadatas.append(metadata)
    ids.append(id)
    embeddings.append(ast.literal_eval(row["embedding"]))
    batch.append(row[doc_key])

    if len(batch) >= batch_size:
        start = time()
        collection.upsert(
            ids=ids,
            documents=batch,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        inserted_rows += len(batch)
        print(
            f"rows inserted until now: {inserted_rows} (time for one batch: {time()-start:.4f}s)"
        )
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
