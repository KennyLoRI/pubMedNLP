from elasticsearch import helpers, Elasticsearch
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
        maxInt = int(maxInt/10)

csv.field_size_limit(maxInt)

es = Elasticsearch("http://localhost:9200")

es.options(ignore_status=[400,404]).indices.delete(index="pubmed_embeddings")

index = "pubmed_embeddings"
mappings = {
    "properties": {
        "combined_doc": {"type": "text"},
        "embedding": {"type": "dense_vector", "dims": 768}
    }
}
es.indices.create(index=index, mappings=mappings)

with open("../data/01_raw/doc_embeddings.csv") as csv_file:
    reader = csv.DictReader(csv_file)
    batch = []
    batch_size = 1000
    inserted_rows = 0
    for row in reader:
        row["embedding"] = ast.literal_eval(row["embedding"])
        batch.append(row)
        if len(batch) >= batch_size:
            start = time()
            helpers.bulk(es, batch, index="pubmed_embeddings")
            inserted_rows += len(batch)
            print(f"rows inserted until now: {inserted_rows} (time for one batch: {time()-start:.4f}s)")
            batch = []
    if batch:
        helpers.bulk(es, batch, index="pubmed_embeddings")
        inserted_rows += len(batch)
print("done!")
print(f"inserted in total {inserted_rows} rows")