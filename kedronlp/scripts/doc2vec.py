import torch
import numpy as np
import csv
import sys

# user libraries
import scripts_utils
import emb_utils


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print()

maxInt = sys.maxsize

while True:
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt / 10)

csv.field_size_limit(sys.maxsize)


def get_doc_embeddings(doc_batch, model):
    doc_embeddings = model.encode(doc_batch)
    return np.stack(doc_embeddings, axis=0).tolist()


def prepare_write_output(doc_batch, embeddings):
    rows = []
    for doc, embedding in zip(doc_batch, embeddings):
        row = {"combined_doc": doc, "embedding": embedding}
        rows.append(row)
    return rows


def execute_write_pipeline(doc_batch, model, writer):
    embeddings = get_doc_embeddings(doc_batch, model)
    rows = prepare_write_output(doc_batch, embeddings)
    writer.writerows(rows)


model = emb_utils.PubMedBert(device=device)

input_csv = open("../data/01_raw/extract_data.csv")
reader = csv.DictReader(input_csv)

output_csv = open("../data/01_raw/doc_embeddings.csv", "w")
writer = csv.DictWriter(output_csv, fieldnames=["combined_doc", "embedding"])
writer.writeheader()

doc_batch = []
batch_size = 256
total_docs_processed = 0
for row in reader:
    row = scripts_utils.preprocess_row(row)
    combined_doc = scripts_utils.get_combined_doc(row)
    doc_batch.append(combined_doc)
    if len(doc_batch) >= batch_size:
        execute_write_pipeline(doc_batch, model, writer)
        total_docs_processed += len(doc_batch)
        print(f"until now processed {total_docs_processed} documents")
        doc_batch = []

if doc_batch:
    execute_write_pipeline(doc_batch, model, writer)
    total_docs_processed += len(doc_batch)
print("done!")
print(f"processed in total {total_docs_processed} documents")

input_csv.close()
output_csv.close()
