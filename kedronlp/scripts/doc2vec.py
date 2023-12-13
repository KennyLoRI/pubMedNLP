import torch
import csv
import sys
import ast

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


def prepare_write_output(doc_batch, embeddings):
    rows = []
    for doc, embedding in zip(doc_batch, embeddings):
        row = {"doc": doc, "embedding": embedding}
        rows.append(row)
    return rows


def execute_write_pipeline(doc_batch, model, writer):
    embeddings = model.encode(doc_batch)
    rows = prepare_write_output(doc_batch, embeddings)
    writer.writerows(rows)


model = emb_utils.PubMedBert(device=device)

input_csv = open("../data/02_intermediate/paragraphed_data.csv")
reader = csv.DictReader(input_csv)

output_csv = open("../data/01_raw/doc_embeddings.csv", "w")
writer = csv.DictWriter(output_csv, fieldnames=["doc", "embedding"])
writer.writeheader()

id_lookup = set()
doc_batch = []
duplicate_docs = 0
batch_size = 256
total_docs_processed = 0
for row in reader:
    row.pop("Abstract", None)
    row = scripts_utils.preprocess_row(row)
    combined_doc = scripts_utils.get_combined_doc(row)
    document_id = combined_doc.split("\nparagraphs")[0]
    paragraphs = ast.literal_eval(row["paragraphs"])

    for i, paragraph in enumerate(paragraphs):
        combined_paragraph = document_id + f"\nParagraph {i}:" + paragraph

        if combined_paragraph not in id_lookup:
            id_lookup.add(combined_paragraph)
            doc_batch.append(combined_paragraph)

            if len(doc_batch) >= batch_size:
                execute_write_pipeline(doc_batch, model, writer)
                total_docs_processed += len(doc_batch)
                print(f"until now processed {total_docs_processed} documents")
                doc_batch = []

        else:
            duplicate_docs += 1

if doc_batch:
    execute_write_pipeline(doc_batch, model, writer)
    total_docs_processed += len(doc_batch)

input_csv.close()
output_csv.close()

print("done!")
print(f"processed in total {total_docs_processed} documents")
print(f"found {duplicate_docs} duplicate documents")
