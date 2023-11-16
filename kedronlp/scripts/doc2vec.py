from transformers import AutoTokenizer, AutoModel
import torch
import csv
import sys

# user libraries
import scripts_utils


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


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def get_doc_embeddings(doc_batch, tokenizer, model):
    encoded_input = tokenizer(
        doc_batch, padding=True, truncation=True, return_tensors="pt"
    )
    encoded_input = encoded_input.to(device)
    with torch.no_grad():
        model_output = model(**encoded_input)
    doc_embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
    return doc_embeddings


def prepare_write_output(combined_docs, embeddings):
    rows = []
    for combined_doc, embedding in zip(combined_docs, embeddings):
        row = {"combined_doc": combined_doc, "embedding": embedding.detach().tolist()}
        rows.append(row)
    return rows


def execute_write_pipeline(doc_batch, tokenizer, model, writer):
    embeddings = get_doc_embeddings(doc_batch, tokenizer, model)
    rows = prepare_write_output(doc_batch, embeddings)
    writer.writerows(rows)


tokenizer = AutoTokenizer.from_pretrained(
    "pritamdeka/S-PubMedBert-MS-MARCO", model_max_length=512
)
model = AutoModel.from_pretrained("pritamdeka/S-PubMedBert-MS-MARCO").to(device)

input_csv = open("../data/01_raw/extract_data.csv")
reader = csv.DictReader(input_csv)

output_csv = open("../data/01_raw/doc_embeddings.csv", "w")
writer = csv.DictWriter(output_csv, fieldnames=["combined_doc", "embedding"])
writer.writeheader()

doc_batch = []
batch_size = 256
total_docs_processed = 0
while True:
    try:
        row = next(reader)
        row = scripts_utils.preprocess_row(row)
        combined_doc = scripts_utils.get_combined_doc(row)
        doc_batch.append(combined_doc)
        if len(doc_batch) >= batch_size:
            execute_write_pipeline(doc_batch, tokenizer, model, writer)
            total_docs_processed += len(doc_batch)
            print(f"until now processed {total_docs_processed} documents")
            doc_batch = []

    except StopIteration:
        execute_write_pipeline(doc_batch, tokenizer, model, writer)
        print("done!")
        total_docs_processed += len(doc_batch)
        print(f"processed in total {total_docs_processed} documents")
