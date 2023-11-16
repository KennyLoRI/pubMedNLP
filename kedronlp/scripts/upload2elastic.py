from elasticsearch import helpers, Elasticsearch
import ast
import csv
import sys

# user libraries
import scripts_utils

maxInt = sys.maxsize

while True:
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)

csv.field_size_limit(sys.maxsize)

es = Elasticsearch("http://localhost:9200")

es.options(ignore_status=[400,404]).indices.delete(index="pubmed")

with open("../data/01_raw/extract_data.csv") as csv_file:
    reader = csv.DictReader(csv_file)
    batch = []
    batch_size = 10000
    inserted_rows = 0
    while True:
        try:
            row = next(reader)
            row = scripts_utils.preprocess_row(row)
            batch.append(row)
            if len(batch) >= batch_size:
                helpers.bulk(es, batch, index="pubmed")
                inserted_rows += len(batch)
                print(f"rows inserted until now: {inserted_rows}")
                batch = []
        except StopIteration:
            helpers.bulk(es, batch, index="pubmed")
            inserted_rows += len(batch)
            print("done!")
            print(f"inserted in total {inserted_rows} rows")
            break
