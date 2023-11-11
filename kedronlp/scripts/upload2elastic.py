from elasticsearch import helpers, Elasticsearch
import csv
import ast

es = Elasticsearch(
    [
        {
            "host": "localhost",
            "port": 9200,
            "scheme": "http",
        }
    ]
)

es.options(ignore_status=[400,404]).indices.delete(index="pubmed")

def get_descriptors(descriptor_string):
    descriptor_list = []
    tmp = descriptor_string.replace("\'", "").split("StringElement(")
    for item in tmp:
        pos = item.find(",")
        if pos != -1:
            descriptor = item[:pos]
            descriptor_list.append(descriptor)
    return descriptor_list

with open("../data/01_raw/extract_data.csv") as csv_file:
    reader = csv.DictReader(csv_file)
    batch = []
    batch_size = 1000
    inserted_rows = 0
    while True:
        try:
            row = next(reader)
            for key in ["Authors", "Affiliations", "Qualifier", "Major Qualifier"]:
                if row[key] != "NA":
                    row[key] = ast.literal_eval(row[key])
                else:
                    row[key] = []
            for key in ["Descriptor", "Major Descriptor"]:
                if row[key] != "NA":
                    row[key] = get_descriptors(row[key])
                else:
                    row[key] = []
            batch.append(row)
            if len(batch) >= batch_size:
                helpers.bulk(es, batch, index="pubmed")
                inserted_rows += len(batch)
                batch = []
        except StopIteration:
            helpers.bulk(es, batch, index="pubmed")
            inserted_rows += len(batch)
            print("done!")
            print(f"inserted {inserted_rows} rows")
            break
