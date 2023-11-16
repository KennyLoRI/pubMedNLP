from elasticsearch import helpers, Elasticsearch
from sentence_transformers import SentenceTransformer

# user libraries
import scripts_utils

es = Elasticsearch("http://localhost:9200")

index = "pubmed"
new_mapping = {
    "combined_doc": {"type": "text"},
    "embedding": {"type": "dense_vector", "dims": 768},
}
es.indices.put_mapping(properties=new_mapping, index=index)
model = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO")

resp = helpers.scan(
    es, query={"query": {"match_all": {}}}, index=index, size=100
)
index_reqs = []
batch_size = 100
docs_processed = 0
while True:
    try:
        hit = next(resp)
        id = hit["_id"]
        combined_doc = scripts_utils.get_combined_doc(hit["_source"])
        embedding = model.encode(combined_doc)
        index_req = {
            "_op_type": "index",
            "_index": index,
            "_id": id,
            "combined_doc": combined_doc,
            "embedding": embedding,
        } | hit["_source"]
        index_reqs.append(index_req)
        if len(index_reqs) >= batch_size:
            helpers.bulk(es, index_reqs)
            docs_processed += batch_size
            print(f"number of docs processed: {docs_processed}")
            index_reqs = []
    except StopIteration:
        helpers.bulk(es, index_reqs)
        docs_processed += len(index_reqs)
        print(f"number of docs processed: {docs_processed}")
        print("done!")
        break
