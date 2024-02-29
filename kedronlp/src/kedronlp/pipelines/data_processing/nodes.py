import pandas
import pandas as pd
from sentence_transformers import SentenceTransformer
import spacy
from functools import partial
import torch
from kedronlp.extract_utils import get_article_IDs, fetch_details, get_paragraphs
import kedronlp.scripts_utils as scripts_utils
import kedronlp.embedding_utils as emb_utils
from sklearn.metrics.pairwise import cosine_similarity
from scipy.signal import argrelextrema
import numpy as np
import ast
import chromadb
from time import time

def extract_data(extract_params) -> pandas.DataFrame:
    """
    Function that calls the Pubmed API via the Entrez package.
    :param
        extract_params:
            hyperparameters specifying start and end date of the data to be extracted
            as well as the time window in which the request is done to receive data in a batch-wise manner.
            If this node fails reduce window duration because request maximum is likely exceeded

    :return:
        pandas dataframe containing the crawled details (abstracts, kewords etc.) of all articles matching the query including.
    """

    title_list = []
    authors_list = []
    affiliation_list = []
    abstract_list = []
    journal_list = []
    language_list = []
    pubdate_year_list = []
    pubdate_month_list = []
    major_descriptor_list = []
    descriptor_list = []
    major_qualifier_list = []
    qualifier_list = []

    studiesIdList = get_article_IDs(extract_params) #calls IDs of the articles to fetch detailed data for
    chunk_size = 500  # reduce chunksize to not exceed request limits
    for chunk_i in range(0, len(studiesIdList), chunk_size):
        chunk = studiesIdList[chunk_i:chunk_i + chunk_size]
        papers = fetch_details(chunk)
        for i, paper in enumerate(papers['PubmedArticle']):
            title_list.append(paper['MedlineCitation']['Article']['ArticleTitle'])
            try:
                abstract_list.append(paper['MedlineCitation']['Article']['Abstract']['AbstractText'][0])
            except:
                abstract_list.append('NA')
            try:
                journal_list.append(paper['MedlineCitation']['Article']['Journal']['Title'])
            except:
                journal_list.append('NA')
            try:
                language_list.append(paper['MedlineCitation']['Article']['Language'][0])
            except:
                language_list.append('NA')
            try:
                descr = [descriptor['DescriptorName'] for descriptor in paper['MedlineCitation']['MeshHeadingList']]
                descriptor_list.append(descr)
            except:
                descriptor_list.append("NA")
            try:
                mdescr = [descriptor['DescriptorName'] for descriptor in paper['MedlineCitation']['MeshHeadingList'] if
                          descriptor['DescriptorName'].attributes.get('MajorTopicYN') == 'Y']
                major_descriptor_list.append(mdescr)
            except:
                major_descriptor_list.append('NA')
            try:
                qualif = [str(descriptor['QualifierName'][0]) for descriptor in
                          paper['MedlineCitation']['MeshHeadingList'] if descriptor['QualifierName']]
                qualifier_list.append(list(set(qualif)))  # append only unique qualifiers
            except:
                qualifier_list.append('NA')
            try:
                maj_qualif = [str(descriptor['QualifierName'][0]) for descriptor in
                              paper['MedlineCitation']['MeshHeadingList'] if
                              descriptor['QualifierName'] and descriptor['QualifierName'].attributes.get(
                                  'MajorTopicYN') == 'Y']
                major_qualifier_list.append(list(set(maj_qualif)))  # only unique
            except:
                major_qualifier_list.append('NA')
            try:
                authors_list.append([", ".join([author.get('LastName'), author.get('ForeName')]) for author in
                                     paper['MedlineCitation']['Article']['AuthorList']])
            except:
                authors_list.append('NA')
            try:
                affiliation_lst = []
                for i, author in enumerate(paper['MedlineCitation']['Article']['AuthorList']):
                    try:
                        affiliation_lst.append(
                            [affiliation.get('Affiliation', '') for affiliation in author.get('AffiliationInfo')][0])
                    except:
                        continue
                affiliation_list.append(affiliation_lst)
            except:
                affiliation_list.append('NA')
            try:
                pubdate_year_list.append(
                    paper['MedlineCitation']['Article']['Journal']['JournalIssue']['PubDate']['Year'])
            except:
                pubdate_year_list.append('NA')
            try:
                pubdate_month_list.append(
                    paper['MedlineCitation']['Article']['Journal']['JournalIssue']['PubDate']['Month'])
            except:
                pubdate_month_list.append('NA')
        print(f"percentage fetched: {chunk_i / len(studiesIdList)}")
        print(f"chunk:{chunk_i}")
    df = pd.DataFrame(
            list(zip(
            title_list,
            authors_list,
            affiliation_list,
            qualifier_list,
            major_qualifier_list,
            descriptor_list,
            major_descriptor_list,
            abstract_list,
            journal_list,
            language_list,
            pubdate_year_list,
            pubdate_month_list
            )),
            columns=[
                'Title', 'Authors', 'Affiliations', 'Qualifier', 'Major Qualifier', 'Descriptor', 'Major Descriptor',
                'Abstract', 'Journal', 'Language', 'Year', 'Month'
            ])
    return df

def process_extract_data(extract_data: pandas.DataFrame) -> pandas.DataFrame:
    spacy.cli.download("en_core_web_sm")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO", device=device)
    nlp = spacy.load("en_core_web_sm")

    # Create a partial function with constant arguments
    partial_get_paragraphs = partial(get_paragraphs, nlp=nlp, model=model)
    extract_data_clean = extract_data.dropna().head(3) #remove head when not testing
    extract_data_clean['paragraphs'] = extract_data_clean['Abstract'].apply(partial_get_paragraphs)
    return extract_data_clean

def create_paragraphs(extract_data: pandas.DataFrame) -> pandas.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = emb_utils.PubMedBert(device=device)

    paragraphs_df = pd.DataFrame(columns=["doc_info", "paragraphs"])

    nlp = spacy.load("en_core_web_sm")

    paragraphs_rows = []
    batch_size = 256
    processed = 0

    for _, row in extract_data.iterrows():
        row = scripts_utils.preprocess_row(row)
        doc_info = scripts_utils.get_doc_info(row)
        sentences = nlp(row["Abstract"])
        sentences = [str(sent) for sent in sentences.sents]

        if len(sentences) <= 2:
            paragraphs_row = {"doc_info": doc_info, "paragraphs": [" ".join(sentences)]}
            paragraphs_rows.append(paragraphs_row)

        else:
            embeddings = model.encode(sentences)
            similarities = cosine_similarity(embeddings)
            values = similarities.diagonal(1)
            for i in range(2, similarities.shape[0]):
                values = np.append(values, similarities.diagonal(i))
            relevant_mean = np.mean(values)
            similarities -= relevant_mean

            num_weights = 3
            if len(sentences)-1 < num_weights:
                num_weights = len(sentences)-1

            def sigmoid(x):
                return (1 / (1 + np.exp(-x)))

            sig = np.vectorize(sigmoid)
            x = np.linspace(5, -5, num_weights)
            activation_weights = np.pad(sig(x), (0, similarities.shape[0]-num_weights))
            sim_rows = [similarities[i, i+1:] for i in range(similarities.shape[0])]
            sim_rows = [np.pad(sim_row, (0, similarities.shape[0]-len(sim_row))) for sim_row in sim_rows]
            sim_rows = np.stack(sim_rows) * activation_weights
            weighted_sums = np.insert(np.sum(sim_rows, axis=1), [0], [0])

            minimas = argrelextrema(weighted_sums, np.less)
            split_points = [minima for minima in minimas[0]]

            if split_points:
                paragraphs = []
                start = 0
                for split_point in split_points:
                    paragraphs.append(sentences[start:split_point])
                    start = split_point
                paragraphs.append(sentences[split_points[-1]:])
                paragraphs = [" ".join(sentence_list) for sentence_list in paragraphs]

                paragraphs_row = {"doc_info": doc_info, "paragraphs": paragraphs}
                paragraphs_rows.append(paragraphs_row)

            else:
                paragraphs_row = {"doc_info": doc_info, "paragraphs": [" ".join(sentences)]}
                paragraphs_rows.append(paragraphs_row)

        if len(paragraphs_rows) >= batch_size:
            new_rows_df = pd.DataFrame(paragraphs_rows)
            paragraphs_df = pd.concat([paragraphs_df, new_rows_df], ignore_index=True)
            processed += len(paragraphs_rows)
            paragraphs_rows = []
            print(f"until now processed {processed} documents")

    if paragraphs_rows:
        new_rows_df = pd.DataFrame(paragraphs_rows)
        paragraphs_df = pd.concat([paragraphs_df, new_rows_df], ignore_index=True)
        processed += len(paragraphs_rows)
        paragraphs_rows = []

    print(f"processed in total {processed} documents")
    
    return paragraphs_df

def paragraph2vec(paragraphs: pandas.DataFrame) -> pandas.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    def prepare_write_output(description_batch, embeddings):
        rows = []
        for doc, embedding in zip(description_batch, embeddings):
            row = {"doc": doc, "embedding": embedding}
            rows.append(row)
        return pd.DataFrame(rows)

    model = emb_utils.PubMedBert(device=device)

    paragraph_embeddings = pd.DataFrame(columns=["doc", "embedding"])

    id_lookup = set()
    doc_batch = []
    description_batch = []
    duplicate_docs = 0
    batch_size = 256
    total_docs_processed = 0
    for _, row in paragraphs.iterrows():
        paragraphs_list = ast.literal_eval(row["paragraphs"])

        for i, paragraph in enumerate(paragraphs_list):
            description = row["doc_info"] + f"Paragraph-{i}: " + paragraph

            if description in id_lookup:
                duplicate_docs += 1
                continue

            id_lookup.add(description)
            description_batch.append(description)
            doc_batch.append(paragraph)

            if len(doc_batch) >= batch_size:
                embeddings = model.encode(doc_batch)
                new_rows_df = prepare_write_output(description_batch, embeddings)
                paragraph_embeddings = pd.concat([paragraph_embeddings, new_rows_df], ignore_index=True)
                total_docs_processed += len(doc_batch)
                print(f"until now processed {total_docs_processed} documents")
                doc_batch = []
                description_batch = []
                

    if doc_batch:
        embeddings = model.encode(doc_batch)
        new_rows_df = prepare_write_output(description_batch, embeddings)
        paragraph_embeddings = pd.concat([paragraph_embeddings, new_rows_df], ignore_index=True)
        total_docs_processed += len(doc_batch)


    print("done!")
    print(f"processed in total {total_docs_processed} documents")
    print(f"found {duplicate_docs} duplicate documents")

    return paragraph_embeddings


def vec2chroma(paragraph_embeddings: pandas.DataFrame) -> pandas.DataFrame:

    client = chromadb.PersistentClient(path="chroma_store/")

    try:
        client.delete_collection(name="pubmed_embeddings")
    except ValueError:
        pass

    collection = client.create_collection(
        name="pubmed_embeddings",
        embedding_function=None,
        metadata={"hnsw:space": "cosine"},
    )

    id_lookup = set()
    duplicate_docs = 0
    ids = []
    embeddings = []
    batch = []
    metadatas = []
    batch_size = 500
    inserted_rows = 0
    for _, row in paragraph_embeddings.iterrows():
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

    print("done!")
    print(f"inserted in total {inserted_rows} documents to chromadb")
    print(f"found {duplicate_docs} duplicate documents")
