from Bio import Entrez
from datetime import datetime, timedelta
import time
import io
import spacy
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def search(query, mindate, maxdate):
    """
    Function to access IDs to queried data using Entrez
    :param
        query: term for which to search
        mindate: end of time period to be extracted
        maxdate: start of time period to be extracteed
    :return:
        Dictionary with the following keys: 'Count', 'RetMax', 'RetStart', 'IdList', 'TranslationSet', 'QueryTranslation'
    """
    #docs: https://www.ncbi.nlm.nih.gov/books/NBK25499/#chapter4.ESearch
    Entrez.email = 'emails@examples.com'
    handle = Entrez.esearch(db='pubmed',
                            sort='relevance',
                            retmax='10000',
                            retmode='xml',
                            term=query,
                            mindate=mindate,
                            maxdate=maxdate)
    results = Entrez.read(handle)
    return results

def fetch_details(id_list) -> dict:
    """
    Function to fetch detailed data for a list of ID's of articles in Pubmed

    :param
        id_list: list of IDs from esearch
    :return:
        nested Dictionary containing the detailed data (in XML)
    """
    ids = ','.join(id_list)
    Entrez.email = 'emails@examples2.com'
    handle = Entrez.efetch(db='pubmed',
                           retmode='xml',
                           id=ids)
    results = None
    max_retries = 3
    for attempt in range(max_retries):
        try:
            xml_data = handle.read()
            results = Entrez.read(io.BytesIO(xml_data), validate=False)
            break  # Break out of the loop if successful
        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {e}")
    else:
        print("Failed after maximum retries.")
    handle.close()
    return results

def get_article_IDs(extract_params) -> list:
    """
    Function that extracts article IDs for later fetching in a batch-wise manner as specified by a window.
    :param
        extract_params:
            Parameters defining start and end date of the query as well as a window iterator
    :return
        list of IDs
    """
    result_dicts = {}
    start_date = datetime.strptime(extract_params['start_date'], '%Y/%m/%d')
    end_date = datetime.strptime(extract_params['end_date'], '%Y/%m/%d')
    window_duration = timedelta(days=extract_params['window_duration_days'])  # timedelta(days=30)
    current_date = start_date
    window_end = start_date + window_duration

    # Loop over time windows of 2 months
    last_iteration = False
    while window_end <= end_date:
        returned_dicts = search('Intelligence', current_date.strftime('%Y/%m/%d'), window_end.strftime('%Y/%m/%d'))

        # accumulate dictionary values
        for key, value in returned_dicts.items():
            if key in result_dicts:
                if isinstance(value, list):
                    if isinstance(result_dicts[key], list):
                        # If both are lists, extend the existing list with the new list
                        result_dicts[key].extend(value)
                    else:
                        # If the existing value is not a list, create a new list with both values
                        result_dicts[key] = [result_dicts[key]] + value
                else:
                    if isinstance(result_dicts[key], list):
                        # If the existing value is a list, append the new value to it
                        result_dicts[key].append(value)
                    else:
                        # If neither is a list, create a list with both values
                        result_dicts[key] = [result_dicts[key], value]
            else:
                # Add the key-value pair to result_dicts
                result_dicts[key] = value
        print(f"window end date processed:{window_end}")
        current_date = window_end
        window_end = current_date + window_duration
        if last_iteration:
            break
        elif window_end > end_date:
            window_end = end_date
            last_iteration = True

    return result_dicts['IdList']

def process(text: str, nlp, model):
    doc = nlp(str(text))
    sents = [str(sent) for sent in doc.sents]
    embeddings = model.encode(sents)
    return list(doc.sents), embeddings
def cluster_text(sents, vecs, threshold):
    """
    Function that takes a list of sentences and clusters them into similar paragraphs based on cosine similarity

    :param sents: list of sentences
    :param vecs: list of embedding vectors
    :param threshold: threshold for cosine similarity
    :return: list of lists where each sublist is one cluster
    """
    clusters = [[0]]
    for i in range(1, len(sents)):
        if cosine_similarity(vecs[i].reshape(1, -1), vecs[i-1].reshape(1, -1)) < threshold:
            clusters.append([]) #create nwe cluster if similary < threshold
        clusters[-1].append(i) #if similarity > threshold add current sentence to the last cluster in the clusters list
    return clusters
def get_paragraphs(abstract_text, nlp, model):

    # Initialize the clusters lengths list and final texts list
    clusters_lens = []
    final_texts = []

    # Process the chunk
    initial_threshold = 0.8
    max_iteration = 10
    sents, vecs = process(abstract_text, nlp, model)

    # Cluster the sentences
    clusters = cluster_text(sents, vecs, initial_threshold)

    for cluster in clusters:
        cluster_txt = ' '.join([sents[i].text for i in cluster])
        cluster_len = len(cluster_txt)
        print(f"cluster_txt: {cluster_txt}")


        # Check if the cluster is too long
        if cluster_len > 800:
            iterator = 1
            # Track the best subcluster lengths
            while cluster_len > 800 and iterator < max_iteration:
                div_lens = []
                div_texts = []
                len_collector = []
                threshold = min(initial_threshold+(0.02*iterator), 0.99)
                sents_div, vecs_div = process(cluster_txt, nlp=nlp, model=model)
                reclusters = cluster_text(sents_div, vecs_div, threshold)

                for subcluster in reclusters:
                    div_txt = ' '.join([sents_div[i].text for i in subcluster])
                    div_len = len(div_txt)
                    len_collector.append(div_len)
                    print(f"div_len: {div_len}") #for debugging

                    if div_len > 60 and div_len < 800:
                        div_lens.append(div_len)
                        div_texts.append(div_txt)

                    cluster_len = max(len_collector)

                iterator+=1 #for debugging
                print(f"subclusters {iterator}: {div_lens}")

            clusters_lens.extend(div_lens)
            final_texts.extend(div_texts)
        else:
            clusters_lens.append(cluster_len)
            final_texts.append(cluster_txt)
    return final_texts