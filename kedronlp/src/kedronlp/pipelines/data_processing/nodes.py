import pandas
import pandas as pd
from sentence_transformers import SentenceTransformer
import spacy
from functools import partial
import torch
from kedronlp.extract_utils import get_article_IDs, fetch_details, get_paragraphs

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

def process_extract_data(extract_data):
    spacy.cli.download("en_core_web_sm")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO", device=device)
    nlp = spacy.load("en_core_web_sm")

    # Create a partial function with constant arguments
    partial_get_paragraphs = partial(get_paragraphs, nlp=nlp, model=model)
    extract_data_clean = extract_data.dropna().head(3) #remove head when not testing
    extract_data_clean['paragraphs'] = extract_data_clean['Abstract'].apply(partial_get_paragraphs)
    return extract_data_clean

