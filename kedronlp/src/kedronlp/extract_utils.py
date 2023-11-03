from Bio import Entrez

def search(query):
    """
    Function to access IDs to queried data using Entrez
    :param
        query: term for which to search
    :return:
        Dictionary with the following keys: 'Count', 'RetMax', 'RetStart', 'IdList', 'TranslationSet', 'QueryTranslation'
    """
    #docs: https://www.ncbi.nlm.nih.gov/books/NBK25499/#chapter4.ESearch
    Entrez.email = 'email@example.com'
    handle = Entrez.esearch(db='pubmed',
                            sort='relevance',
                            retmax='1',
                            retmode='xml',
                            term=query,
                            mindate=2013,
                            maxdate=2023)
    results = Entrez.read(handle)
    return results

def fetch_details(id_list):
    """
    Function to fetch detailed data for a list of ID's of articles in Pubmed

    :param
        id_list: list of IDs from esearch
    :return: nested Dictionary containing the detailed data (in XML)
    """
    ids = ','.join(id_list)
    Entrez.email = 'email@example.com'
    handle = Entrez.efetch(db='pubmed',
    retmode='xml',
    id=ids)
    results = Entrez.read(handle)
    return results