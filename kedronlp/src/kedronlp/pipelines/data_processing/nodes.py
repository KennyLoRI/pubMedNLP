import pandas as pd
from kedronlp.extract_utils import get_article_IDs, fetch_details

# def preprocess_companies(companies: pd.DataFrame) -> pd.DataFrame:
#     """Preprocesses the data for companies.
#
#     Args:
#         companies: Raw data.
#     Returns:
#         Preprocessed data, with `company_rating` converted to a float and
#         `iata_approved` converted to boolean.
#     """
#     companies["iata_approved"] = _is_true(companies["iata_approved"])
#     companies["company_rating"] = _parse_percentage(companies["company_rating"])
#     return companies


def extract_data(extract_params):
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

    studiesIdList = get_article_IDs(extract_params)
    #studies = fetch_details(studiesIdList)
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
                descriptor_list.append("No Data")
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
