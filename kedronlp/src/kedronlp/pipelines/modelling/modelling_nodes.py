import pandas as pd
import torch
import regex as re
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from kedronlp.embedding_utils import get_langchain_chroma
from kedronlp.modelling_utils import extract_abstract, get_context_details, instantiate_llm, extract_date_range
from spellchecker import SpellChecker
from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, MultiQueryRetriever
import spacy
from textblob import TextBlob
import sys
from langchain.chains.query_constructor.base import (
    StructuredQueryOutputParser,
    get_query_constructor_prompt,
    AttributeInfo,
)
import ast
import itertools

def get_user_query(modelling_params, is_evaluation = False, **kwargs): #TODO: here we can think of a way to combine embeddings of previous queries
    #print user information
    print("""
    Welcome to our PubMed RAG System. 
    Get ready for chatting with over 190k medical abstracts!
    
    ***********
    To get the most out of this system, enclose special abbreviations or weird medical terms in asterisks (*word*). 
    Example question: What is the *TT100K* dataset?
    Thanks and have fun!
    """)

    # Obtain query
    spell = SpellChecker()
    nlp = spacy.load('en_core_web_sm')
    if not is_evaluation:
        user_input = input("Please enter your question (use *word* for abbreviations or special terms): ")
    else:
        evaluation_input = kwargs.get("evaluation_input", None) # to get evaluation input: get_user_query(is_evaluation=True, evaluation_input = "input_string")
        user_input = evaluation_input

    # Correct query
    if modelling_params["spell_checker"] == True:
        # Treat highlighted words and qustion mark specifically
        pattern = r'\*(.*?)\*'  # Regular expression to match words enclosed in *
        highlighted_words = re.findall(pattern, user_input)
        question_mark = '?' if '?' in user_input else ''

        # Apply spell correction excluding asterisked words
        try:
            spell = SpellChecker()
            corrected_list = [spell.correction(token.strip('?').strip('*')) if token.strip('?').strip('*') not in highlighted_words else token.strip('?').strip('*') for token in user_input.split()]
            correct_query = ' '.join(corrected_list)+question_mark
        except:
            correct_query = user_input
    else:
        correct_query = user_input

    # Extract metadata-filter intention out of query
    if modelling_params["metadata_strategy"] == "parser":
        author_names = []
        paper_titles = []

        doc_correct = nlp(correct_query)

        #extract authors & titles using NER
        for ent in doc_correct.ents:
            if ent.label_ == "PERSON":
                author_names.append(ent.text)
            elif ent.label_ == "WORK_OF_ART":
                paper_titles.append(ent.text)

        # extract dates using handcrafted rules
        date_range = extract_date_range(doc_correct)

        #output query filters
        structured_query = {
            "query": correct_query,
            "author_names": author_names,
            "publishing_dates": date_range,
            "paper_titles": paper_titles
        }

    if modelling_params["metadata_strategy"] == "llm_detection":
        # Define filter structure & prompt template
        metadata_field_info = [
            AttributeInfo(
                name="title", #todo check if this should be excluded in evaluation
                description="Title of the paper",
                type="string",
            ),
            AttributeInfo(
                name="year",
                description="The year the paper was published",
                type="integer",
            ),
            AttributeInfo(
                name="authors",
                description="The authors who wrote the paper",
                type="string",
            )
        ]

        document_content_description = "Brief summary of a scientific paper"
        prompt = get_query_constructor_prompt(
            document_content_description,
            metadata_field_info,
        )

        # Instantiate llm for creating the structured query
        llm = instantiate_llm(modelling_params["temperature"],
                              modelling_params["max_tokens"],
                              modelling_params["n_ctx"],
                              modelling_params["top_p"],
                              modelling_params["n_gpu_layers"],
                              modelling_params["n_batch"],
                              modelling_params["verbose"], )

        #define output parser and create the query construction pipeline
        output_parser = StructuredQueryOutputParser.from_components()
        query_constructor = prompt | llm | output_parser

        # get structured query for user input
        structured_query = query_constructor.invoke(
            {
                "query": correct_query
            }
        )


    return correct_query, str(structured_query)


def modelling_answer(user_input, top_k_docs, modelling_params):
    # Define a prompt
    prompt = PromptTemplate(template=modelling_params["prompt_template"], input_variables=["context", "question"])

    # obtain context for prompt
    context = top_k_docs.values.flatten().tolist()

    # If no context retrieved inform the user that no data to the query is available
    if not context:
        print("""Unfortunately I have no information on your question in my database. 
              This might be the case since I only consider abstracts from Pubmed that match the keyword intelligence. 
              Furthermore, I only consider papers published between 2013 and 2023. 
              In case your question matches these requirements please try reformulating your query""")
        sys.exit()

    # extract and structure context for input
    input_dict = get_context_details(context=context, print_context = False, as_input_dict = True, user_input = user_input, abstract_only = modelling_params["abstract_only"])

    # create chain
    llm = instantiate_llm(modelling_params["temperature"],
                          modelling_params["max_tokens"],
                          modelling_params["n_ctx"],
                          modelling_params["top_p"],
                          modelling_params["n_gpu_layers"],
                          modelling_params["n_batch"],
                          modelling_params["verbose"],)

    llm_chain = LLMChain(prompt=prompt, llm=llm)

    # Reading & Responding
    response = llm_chain.run(input_dict)

    # If response is empty, save the retrieved context but print apologies statement
    if not response or len(response.strip()) == 0:
        context_dict = get_context_details(context=context, print_context=False)

        print("""Answer: Unfortunately I have no information on your question at hand. 
        This might be the case since I only consider abstracts from Pubmed that match the keyword intelligence. 
        Furthermore, I only consider papers published between 2013 and 2023. 
        In case your question matches these requirements please try reformulating your query""")

        response = """Answer: Unfortunately I have no information on your question at hand. 
        This might be the case since I only consider abstracts from Pubmed that match the keyword intelligence. 
        Furthermore, I only consider papers published between 2013 and 2023. 
        In case your question matches these requirements please try reformulating your query"""

    # print and save context details
    else:
        context_dict = get_context_details(context=context)

    return pd.DataFrame({"response": response, "query": user_input, **context_dict}) #TODO check if working when response empty

def top_k_retrieval(user_input, top_k_params, modelling_params, user_query_filters):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vectordb = get_langchain_chroma(device=device)
    user_query_filters = ast.literal_eval(user_query_filters)
    start, end = user_query_filters["publishing_dates"]
    author_names = user_query_filters["author_names"]
    paper_titles = user_query_filters["paper_titles"]
    # start, end = 2021, 2023
    # author_names = ["Bill Gates", "Steve Jobs", "Elon Musk"]
    # paper_titles = ["a paper title", "another paper title"]

    filter = {}
    strategy = modelling_params["metadata_strategy"]
    if strategy == "parser" or strategy == "llm_detection":
        publishing_dates_filter = {}
        if start != None:
            gte_start = {
                "Year": {
                    "$gte": start
                }
            }
        if end != None:
            lte_end = {
                "Year": {
                    "$lte": end
                }
            }

        if start != None and end != None:
            publishing_dates_filter = {
                "$and": [gte_start, lte_end]
            }
        if start != None and end == None:
            publishing_dates_filter = gte_start
        if start == None and end != None:
            publishing_dates_filter = lte_end
        
        author_names_filter = {}
        if author_names:
            if len(author_names) == 1:
                author_name = author_names[0]
                author_names_filter = {
                    "Authors": {
                        "$eq": author_name
                    }
                }
            else:
                author_permutations = list(itertools.permutations(author_names))
                author_permutations = [", ".join(authors) for authors in author_permutations]
                author_names_filter = {
                    "Authors": {
                        "$in": author_permutations
                    }
                }

        paper_titles_filter = {}
        if paper_titles:
            paper_titles_filter = {
                "Title": {
                    "$in": paper_titles
                }
            }

        potential_filters = [publishing_dates_filter, author_names_filter, paper_titles_filter]
        filter = {
            "$and": [filter for filter in potential_filters if filter]
        }
        if len(filter["$and"]) == 1:
            filter = filter["$and"][0]
        elif len(filter["$and"]) == 0:
            filter = None

    #basic similarity search
    if top_k_params["retrieval_strategy"] == "similarity":
        docs = vectordb.similarity_search(user_input, k=top_k_params["top_k"], filter=filter)
        print(f"vectordb:{vectordb}, user_input:{user_input}, number_docs_in_chroma:{vectordb._collection.count()}")

    #diversity enforcing similarity search
    if top_k_params["retrieval_strategy"] == "max_marginal_relevance":
        # enforces more diversity of the top_k documents
        docs = vectordb.max_marginal_relevance_search(user_input, k=top_k_params["top_k"])

    #hybrid similarity search including BM25 for keyword
    if top_k_params["retrieval_strategy"] == "ensemble_retrieval":
        #initiate BM25 retriever
        lang_docs = [Document(page_content=doc) for doc in vectordb.get().get("documents", [])] # TODO: status quo is an inefficient workaround - no chroma bm25 integration yet
        bm25_retriever = BM25Retriever.from_documents(lang_docs)
        bm25_retriever.k = top_k_params["top_k"]

        #initiate similarity retriever
        similarity_retriever = vectordb.as_retriever(search_kwargs={"k": top_k_params["top_k"], "search_type": top_k_params["advanced_dense_retriever"]})

        #initiate ensemble (uses reciprocal rank fusion in the background with default settings)
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, similarity_retriever], weights=[0.5, 0.5]
        )
        docs = ensemble_retriever.get_relevant_documents(user_input)

    # Given a query, use an LLM to write a set of queries (default: 3).
    # Retrieve docs for each query. Return the unique union of all retrieved docs.
    if top_k_params["retrieval_strategy"] == "multi_query_retrieval":
        llm = instantiate_llm(modelling_params["temperature"],
                              modelling_params["max_tokens"],
                              modelling_params["n_ctx"],
                              modelling_params["top_p"],
                              modelling_params["n_gpu_layers"],
                              modelling_params["n_batch"],
                              modelling_params["verbose"])
        # TODO: not working yet, since generate_queries function of .from_llm falsely creates empty strings.
        # Note: Generated queries were not of high quality since the used llm is not super powerful.
        multiquery_llm_retriever = MultiQueryRetriever.from_llm(
            retriever = vectordb.as_retriever(),
            llm=llm,
            include_original = top_k_params["mq_include_original"]
        )
        docs = multiquery_llm_retriever.get_relevant_documents(query=user_input)
    top_k_df = pd.DataFrame([doc.page_content for doc in docs])
    return top_k_df.drop_duplicates().head(top_k_params["top_k"]) # makes sure that only top_k_params["top_k"] docs  are returned also in ensemble & multiquery method





