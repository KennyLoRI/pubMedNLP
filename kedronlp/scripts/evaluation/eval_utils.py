import pandas as pd
import torch
import regex as re
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from embedding_utils import get_langchain_chroma
from modelling_utils import get_context_details, instantiate_llm, extract_date_range
from spellchecker import SpellChecker
from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, MultiQueryRetriever
import spacy
from langchain.chains.query_constructor.base import (
    StructuredQueryOutputParser,
    get_query_constructor_prompt,
    AttributeInfo,
)
import ast
import os

# taken from chat pipeline with small adjustments
def get_predictions(llm, query_list, modelling_params, top_k_params, retriever):

    prompt = PromptTemplate(template=modelling_params["prompt_template"], input_variables=["context", "question"])

    # create chain
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    nlp = spacy.load('en_core_web_sm')

    query_responses = []
    contexts = []
    for user_input in query_list:
        user_input = str(user_input) # make sure it is a string
        print(f"Provided question: {user_input}")

        # Correct query
        if modelling_params["spell_checker"] == True:
            # Treat highlighted words and qustion mark specifically
            pattern = r'\*(.*?)\*'  # Regular expression to match words enclosed in *
            highlighted_words = re.findall(pattern, user_input)
            question_mark = '?' if '?' in user_input else ''

            # Apply spell correction excluding asterisked words
            try:
                corrected_list = [token.strip('?').strip('*') if token.strip('?').strip(
                    '*') not in highlighted_words else token.strip('?').strip('*') for token in user_input.split()]
                correct_query = ' '.join(corrected_list) + question_mark
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

            #define output parser and create the query construction pipeline
            output_parser = StructuredQueryOutputParser.from_components()
            query_constructor = prompt | llm | output_parser

            # get structured query for user input
            structured_query = query_constructor.invoke(
                {
                    "query": correct_query
                }
            )

        metadata_strategy = modelling_params["metadata_strategy"]
            
        user_input = correct_query

        filter = {}
        if metadata_strategy == "parser" or metadata_strategy == "llm_detection":
            user_query_filters = str(structured_query)
            user_query_filters = ast.literal_eval(user_query_filters)
            start, end = user_query_filters["publishing_dates"]
            author_names = user_query_filters["author_names"]
            paper_titles = user_query_filters["paper_titles"]
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
                author_names = ", ".join(author_names)
                author_names_filter = {
                    "Authors": {
                        "$eq": author_names
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

            print(f"filter: {filter}")
        else:
            filter = None

        #basic similarity search
        if top_k_params["retrieval_strategy"] == "similarity":
            docs = retriever.similarity_search(user_input, k=top_k_params["top_k"], filter=filter)

        #diversity enforcing similarity search
        if top_k_params["retrieval_strategy"] == "max_marginal_relevance":
            # enforces more diversity of the top_k documents
            docs = retriever.max_marginal_relevance_search(user_input, k=top_k_params["top_k"], filter=filter)

        #hybrid similarity search including BM25 for keyword
        if top_k_params["retrieval_strategy"] == "ensemble_retrieval":
            docs = retriever.get_relevant_documents(user_input, metadata=filter)

        # Given a query, use an LLM to write a set of queries (default: 3).
        # Retrieve docs for each query. Return the unique union of all retrieved docs.
        if top_k_params["retrieval_strategy"] == "multi_query_retrieval":
            # TODO: not working yet, since generate_queries function of .from_llm falsely creates empty strings.
            # Note: Generated queries were not of high quality since the used llm is not super powerful.
            multiquery_llm_retriever = MultiQueryRetriever.from_llm(
                retriever = retriever.as_retriever(),
                llm=llm,
                include_original = top_k_params["mq_include_original"]
            )
            docs = multiquery_llm_retriever.get_relevant_documents(query=user_input, metadata=filter)

        top_k_docs = pd.DataFrame([doc.page_content for doc in docs])
        top_k_docs = top_k_docs.drop_duplicates().head(top_k_params["top_k"])

        # obtain context for prompt
        context = top_k_docs.values.flatten().tolist()

        # If no context retrieved inform the user that no data to the query is available
        if not context:
            response = """Unfortunately I have no information on your question in my database. 
                This might be the case since I only consider abstracts from Pubmed that match the keyword intelligence. 
                Furthermore, I only consider papers published between 2013 and 2023. 
                In case your question matches these requirements please try reformulating your query"""
            query_responses.append(response)
            continue

        # extract and structure context for input
        input_dict = get_context_details(context=context, top_k_params=top_k_params, print_context = False, as_input_dict = True, user_input = user_input, abstract_only = modelling_params["abstract_only"])
        # Reading & Responding
        response = llm_chain.invoke(input_dict)["text"]

        # If response is empty, save the retrieved context but print apologies statement
        if not response or len(response.strip()) == 0:

            response = """Answer: Unfortunately I have no information on your question at hand. 
            This might be the case since I only consider abstracts from Pubmed that match the keyword intelligence. 
            Furthermore, I only consider papers published between 2013 and 2023. 
            In case your question matches these requirements please try reformulating your query"""

        query_responses.append(response)
        retrieved_passages = []
        if modelling_params["abstract_only"]:
            passages = input_dict["context"].split("\n\n\n\n")
            for passage in passages:
                retrieved_passages.append(passage)
        elif not modelling_params["abstract_only"]:
            for a_context in context:
                pos = a_context.rfind(": ")
                passage = a_context[pos+2:]
                retrieved_passages.append(passage)
        contexts.append(retrieved_passages)

    return query_responses, contexts