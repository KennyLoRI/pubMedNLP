import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores.chroma import Chroma
from langchain import PromptTemplate
from langchain.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from kedronlp.embedding_utils import get_langchain_chroma
from kedronlp.modelling_utils import extract_abstract, print_context_details, instantiate_llm, is_within_range
from spellchecker import SpellChecker
from langchain.schema import Document
from langchain.retrievers import BM25Retriever, EnsembleRetriever, MultiQueryRetriever
import spacy
import sys
from langchain.chains.query_constructor.base import (
    StructuredQueryOutputParser,
    get_query_constructor_prompt,
    AttributeInfo,
)

import string

def get_user_query(modelling_params, is_evaluation = False, **kwargs): #TODO: here we can think of a way to combine embeddings of previous queries
    #load model
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO", device=device)
    #get input

    # Obtain query
    spell = SpellChecker()
    nlp = spacy.load('en_core_web_sm')
    if not is_evaluation:
        user_input = input("Please enter your question: ")
    else:
        evaluation_input = kwargs.get("evaluation_input", None) # to get evaluation input: get_user_query(is_evaluation=True, evaluation_input = "input_string")
        user_input = evaluation_input

    # Correct query
    doc = nlp(user_input)
    corrected_list = [spell.correction(token.text) + token.whitespace_ if spell.correction(
        token.text) is not None else token.text + token.whitespace_ for token in doc] # If word unknown spell() returns None - Then use original word (medical terms)
    correct_query = ''.join(corrected_list)

    # Extract metadata-filter intention out of query
    if modelling_params["metadata_strategy"] == "parser":
        author_names = []
        publishing_dates = []
        paper_titles = []
        for ent in doc.ents:
            if ent.label_ == "PERSON":  # Assuming authors are labeled as PERSON entities
                author_names.append(ent.text)
            elif ent.label_ == "DATE":
                publishing_dates.append(ent.text)
            elif ent.label_ == "WORK_OF_ART":  # Assuming titles are labeled as WORK_OF_ART entities
                paper_titles.append(ent.text)

        structured_query = {
            "author_names": author_names,
            "publishing_dates": publishing_dates,
            "paper_titles": paper_titles
        }

    if modelling_params["metadata_strategy"] == "llm_detection":
        # Define filter structure & prompt template
        metadata_field_info = [
            AttributeInfo(
                name="major topics", #todo check if this should be excluded in evaluation
                description="The major topics covered in the paper",
                type="string",
            ),
            AttributeInfo(
                name="year",
                description="The year the paper was published",
                type="integer",
            ),
            AttributeInfo(
                name="month",
                description="The month the paper was published",
                type="string",
            )
        ]

        document_content_description = "Abstract of a scientific paper"
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

    # prepare context for prompt
    context = top_k_docs.values.flatten().tolist()
    if not context:
        print("""Unfortunately I have no information on your question at hand. 
              This might be the case since I only consider abstracts from Pubmed that match the keyword intelligence. 
              Furthermore, I only consider papers published between 2013 and 2023. 
              In case your question matches these requirements please try reformulating your query""")
        sys.exit()

    input_dict = extract_abstract(context=context, question=user_input)

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
    if not response or len(response.strip()) == 0:

        #catch context for debugging but don't print
        context_dict = print_context_details(context=context, print_context=False)

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
        context_dict = print_context_details(context=context)

    return pd.DataFrame({"response": response, "query": user_input, **context_dict}) #TODO check if working when response empty

def top_k_retrieval(user_input, top_k_params, modelling_params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vectordb = get_langchain_chroma(device=device)

    #basic similarity search
    if top_k_params["retrieval_strategy"] == "similarity":
        docs = vectordb.similarity_search(user_input, k=top_k_params["top_k"])
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





