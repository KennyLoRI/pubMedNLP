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
from kedronlp.modelling_utils import extract_abstract, print_context_details, instantiate_llm

def get_user_query(): #TODO: here we can think of a way to combine embeddings of previous queries
    #load model
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO", device=device)
    #get input
    user_input = input("Please enter your question: ")
    #embeddings = model.encode(user_input)
    return user_input


def modelling_answer(user_input, top_k_docs, modelling_params):
    # Define a prompt
    template = """Answer the question as short as possible and only based on the following context:
      {context}
      Question: {question}"""  # TODO put this into the parameters.yml file
    prompt = PromptTemplate(template=modelling_params["prompt_template"], input_variables=["context", "question"])

    # prepare context for prompt
    context = top_k_docs.values.flatten().tolist()
    input_dict = extract_abstract(context=context, question=user_input)

    # create chain
    llm = instantiate_llm()
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    # Reading & Responding
    response = llm_chain.run(input_dict)

    # print context details
    print_context_details(context=context)

def top_k_retrieval(user_input, top_k_params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vectordb = get_langchain_chroma(device=device)

    if top_k_params["retrieval_strategy"] == "similarity_search":
        docs = vectordb.similarity_search(user_input, k=top_k_params["top_k"])
    if top_k_params["retrieval_strategy"] == "max_marginal_relevance":
        # enforces more diversity of the top_k documents
        docs = vectordb.max_marginal_relevance_search(user_input, k=top_k_params["top_k"])

    return pd.DataFrame([doc.page_content for doc in docs])







