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
from spellchecker import SpellChecker
import string

def get_user_query(): #TODO: here we can think of a way to combine embeddings of previous queries
    #load model
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO", device=device)
    #get input

    # Create a SpellChecker object
    spell = SpellChecker()

    # Load spaCy English language model
    nlp = spacy.load('en_core_web_sm')

    # Get user input
    user_input = input("Please enter your question: ")

    # tokenize etc
    doc = nlp(user_input)

    # Correct each token if necessary. If work unknown spell() returns None - Then use original word (medical terms)
    corrected_list = [spell.correction(token.text) + token.whitespace_ if spell.correction(
        token.text) is not None else token.text + token.whitespace_ for token in doc]

    correct_query = ''.join(corrected_list)

    #embeddings = model.encode(user_input)
    return correct_query


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
    top_k_df = pd.DataFrame([doc.page_content for doc in docs])
    return top_k_df.drop_duplicates() #drop duplicates







