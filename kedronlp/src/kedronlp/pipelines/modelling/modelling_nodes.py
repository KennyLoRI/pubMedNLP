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


def modelling_answer(user_input):
    # Define a prompt
    template = """Answer the question as short as possible and only based on the following context:
      {context}
      Question: {question}"""  # TODO put this into the parameters.yml file
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    # get top k documents for user query
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vectordb = get_langchain_chroma(device=device) #TODO: make sure thi sis not setting up a new chromadb store but just loads it
    docs = vectordb.similarity_search(user_input, k=3)  # TODO: Finetune doc retrieval

    # prepare context for prompt
    context = [doc.page_content for doc in docs]
    input_dict = extract_abstract(context=context, question=user_input)

    # create chain
    llm = instantiate_llm()
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    # Reading & Responding
    response = llm_chain.run(input_dict)

    # print context details
    print_context_details(context=context)

