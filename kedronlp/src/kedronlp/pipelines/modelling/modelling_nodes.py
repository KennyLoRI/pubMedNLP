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
from langchain.schema import Document
from langchain.retrievers import BM25Retriever, EnsembleRetriever, MultiQueryRetriever
import spacy

import string

def get_user_query(is_evaluation = False, **kwargs): #TODO: here we can think of a way to combine embeddings of previous queries
    #load model
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO", device=device)
    #get input

    # Create a SpellChecker object
    spell = SpellChecker()

    # Load spaCy English language model
    nlp = spacy.load('en_core_web_sm')

    if not is_evaluation: #TODO include evaluation logic throughout the whole pipeline
        # Get user input
        user_input = input("Please enter your question: ")
    else:
        evaluation_input = kwargs.get("evaluation_input", None) #to get evaluation input: get_user_query(is_evaluation=True, evaluation_input = "input_string")
        user_input = evaluation_input

    # tokenize etc
    doc = nlp(user_input)

    # Correct each token if necessary. If work unknown spell() returns None - Then use original word (medical terms)
    corrected_list = [spell.correction(token.text) + token.whitespace_ if spell.correction(
        token.text) is not None else token.text + token.whitespace_ for token in doc]

    correct_query = ''.join(corrected_list)

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
    if not response:
        print("""Unfortunately I have no information on your question at hand. 
              This might be the case since I only consider abstracts from Pubmed that match the keyword intelligence. 
              Furthermore, I only consider papers published between 2013 and 2023. 
              In case your question matches these requirements please try reformulating your query""")

    # print and save context details
    context_dict = print_context_details(context=context)

    return pd.DataFrame({"response": response, "query": user_input, **context_dict })

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


def evaluate(llm_response, reference_data):

    return

def create_reference_data():
    import pandas as pd

    data = {
        "question_type": [
            "Out of domain questions",
            "Out of domain questions",
            "In domain questions",
            "In domain questions",
            "Query contains named entities that should be treated in a special way",
            "Query contains named entities that should be treated in a special way",
            "Handle fully specified questions that might contain stop words etc.",
            "Handle fully specified questions that might contain stop words etc.",
            "Fully support both semantic search and lexicographical search",
            "Fully support both semantic search and lexicographical search",
            "Confirmation Questions [yes or no]",
            "Confirmation Questions [yes or no]",
            "Factoid-type Questions [what, which, when, who, how]",
            "Factoid-type Questions [what, which, when, who, how]",
            "List-type Questions",
            "List-type Questions",
            "Causal Questions [why or how]",
            "Causal Questions [why or how]",
            "Hypothetical Questions",
            "Hypothetical Questions",
            "Complex Questions",
            "Complex Questions",
            "Test for edge cases",
            "Test for edge cases",
            "Time based questions",
            "Time based questions",
            "Location based questions",
            "Location based questions",
            "Cause and effect questions",
            "Cause and effect questions",
            "Scenario based questions",
            "Scenario based questions",
            "Fact based questions",
            "Fact based questions",
            "Descriptive questions",
            "Descriptive questions",
            "Procedural questions",
            "Procedural questions",
            "Comparative questions",
            "Comparative questions",
        ],
        "question": [
            "How tall is the Golden Gate Bridge?",
            "Where is the Eiffel Tower located?",
            "What are the predictors of nurses' attitudes towards communication?",
            "What is Autism Spectrum Disorder (ASD) characterized by?",
            "What is the TT100K dataset?",
            "What is the full form of EHR? / Tell me about the strenghts and limitations of EHR (Electronic Health Records)",
            "Tell me about the recent neuroscience evidence to elucidate how general intelligence, g, emerges from individual differences in the network architecture of the human brain?",
            "The best fit between brain traits and degrees of intelligence among mammals is reached by?",
            "(Semantic query) What is intelligence?",
            'Lexicographical query) Which paper was published first? Title "Fluid intelligence is related to capacity in memory as well as attention: Evidence from middle childhood and adulthood" or "What is theory of mind? A psychometric study of theory of mind and intelligence"?',
            "Are forms of narcissism (grandiose and vulnerable) related to objective intelligence?",
            "For children who participated in a task-switching experiment, where the children performed a task repeatedly (single-trial blocks) or switched between two different tasks (mixed-trial blocks), did intellectually gifted children perform quicker than the average group for both mixed and single-trial blocks?",
            "What is the Flynn Effect, and how does it change our understanding of IQ?",
            "How is emotional exhaustion related to burnout?",
            "List the different types of intelligence?",
            "List some abreivations used in the field of intelligence in PubMed",
            "Why Did Cephalopods Evolve Intelligence?",
            "Why is intelligence associated with stability of happiness?",
            "What would happen if there was a direct link between diet and cognitive performance in old age?",
            "What would happen if business intelligence was used to manage supply costs?",
            "What themes emerged in the study of the use of emotional intelligence capabilities in clinical reasoning and decision-making?",
            "What tools and methods were employed in the study of the relationship between intellectual development and intrinsic motivation?",
            "Jochen Kruppa, Yufeng Liu, Gérard Biau, Michael Kohler, Inke R. König, James D. Malley, and Andreas Ziegle",
            "sdkfjasdkjfhasdkjfhaskjdfh",
            'When was the paper "On the visual analytic intelligence of neural networks" published?',
            'When was the paper "The neuroscience of empathy and compassion in pro-social behavior" published?',
            "In which country is the Individuals with Disabilities Education Act (IDEA) located?",
            "Fill in the blank: Impact of Artificial Intelligence on Regional Green Development under _____'s Environmental Decentralization System-Based on Spatial Durbin Model and Threshold Effect",
            "What are the causes of international differences in student assessment and psychometric IQ test results?",
            "What is significantly associated with higher intelligent quotient (IQ) scores?",
            "What would happen in a scenario where you lower the IQ of a person?",
            "What are the potential applications of AI systems in different nursing care settings?",
            "What is the TT100K dataset?",
            "Are forms of narcissism (grandiose and vulnerable) related to objective intelligence?",
            "Tell me about the practical implementation of Industrial Internet of Things over 5G",
            "Tell me about End-to-End Automated License Plate Recognition System Using YOLO Based Vehicle and License Plate Detection with Vehicle Classification",
            "",
            "",
            "What IQ is higher - 100 or 120?",
            "What is the difference between IQ and EQ?",
        ],
        "answer_human": [
            "Should deny the question as it is out of domain",
            "Should deny the question as it is out of domain",
            "Empathy and emotional intelligence are predictors of nurses' attitudes towards communication, and the cognitive dimension of attitude is a good predictor of the behavioural dimension of attitudes towards communication of nurses in both regression models and fuzzy-set qualitative comparative analysis.",
            "Autism spectrum disorders (ASD) are characterized by persistent deficits in social communication and social interaction across contexts, and are associated with restricted patterns of behavior. The developmental quotient (DQ) is based on the developmental age and chronological age of children.",
            "The Tsinghua-Tencent 100K (TT100K) traffic sign dataset. (It is a dataset for traffic sign detection, which contains 100,000 images with 224,024 annotated traffic signs.)",
            "Electronic health records (EHRs) are digital records of health information. Electronic health record (EHR) was hailed as a major step towards making healthcare more transparent and accountable. All the developed nations digitised their health records which were meant to be safe, secure and could be accessed on demand. This was intended to benefit all stakeholders. However, the jury is still out if the EHR has been worth it. There have been incidences of data breaches despite cybersecurity checks and of manipulation compromising clinicians' integrity and patients' safety. EHRs have also been blamed for doctor burnout in overloading them with a largely avoidable administrative burden. The lack of interoperability amongst various EHR software systems is creating obstacles in seamless workflow.",
            "The reviewed findings motivate new insights about how network topology and dynamics account for individual differences in g, represented by the Network Neuroscience Theory. According to this framework, g emerges from the small-world topology of brain networks and the dynamic reorganization of its community structure in the service of system-wide flexibility and adaptation.",
            "The best fit between brain traits and degrees of intelligence among mammals is reached by a combination of the number of cortical neurons, neuron packing density, interneuronal distance and axonal conduction velocity--factors that determine general information processing capacity (IPC), as reflected by general intelligence. The highest IPC is found in humans, followed by the great apes, Old World and New World monkeys.",
            "Intelligence is the ability to learn from experience and to adapt to, shape, and select environments. Intelligence as measured by (raw scores on) conventional standardized tests varies across the lifespan, and also across generations.",
            "Fluid intelligence is related to capacity in memory as well as attention: Evidence from middle childhood and adulthood which was published on 22 August 2019 as compared to the other paper which was published in August 2022",
            "No (Both forms of narcissism (grandiose and vulnerable) were unrelated to objective intelligence.)",
            "Yes (Intellectually gifted children performed quicker than the average group for both mixed and single-trial blocks.)",
            "In 1981, psychologist James Flynn noticed that IQ scores had risen streadily over nearly a century a staggering difference of 18 points over two generations. After a careful analysis, he concluded the cause to be culture. Society had become more intelligent-come to grips with bigger, more abstract ideas over time-and had made people smarter. This observation, combined with solid evidence that IQ scores are also not fixed within an individual, neatly dispels the idea of intelligence being an innate and fixed entity. While intelligence clearly has a biological component, it is best defined, as a set of continually developed skills.",
            "The data indicates that a worker's age influences his/her capacity to work with method and order, and that workers with emotional exhaustion (a basic feature of burnout) have lower scores in method and order. Greater emotional exhaustion and greater depersonalization were related to lower personal accomplishment and greater burnout.",
            'The different types of intelligence are: 1. Naturalist Intelligence ("Nature Smart") 2. Musical Intelligence ("Musical Smart") 3. Logical-Mathematical Intelligence (Number/Reasoning Smart) 4. Existential Intelligence 5. Interpersonal Intelligence (People Smart) 6. Bodily-Kinesthetic Intelligence ("Body Smart") 7. Linguistic Intelligence (Word Smart) 8. Intra-personal Intelligence (Self Smart) 9. Spatial Intelligence ("Picture Smart") (Any of the points if listed are counted as valid answers)',
            "Some abreivations used in the field of intelligence in PubMed are: 1. IQ (Intelligence Quotient) 2. G (General Intelligence) 3. EI (Emotional Intelligence) 4. CQ (Cultural Intelligence) 5. SQ (Social Intelligence) 6. PQ (Political Intelligence) 7. AQ (Adversity Intelligence) 8. MQ (Moral Intelligence) 9. FQ (Financial Intelligence) 10. RQ (Rational Intelligence) 11. TQ (Technical Intelligence) 12. EQ (Emotional Quotient) 13. SQ (Social Quotient) 14. PQ (Political Quotient) 15. AQ (Adversity Quotient) 16. MQ (Moral Quotient) 17. FQ (Financial Quotient) 18. RQ (Rational Quotient) 19. TQ (Technical Quotient) (Any of the points if listed are counted as valid answers)",
            "Here, we suggest that the loss of the external shell in cephalopods (i) caused a dramatic increase in predatory pressure, which in turn prevented the emergence of slow life histories, and (ii) allowed the exploitation of novel challenging niches, thus favouring the emergence of intelligence.",
            "In the National Child Development Study, life-course variability in happiness over 18 years was significantly negatively associated with its mean level (happier individuals were more stable in their happiness, and it was not due to the ceiling effect), as well as childhood general intelligence and all Big Five personality factors (except for Agreeableness). In a multiple regression analysis, childhood general intelligence was the strongest predictor of life-course variability in life satisfaction, stronger than all Big Five personality factors, including Emotional stability. More intelligent individuals were significantly more stable in their happiness, and it was not entirely because: (1) they were more educated and wealthier (even though they were); (2) they were healthier (even though they were); (3) they were more stable in their marital status (even though they were); (4) they were happier (even though they were); (5) they were better able to assess their own happiness accurately (even though they were); or (6) they were better able to recall their previous responses more accurately or they were more honest in their survey responses (even though they were both). While I could exclude all of these alternative explanations, it ultimately remained unclear why more intelligent individuals were more stable in their happiness.",
            "Our models show no direct link between diet and cognitive performance in old age; instead they are related via the lifelong-stable trait of intelligence. (No correct answer here)",
            "No definitive answer here",
            "Three themes emerged: the sensibility to engage EI capabilities in clinical contexts, motivation to actively engage with emotions in clinical decision-making and incorporating emotional and technical perspectives in decision-making.",
            "To test this hypothesis, we administered the Learning Context Questionnaire to measure intellectual development. In addition, we administered the Intrinsic Motivation Inventory to assess our students' intrinsic motivation. Furthermore, we performed regression analyses between intellectual development with both intrinsic motivation and class performance. The results document a positive relationship among intellectual development, intrinsic motivation, and class performance for female students only. In sharp contrast, there was a negative relationship between intellectual development, intrinsic motivation, and class performance for male students. The slope comparisons documented significant differences in the slopes relating intellectual development, intrinsic motivation, and class performance between female and male students. Thus, female students with more sophisticated beliefs that knowledge is personally constructed, complex, and evolving had higher intrinsic motivation and class performance. In contrast, male students with the naive beliefs that the structure of knowledge is simple, absolute, and certain had higher levels of intrinsic motivation and class performance.",
            'From the paper "What subject matter questions motivate the use of machine learning approaches compared to statistical models for probability prediction?"',
            "No definitive answer here",
            "25 September 2023",
            "20 August 2021",
            "United States of America",
            "China",
            "Education was rated by N = 71 experts as the most important cause of international ability differences. Genes were rated as the second most relevant factor but also had the highest variability in ratings. Culture, health, wealth, modernization, and politics were the next most important factors, whereas other factors such as geography, climate, test bias, and sampling error were less important.",
            "Happiness is significantly associated with IQ. Those in the lowest IQ range (70-99) reported the lowest levels of happiness compared with the highest IQ group (120-129). Mediation analysis using the continuous IQ variable found dependency in activities of daily living, income, health and neurotic symptoms were strong mediators of the relationship, as they reduced the association between happiness and IQ by 50%.",
            "Those with lower IQ are less happy than those with higher IQ. Interventions that target modifiable variables such as income (e.g. through enhancing education and employment opportunities) and neurotic symptoms (e.g. through better detection of mental health problems) may improve levels of happiness in the lower IQ groups.",
            "No definitive answer as scenario based questions are usually hypothetical.",
            "The Tsinghua-Tencent 100K (TT100K) traffic sign dataset. (It is a dataset for traffic sign detection, which contains 100,000 images with 224,024 annotated traffic signs.)",
            "No (Both forms of narcissism (grandiose and vulnerable) were unrelated to objective intelligence.)",
            "The next generation of mobile broadband communication, 5G, is seen as a driver for the industrial Internet of things (IIoT). The expected 5G-increased performance spanning across different indicators, flexibility to tailor the network to the needs of specific use cases, and the inherent security that offers guarantees both in terms of performance and data isolation have triggered the emergence of the concept of public network integrated non-public network (PNI-NPN) 5G networks. These networks might be a flexible alternative for the well-known (albeit mostly proprietary) Ethernet wired connections and protocols commonly used in the industry setting. With that in mind, this paper presents a practical implementation of IIoT over 5G composed of different infrastructure and application components. From the infrastructure perspective, the implementation includes a 5G Internet of things (IoT) end device that collects sensing data from shop floor assets and the surrounding environment and makes these data available over an industrial 5G Network. Application-wise, the implementation includes an intelligent assistant that consumes such data to generate valuable insights that allow for the sustainable operation of assets. These components have been tested and validated in a real shop floor environment at Bosch Termotecnologia (Bosch TT). Results show the potential of 5G as an enhancer of IIoT towards smarter, more sustainable, green, and environmentally friendly factories.",
            "An accurate and robust Automatic License Plate Recognition (ALPR) method proves surprising versatility in an Intelligent Transportation and Surveillance (ITS) system. However, most of the existing approaches often use prior knowledge or fixed pre-and-post processing rules and are thus limited by poor generalization in complex real-life conditions. In this paper, we leverage a YOLO-based end-to-end generic ALPR pipeline for vehicle detection (VD), license plate (LP) detection and recognition without exploiting prior knowledge or additional steps in inference. We assess the whole ALPR pipeline, starting from vehicle detection to the LP recognition stage, including a vehicle classifier for emergency vehicles and heavy trucks. We used YOLO v2 in the initial stage of the pipeline and remaining stages are based on the state-of-the-art YOLO v4 detector with various data augmentation and generation techniques to obtain LP recognition accuracy on par with current proposed methods. To evaluate our approach, we used five public datasets from different regions, and we achieved an average recognition accuracy of 90.3% while maintaining an acceptable frames per second (FPS) on a low-end GPU.",
            "",
            "",
            "120",
            "IQ is the ability to learn, understand, and apply information. EQ is the ability to understand and manage your own emotions and those of others. (question comparing different types of intelligence)",
        ],
        "PMID": [
            "",
            "",
            "29495095",
            "26933939",
            "36502047",
            "32936099",
            "29167088",
            "26598734",
            "22577301",
            "35751918",
            "31654584",
            "24897910",
            "27906516",
            "32024740",
            "",
            "",
            "30446408",
            "24943404",
            "23732046",
            "23957184",
            "29048766",
            "26330034",
            "24615759",
            "",
            "37749085",
            "34186105",
            "31766555",
            "36429493",
            "27047425",
            "22998852",
            "22998852",
            "34847057",
            "36502047",
            "31654584",
            "37299925",
            "36502178",
            "",
            "",
            "",
            "",
        ],
    }

    reference_df = pd.DataFrame(data)
    reference_df.insert(loc=reference_df.columns.get_loc("answer_human") + 1, column="answer_chatgpt", value="")
    reference_df.index += 1

    return reference_df






