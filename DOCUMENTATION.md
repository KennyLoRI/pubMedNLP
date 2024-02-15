# Retrieval Augmented Generation based on PubMed Data

## Key Information

### Team Members
- [Kenneth Styppa](mailto:kenneth.styppa@web.de) (3735069, Scientific Computing)
- [Full Name 2](mailto:email2@example.com) (Matriculation Number, Course of Study)
- [Arjan Siddhpura](mailto:arjan.siddhpura@stud.uni-heidelberg.com) (3707267, Bachelor Informatik)

### Member Contribution

#### Kenneth Styppa
- Technical project orchestration via Kedro 
- Data Retrieval via Entrez API
- User input retrieval & preprocessing
- Filter Intention Extraction & NER on user input for improving document retrieval.
- Document Retrieval (Dense, BM25, Hybrid, MMR, Filter)
- Text Generation Pipeline
- Organizational project orchestration

#### Full Name 2
- Contribution Description 2
- Challenges Faced 2

#### Full Name 3
- Contribution Description 3
- Challenges Faced 3

### Advisor
- [Satya Almasian](mailto:satya.almasian@gmail.com)

### Anti-plagiarism Confirmation
We affirm that we did the project on our own.

## Introduction

Although being pretrained on a vast corpus of data, Large Language Models (LLMs) applied for Q&A applications face severe challenges when being confronted with out-of-scope user requests. Especially in highly sensitive domains, such as the medical domain, sharing data with companies that have the resources to train these models is oftentimes not an option. Retrieval augmented generation systems can lead out of this pitfall by providing a way, to use the power of LLMs on one's data without sharing it.
The purpose of this project was to create such a system based on 190k scientific abstracts from the Pubmed database, to test and evaluate the possibility of creating  a privacy-preserving, question-answering application that meets the high standards required in the medical domain.

## Related Work
### A short overview on Retrieval Augmented Generation

Discuss prior work related to the project, emphasizing differences from previous work and the context of current research.

## Methods/Approach

```
Provide conceptual details of the system, including data processing pipelines, algorithms, and key methods. Be specific about methods, and include equations and figures where necessary. Clearly distinguish original contributions from existing methods.
```
### Project Orchestration
![Overview of the deployed pipelines](project_docs/pipelinesOverview.png)
For our technical orchestration, we employed Kedro, an open-source data pipeline framework to create reproducible, maintainable, and modular data science code ready for production [^1]. We organized our project into two distinct pipelines, facilitating the process of creating an RAG System end-to-end  in a structured manner.  The first pipeline contained all the nodes that laid the groundwork for the system. These ranged from data extraction to text chunking and preprocessing, as well as document embedding into the chroma vector store. The second pipeline uses the resulting vector store as a database for eventually creating the RAG system as a series of three nodes which encompass obtaining and cleaning user input, retrieving relevant documents, and using both of the former as input and context for text generation via a quantised llama2 model. Due to the complexity and the strong interdependence of these components, using Kedro was crucial to enforcing code quality and system stability. Furthermore, the ability to extract system settings such as prompt templates, retrieval strategies and model hyperparameters into the `parameters.yml` file streamlined our evaluation procedure by providing the possibility to perform grid-search over a predefined set of parameters for the full modelling pipeline. 
In the following paragraphs, we will highlight the noticeable aspects of each of these pipelines and nodes, providing an overview of the challenges faced and the solution built. 

### Data Processing Pipeline
#### Data Extraction
Data extraction was performed in the "extract_data_node" via the Entrez Programming Utilities (API). Provided by the National Center for Biotechnology Information (NCBI) Entrez offers relatively easy, although limited access to retrieve data from various NCBI databases, including PubMed [^2]. The procedure of obtaining the data mainly included two steps: 
1. Using Entrez's research function to retrieve article IDs that match the project requirements i.e. articles referring to the keyword "Intelligence" and published between 2013/01/01 and 2023/11/01.
2. Using Entrez's efetch function to retrieve detailed information for each of the retrieved IDs, and writing the retrieved results returned from the handle into a dataframe.

To avoid API limitations and stability issues for large-scoped retrieval, we conducted these two steps in a batch-wise manner. First, we used a sliding window of 30 days for which we retrieved the article IDs iterating over the whole timeframe on a month-by-month basis. Second, when having obtained the full ID list, we fetched the abstracts including their metadata for chunks of 500. 

#### Text Preprocessing and Chunking
Having extracted the relevant abstracts, the next step is to split them into right-sized fragments to ensure the relevance of the retrieved results downstream in the retriever step of the system. Finding this optimal chunk size is a trade-off between context and specificity. It is generally acknowledged that neither too small nor too large chunk sizes are desirable since they "[...]may lead to sub-optimal outcomes" [^3]. The average size of our retrieved abstracts was moderate [TODO insert abstract data distribution]. Consequently, it was not evident whether embedding the full abstract strategy already suffers from the problems of large chunk sizes. To test this end-to-end, we employed two embedding strategies. One embeds the full abstracts while the other employs a semantic chunking approach that approximately leads to two chunks per abstract and embeds those. To eventually determine the optimal chunk size for our production systems we compared the full systems performance with either of these strategies. 

As part of our evaluation, we compared the system's performance once with the full abstract embeddings and once with the chunk embeddings. 
#### Document Embedding
### Data Modelling Pipeline for Text Generation
### Obtaining and postprocessing user input
The user's question was obtained via a simple command line input prompt. This input was forwarded into a spell-checking procedure correcting only spelling errors on a word-by-word basis.  Due to the special properties of the biomedical vocabulary that were likely to be encountered when creating a medical Q&A system, we additionally gave the user the ability to enclose special terms in asterisks to ensure it is not falsely corrected in the spellchecking procedure. To test the robustness of this task, we created a parameter for this process in our `parameters.yml` file, such that we were able to compare the performance of the full system with and without the spell-checker being used. After the input query has been corrected it is passed into a module that performs named entity recognition using "en_core_web_sm" from Spacy in conjunction with handcrafted linguistic rules to extract author names as well as time ranges indicated in the user query.

### Document retrieval 
For retrieving the relevant documents to a user's input query we first filtered out all documents that matched the filter statements extracted in the previous step, e.g. only publications with a publication year between 2020 and 2023. This narrowed down the database for the actual retrieval operation that followed subsequently. Our retriever node employed two main retrieving strategies that were each tested and compared in the evaluation phase of the project and can be easily switched on and off in the `parameters.yml file`. Independent of which strategy is chosen, the number of retrieved documents was controlled by the top_k parameter in the `parameters.yml` file.

The following paragraphs will briefly introduce each of the employed strategies. 
#### 1) Dense Retrieval Strategies
Dense retrieval with either pure cosine similarity or max marginal relevance, which is also based on cosine similarity but tries to enforce dissimilarity and therefore greater diversity upon the retrieved documents while retaining high similarity with the original query.

###### a) Cosine Similarity
Cosine similarity is a fundamental metric used in natural language processing to quantify the similarity between two vectors. In the context of our retriever, cosine similarity is employed to measure the angle between the query vector \(Q\) and the document vector \(D\), producing a numerical representation of their similarity. The formula for cosine similarity is given by:
$$\text{Cosine Similarity}(Q, D) = \frac{Q \cdot D}{\|Q\| \cdot \|D\|}$$
Where $Q \cdot D$ is the dot product of the query and document vectors, and $\|Q\|$ and $\|D\|$ are the Euclidean norms of the respective vectors.

###### b) Max Marginal Relevance (MMR)
Max Marginal Relevance (MMR) fosters diversity among the retrieved documents while maintaining high relevance to the query. Given a query Q, a list of documents R and a subset S of R that contains already selected documents, MMR as an incremental operation is defined as follows:

$$\text{MMR} = Arg max_{D_i \in R\setminus S} [\lambda (\text{Cosine Similarity}(D_i, Q) - (1 - \lambda) \cdot \max_{D_j \in S}(\text{Cosine Similarity}(D_i, D_j))]$$ 
I.e.,  $\lambda$ controls the trade-off between relevance and diversity. If $lambda = 1$ a cosine similarity output is obtained. For $lambda = 0$ maximum diversity is enforced [^4].

#### 2) Ensemble Retrieval Strategy
Our ensemble retriever combines dense retrieval (based on either cosine similarity or max marginal relevance) with BM25, a term-frequency-based retrieval operation specifically suited for providing exact term-based matches. It has been shown that Okapi BM25 can perform worse than some alternatives such as BM25L, when confronted with longer documents [^6]. Nevertheless, we chose this algorithm as the second retriever since with [TODO: INSERT NUMBER OF TOKENS] neither our abstracts and especially not the chunked paragraphs fall into the category of "very long documents" as Yuanhua et. al classified the problematic cases. The underlying idea of running both retrievers in parallel is to ensure that both exact term-based relevance, as well as context-based relevance, are captured and later combined via the reciprocal fusion rank. The BM25 score for a document D given a query Q consisting of $q_i$ terms with i = 1, ..., n is:   

$$\text{BM25}(Q, D) = \sum^n_{i \in Q} IDF(q_i) \cdot \frac{{tf(q_i,D)*(k_1+1)}}{{tf(q_i, D) + k_{1} \cdot (1 - b + b \cdot \frac{{\text{docLength}}}{{\text{avgDocLength}}})}}$$

Where:
- $tf(q_i,d)$ is the term frequency of term $\(i\)$ in the document d.
- $IDF(q_i) = ln(\frac{N-n(q_i)+0,5}{n(q_i) + 0.5}+1)$ is the inverse document frequency weight of the term $q_i$
- $N$ is the total number of documents in the collection and $n(q_i)$ the number of documents containing $q_i$.
- $k_{1}$ is a tuning parameter (k = 0: no use of term frequency, large k: raw term frequency.
- $b$ is a parameter controlling the impact of document length normalization.
- $\text{docLength}$ is the length of the document.
- $\text{avgDocLength}$ is the average document length in the corpus.

The Reciprocal Fusion Rank formula combines rankings from the dense retrieval strategy and BM25. Given a set D of documents to be ranked and a set of Rankings R (with cardinality 2 in our case), the RRF score is determined by

$$RRF(d \in D) = \sum_{r \in R}\frac{1}{k+r(d)}$$

Out of this eventually ranked set of retrieved documents the `top_k` are extracted.

Since  medical terminology can pose various challenges, no retrieval strategy exhibits a clear conceptual superiority apriori, although intuition suggested that a combination of context and specificity is possibly best suited to the wide-ranging and oftentimes critical queries in the medical domain [^7]. Through deploying, testing and comparing these different retrieving strategies, we thus aimed to identify the retriever which meets the requirements the best using a diverse manually annotated validation set.

### Obtaining LLM output
#### LLM models used

#### Prompting strategies
Once the `top_k` documents are extracted, they are formatted together with the initial (possibly corrected) query into a prompt that is inputted into our model. Due to model size limitations that became evident in the first runs of the pipeline, short and precise prompts were selected. Except for this restriction upfront, all other decisions were based on end-to-end testing of different prompt variants. These variants were: 
1. A standard prompt
2. Chain of thought
3. Inputting abstract information only (controlled in the `abstract_only` parameter:
4. Inputting abstracts enriched with metadata information:


## Experimental setup and results


### Data
Describe the dataset, including its source, collection method, and any insightful metrics.

### Evaluation Method
To reproducibly determine the best possible system with the given components, we ran a grid-search script that tests [TODO: INSERT NUMBER OF COMBINATIONS] combinations end-to-end, computing their performance measured by the [TODO: INSERT USED METRICS] scores on our validation set consisting of [TODO: INSERT NUMBER OF QUESTIONS IN THE VALIDATION SET]. 

TODO: Define the evaluation metric used, whether existing or self-defined. Motivate the choice and clarify what it reflects.

### Experimental Details
Specify configurable parameters, explaining their choice and any optimization methods used.

### Results
Present results using tables and plots, comparing against baselines if available. Comment on results and analyze their alignment with expectations.

### Analysis
Include qualitative analysis. Discuss system performance in different contexts and compare with baselines.

## Conclusion
### Limitations
The most severe and yet unavoidable limitation of our proposed system is that of the used model. Since during development no computational resources were available to us save for our laptops, we needed to use a quantized llama2.cpp model to ensure that developing and testing the system was possible in a reasonable timescope. While the model performed adequately in text generation, its limitations made certain operations that demanded a more powerful model impossible. This was especially noticeable when implementing the model as part of a self-querying logic provided by langchain to extract metadata filters out of the provided user input query (e.g. author names, paper titles etc.). The most severe of the encountered problems was the model inventing filters where none were indicated. Since this hallucinating behaviour would have caused severe issues downstream due to false filter operations previous to document retrieval, we decided to implement a less powerful, although more stable solution. This solution employs Spacy's "en_core_web_sm" module and manually crafted linguistic rules to detect publishing dates and author names. While possibly missing some filter intentions as a result of being trained on typical English texts such that e.g. Chinese author names are missed, this solution does not invent filters where none are indicated and is thus the significantly better choice. And yet, equipped with a more powerful model aside from improved text generation performance, such and similar operations could have had a possibly large impact in improving our system.

Furthermore, during development, some of our initial ideas were not implementable since the underlying code functionality could not be guaranteed due to errors in the source code of Langchain (s. [issue](https://github.com/langchain-ai/langchain/issues/15959) ). This specifically accounts for our idea to implement and test a [multi-query retrieval strategy](https://python.langchain.com/docs/modules/data_connection/retrievers/MultiQueryRetriever), which uses an LLM to generate queries based on the user input, retrieves documents for each query and then outputs the intersection set of all retrieved documents as the final retrieved set. However, given the already-discussed model limitation, it is likely that even if the issue had been fixed, no improvements would have been gained by deploying the strategy since the generated questions likely would not have been superior to the original user query. 


### Future Work

Recap main contributions, highlight achievements, and reflect on limitations. Suggest potential extensions or improvements for future work.

## References
[^1]: B. Deepa and K. Ramesh, "Production Level Data Pipeline Environment for Machine Learning Models," 2021 7th International Conference on Advanced Computing and Communication Systems (ICACCS), Coimbatore, India, 2021, pp. 404-407, doi: 10.1109/ICACCS51430.2021.9442035.

[^2]: Entrez Programming Utilities Help [Internet]. Bethesda (MD): National Center for Biotechnology Information (US); 2010-. Available from: https://www.ncbi.nlm.nih.gov/books/NBK25501/

[^3]: Gao, Yunfan, et al. "Retrieval-augmented generation for large language models: A survey." arXiv preprint arXiv:2312.10997 (2023).

[^4]: Carbonell, Jaime, and Jade Goldstein. "The use of MMR, diversity-based reranking for reordering documents and producing summaries." Proceedings of the 21st annual international ACM SIGIR conference on Research and development in information retrieval. 1998.

[^5]: Brown, Dorian. "Rank_bm25". Retrieved from: https://github.com/dorianbrown/rank_bm25. Date: 15.02.2024

[^6]: Lv, Yuanhua, and ChengXiang Zhai. "When documents are very long, bm25 fails!." Proceedings of the 34th international ACM SIGIR conference on Research and development in Information Retrieval. 2011.

[^7]: Agrawal, Shweta, and Sanjiv Kumar Jain. "Medical text and image processing: applications, issues and challenges." Machine Learning with Health Care Perspective: Machine Learning and Healthcare (2020): 237-262.
