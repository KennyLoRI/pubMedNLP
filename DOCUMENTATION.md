# Retrieval Augmented Generation based on PubMed Data

## Key Information

### Team Members
- [Kenneth Styppa](mailto:kenneth.styppa@web.de) (3735069, Scientific Computing)
- [Full Name 2](mailto:email2@example.com) (Matriculation Number, Course of Study)
- [Full Name 3](mailto:email3@example.com) (Matriculation Number, Course of Study)

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

Discuss prior work related to the project, emphasizing differences from previous work and the context of current research.

## Methods/Approach

```
Provide conceptual details of the system, including data processing pipelines, algorithms, and key methods. Be specific about methods, and include equations and figures where necessary. Clearly distinguish original contributions from existing methods.
```
### Project Orchestration
![Overview of the deployed pipelines](project_docs/pipelinesOverview.png)
For our technical orchestration, we employed Kedro, an open-source data pipeline framework to create reproducible, maintainable, and modular data science code ready for production [^1]. We organized our project into two distinct pipelines, facilitating the process of creating an RAG System end-to-end  in a structured manner.  The first pipeline contained all the nodes that laid the groundwork for the system. These ranged from data extraction to text chunking and preprocessing, as well as document embedding into the chroma vector store. The second pipeline uses the resulting vector store as a database for eventually creating the RAG system as a series of three nodes which encompass obtaining and cleaning user input, retrieving relevant documents, and using both of the former as input and context for text generation via a quantised llama2 model. Due to the complexity and the strong interdependence of these components, using Kedro was crucial to enforcing code quality and system stability. Furthermore, the ability to extract system settings such as prompt templates, retrieval strategies and model hyperparameters into the parameters.yml file streamlined our evaluation procedure by providing the possibility to perform grid-search over a predefined set of parameters for the full modelling pipeline. 
In the following paragraphs, we will highlight the noticeable aspects of each of these pipelines and nodes, providing an overview of the challenges faced and the solution built. 

### Data Processing Pipeline
#### Data Extraction
Data extraction was performed in the "extract_data_node" via the Entrez Programming Utilities (API). Provided by the National Center for Biotechnology Information (NCBI) Entrez offers relatively easy, although limited access to retrieve data from various NCBI databases, including PubMed [^2]. The procedure of obtaining the data mainly included two steps: 
1. Using Entrez's research function to retrieve article IDs that match the project requirements i.e. articles referring to the keyword "Intelligence" and published between 2013/01/01 and 2023/11/01.
2. Using Entrez's efetch function to retrieve detailed information for each of the retrieved IDs, and writing the retrieved results returned from the handle into a dataframe.

To avoid API limitations and stability issues for large-scoped retrieval, we conducted these two steps in a batch-wise manner. First, we used a sliding window of 30 days for which we retrieved the article IDs iterating over the whole timeframe on a month-by-month basis. Second, when having obtained the full ID list, we fetched the abstracts including their metadata for chunks of 500. 

#### Text Preprocessing and Chunking
#### Document Embedding
### Data Modelling Pipeline for Text Generation

## Experimental Setup and Results

### Data
Describe the dataset, including its source, collection method, and any insightful metrics.

### Evaluation Method
Define the evaluation metric used, whether existing or self-defined. Motivate the choice and clarify what it reflects.

### Experimental Details
Specify configurable parameters, explaining their choice and any optimization methods used.

### Results
Present results using tables and plots, comparing against baselines if available. Comment on results and analyze their alignment with expectations.

### Analysis
Include qualitative analysis. Discuss system performance in different contexts and compare with baselines.

## Conclusion, Limitations and Future Work

Recap main contributions, highlight achievements, and reflect on limitations. Suggest potential extensions or improvements for future work.

## References
[^1]: B. Deepa and K. Ramesh, "Production Level Data Pipeline Environment for Machine Learning Models," 2021 7th International Conference on Advanced Computing and Communication Systems (ICACCS), Coimbatore, India, 2021, pp. 404-407, doi: 10.1109/ICACCS51430.2021.9442035.

[^2]: Entrez Programming Utilities Help [Internet]. Bethesda (MD): National Center for Biotechnology Information (US); 2010-. Available from: https://www.ncbi.nlm.nih.gov/books/NBK25501/
```
Example:
[^1]: Smith, J., et al. (2020). *Title of the Paper*. Journal of Scientific Research, 15(3), 123-145. [DOI: 10.1234/jsr.2020.01234](http://dx.doi.org/10.1234/jsr.2020.01234)

[^2]: Johnson, A., & Brown, M. (2019). *Another Title*. Scientific Journal of Advanced Research, 8(2), 67-89. [DOI: 10.5678/sjar.2019.04567](http://dx.doi.org/10.5678/sjar.2019.04567)
```

```

Feel free to customize the template further based on your specific project details and preferences.
