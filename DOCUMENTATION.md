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
- Filter Intention Extraction & NER on user input
- Document Retrieval (Dense, BM25, Ensemble, MMR)
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
For our technical orchestration, we employed Kedro, an open-source data pipeline framework to create reproducible, maintainable, and modular data science code ready for production. We organized our project into two distinct pipelines, facilitating the process of creating an RAG System end-to-end  in a structured manner.  The first pipeline contained all the nodes that laid the groundwork for the system. These ranged from data extraction to text chunking and preprocessing, as well as document embedding into the chroma vector store. The second pipeline uses the resulting vector store as a database for eventually creating the RAG system as a series of three nodes which encompass obtaining and cleaning user input, retrieving relevant documents, and using both of the former as input and context for text generation via a quantised llama2 model. Due to the complexity and the strong interdependence of these components, using Kedro was crucial to enforcing code quality and system stability. Furthermore, the ability to extract system settings such as used prompt templates, retrieval strategies and model hyperparameters into the parameters.yml file streamlined our evaluation procedure by providing the possibility to perform grid-search over a predefined set of parameters for the full modelling pipeline. 
In the following paragraphs, we will highlight the noticeable aspects of each of these pipelines and nodes, providing an overview of the challenges faced and the solution built. 

### Data Processing Pipeline
#### Data Extraction
Data extraction was performed via the Entrez Programming Utilities (API) which is provided by the National Center for Biotechnology Information (NCBI) for programmatically accessing and retrieving data from various NCBI databases, including PubMed. 
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

## Conclusion and Future Work

Recap main contributions, highlight achievements, and reflect on limitations. Suggest potential extensions or improvements for future work.

## References

- [Author1 et al. 2020](#link-to-the-bib-section)
- [Author2 et al. 2021](#link-to-the-bib-section)
```

Feel free to customize the template further based on your specific project details and preferences.
