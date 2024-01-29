# pubMedNLP
Transformer-based Question Answering System trained on PubMed data.
![Overview of the deployed pipelines](project_docs/pipelinesOverview.png)

## Overview

This project utilizes a combination of Kedro, Langchain, ChromaDB, and llama2.cpp to build a retrieval augmented generation system. The project is structured into a modular pipeline that can be run end-2-end to first obtain the data, preprocess and embed the data, and later perform queries to interact with the retrieved information similar to a Q&A chatbot. Due to the modularity, it is only a matter of a different command line prompt to use the latter, i.e. the readily developed Q&A system. 

For the current architecture sketch & assigned responsibilities see: https://miro.com/welcomeonboard/SmlWR0FzNXJjN0RoNzVEQzE4Z2FFQUUzSjU4cjFDanlYY3dVM0c0cGdkUkUxRkVzc2pFWUI4WjV3dGJNVmtSMnwzMDc0NDU3MzU0NTAzMjI4MzU4fDI=?share_link_id=219933613023


## Technologies Used

- **Kedro:** Kedro is a development workflow framework that facilitates the creation, visualization, and deployment of modular data pipelines.

- **Langchain:** Langchain is a framework for developing applications powered by language models.

- **ChromaDB:** Chroma DB is an open-source vector storage system (vector database) designed for storing and retrieving vector embeddings.

- **llama2.cpp:** llama2.cpp implements the Meta's LLaMa architecture in efficient C/C++.

## Installation & set-up

1. **Prerequisites:**
   - Ensure you have Python installed on your system. Your Python version should match 3.11.6
   - Ensure to have conda installed on your system.
   - Create a folder where you want to store the project. Call it e.g. pupMedNLP

   ```bash
   python --version
   conda --version
   # ls, cd etc. to get to your working directory
   ```

2. **Create a Conda Environment:**
   - Create a conda environment
   - Activate the environment
   ```bash
   conda create --name your_project_env python=3.11.6
   conda activate your_project_env
   ```

4. **Clone the Repository into your working directory:**
   ```bash
   git clone https://github.com/KennyLoRI/pubMedNLP.git
   cd your-project
   ```

5. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
6. **Download data and model files and place them into the right location:**
   - Go to Link and download the chromaDB store as well as the llama2.cpp model files. This can take a couple of minutes
   - Insert the chromaDB store at [your_folder]/kedronlp/[chroma_store] Remark: keep the title chroma_store
   - Insert the model file into kedronlp/data/06_models and keep the name.

## Usage
### Using the Q&A system
1. **Navigate to the kedronlp folder in your terminal:**
   ```bash
   cd [your_folder]/kedronlp
   ```

2. **Activate the Q&A System:**
   ```bash
   kedro run --pipeline=modelling
   ```
3. **Interact with the system:**
   - Ask your question
   ```bash
   Please enter your question (use *word* for abbreviations or special terms): [your_question]
   ```
   - The system will answer and terminate the session
   - Start with step 2 again and activate the system
   ```bash
   kedro run --pipeline=modelling
   ```
   - Ask another question (and so on)

### Optional usage possibilities 
1. **Visualize the pipeline:**
   - Use built-in features from Kedro to get an overview of the pipeline in your browser
   ```bash
   kedro viz
   ```
2. **Test the preprocessing pipeline:**
- Note: This is not advised since it may take a long time to extract the abstracts from PubMed and embed them (+ the PubMed API is not altogether stable):
  ```bash
  kedro run --pipeline=data_processing
  ```

## Project Structure

- **data/**: This directory contains the raw and processed data as well as the model files used by the project.
- **src/**: The source code of the project is organized into modules within this directory.
- **conf/**: Configuration files for Kedro and other tools are stored here. If you want to run the pipelines with different retrievers, or hyperparameters you can change them in the parameters.yml file and they will automatically be broadcasted to all necessary files.
- **scripts/**: Contains basic scripts we used during developing the project. They are not relevant to the pipeline.
- **notebooks/**: Contains tests and analysis notebooks. Have a look here to see our own evaluation of our system
- **docs/**: Documentation related to the project.

## Acknowledgments
We thank Prof. Gertz for this engaging course and Satya for her time to give us helpful advice. 
