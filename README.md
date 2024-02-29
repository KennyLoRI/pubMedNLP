# pubMedNLP
Transformer-based Question Answering System trained on PubMed data.
![Overview of the deployed pipelines](project_docs/read_me_graphic.png)

## Contributors: 
- [Kenneth Styppa](mailto:kenneth.styppa@web.de) (GitHub alias 'KennyLoRI' and 'Kenneth Styppa')
- [Daniel Bogacz](mailto:daniel.bogacz@stud.uni-heidelberg.de) (GitHub alias 'bgzdaniel')
- [Arjan Siddhpura](mailto:arjan.siddhpura@stud.uni-heidelberg.com) (GitHub alias 'arjansiddhpura')

## Overview

This project utilizes a combination of Kedro, Langchain, ChromaDB, and llama2.cpp to build a retrieval augmented generation system for medical question answering. The project is structured into modular pipelines that can be run end-2-end to first obtain the data, preprocess and embed the data, and later perform queries to interact with the retrieved information similar to a Q&A chatbot. Due to the modularity, it is only a matter of a different command line prompts to use the latter, i.e. the readily developed Q&A system. 

## Technologies Used

- **Kedro:** Kedro is a development workflow framework that facilitates the creation, visualization, and deployment of modular data pipelines.

- **Langchain:** Langchain is a framework for developing applications powered by language models, including information retrievers, text generation pipelines and other wrappers to facilitate a seamless integration of LLM-related open-source software.

- **ChromaDB:** Chroma DB is an open-source vector storage system designed for efficiently storing and retrieving vector embeddings.

- **llama2.cpp:** llama2.cpp implements Meta's LLaMa2 architecture in efficient C/C++ to enable a fast local runtime.

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
   conda create --name your_project_env python=3.10
   conda activate your_project_env
   ```

4. **Clone the Repository into your working directory:**
   ```bash
   git clone https://github.com/KennyLoRI/pubMedNLP.git
   ```
   When using Mac set pgk_config path:
   ```bash
   export PKG_CONFIG_PATH="/opt/homebrew/opt/openblas/lib/pkgconfig"
   ```

   then switch to the working directory of the project:
   ```bash
   cd pubMedNLP
   ```
   
6. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   
7. **Llama.cpp GPU installation:**
   (When using CPU only, skip this step.)

   This part might be slightly tricky, depending on which system the installation is done. We do NOT recommend installation on Windows. It has been tested, but requires multiple components which need to be downloaded. Please contact [Daniel Bogacz](mailto:daniel.bogacz@stud.uni-heidelberg.de) for details.

   **Linux:**
   ```bash
   CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir
   ```

   **MacOS:**
   ```bash
   CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir
   ```

   If anything goes wrong in this step, please contact [Daniel Bogacz](mailto:daniel.bogacz@stud.uni-heidelberg.de) for **Linux** installation issues and [Kenneth Styppa](mailto:kenneth.styppa@web.de) for **MacOS** installation issues. Also refer to the installation guide provided [here](https://python.langchain.com/docs/integrations/llms/llamacpp) and also [here](https://llama-cpp-python.readthedocs.io/en/latest/install/macos/)

6. **Download chroma store and model files and place them into the right location:**
   - Go to [this](https://drive.google.com/drive/folders/1-6FxGDDKGD-sMwT2Pax7VVMLzuZUH0DG) Google drive link and download the ChromaDB store (folder called `chroma_store_abstracts`) as well as the llama2.cpp [model files](https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF/blob/main/llama-2-7b-chat.Q5_K_M.gguf).
   - Insert the ChromaDB store at `[pubMedNLP]/kedronlp/`
   - Insert the model file into `[pubMedNLP]/kedronlp/data/06_models/` and keep the name

## Usage
### Using the Q&A system
1. **Navigate to the kedronlp folder in your terminal:**
   ```bash
   cd pubMedNLP/kedronlp
   ```

2. **Activate the Q&A System:**
   ```bash
   kedro run --pipeline=chat
   ```
3. **Interact with the system:**
   - Ask your question
   ```bash
   Please enter your question (use *word* for abbreviations or special terms): [your_question]
   ```
   - Ask another question (and so on)
  
Note: Running the system for the first time might take some additional seconds because the model has to be initialized. All questions, following the first one should be answered within a few seconds. If an answer takes more than 30 seconds to be completed, your GPU might not be automatically detected. You can check that by setting verbose = True in the parameters.yml file and then taking a look at the model initialization output. If it prints OPEN_BLAS = 1 somewhere, your GPU is automatically detected and it should be fine. If not please reach out to us in person via the emails provided in the documentation. 
  
### Trouble-shooting: 
- If you encounter an issue during your usage install pyspellchecker separately and try again:
  ```bash
  pip install pyspellchecker
  ```
- When encountering issues in the Llama.cpp installation, make sure you have NVIDIA Toolkit installed. Check with:
   ```bash
   nvcc --version
   ```
   Something similar to the following should appear:
   ```bash
   nvcc: NVIDIA (R) Cuda compiler driver
   Copyright (c) 2005-2023 NVIDIA Corporation
   Built on Wed_Feb__8_05:53:42_Coordinated_Universal_Time_2023
   Cuda compilation tools, release 12.1, V12.1.66
   Build cuda_12.1.r12.1/compiler.32415258_0
   ```
   Also make sure that CMake is installed on your system.

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
3. **Create paragraphs out of abstracts:**
   - For this, the file `extract_data.csv` is required. Place it in `kedronlp/data/01_raw`. See [here](https://drive.google.com/drive/folders/1-6FxGDDKGD-sMwT2Pax7VVMLzuZUH0DG) for the data. Go to `kedronlp/scripts`.
   ```bash
   python create_paragraphs.py
   ```
4. **Embedding of abstracts or paragraphs:**
   - For embedding abstracts the file `extract_data.csv` is required. Place it in `kedronlp/data/01_raw`. See [here](https://drive.google.com/drive/folders/1-6FxGDDKGD-sMwT2Pax7VVMLzuZUH0DG) for the data. Go to `kedronlp/scripts`.
   ```bash
   python abstract2vec.py
   ```
   - For embedding paragraphs the file `paragraphs.csv` is required. Place it in `kedronlp/data/01_raw`. See [here](https://drive.google.com/drive/folders/1-6FxGDDKGD-sMwT2Pax7VVMLzuZUH0DG) for the data. Go to `kedronlp/scripts`.
   ```bash
   python paragraph2vec.py
   ```
5. **Loading embeddings to the vector database ChromaDB:**
   - For loading abstract based embeddings to the vector database, the file `abstract_metadata_embeddings.csv` is required. Place it in `kedronlp/data/01_raw/`. See [here](https://drive.google.com/drive/folders/1-6FxGDDKGD-sMwT2Pax7VVMLzuZUH0DG) for the data. Go to `kedronlp/scripts`.
   ```bash
   python vec2chroma.py --granularity abstracts
   ```
   - For loading paragraph based embeddings to the vector database, the file `abstract_metadata_embeddings.csv` is required. Place it in `kedronlp/data/01_raw/`. See [here](https://drive.google.com/drive/folders/1-6FxGDDKGD-sMwT2Pax7VVMLzuZUH0DG) for the data. Go to `kedronlp/scripts`.
   ```bash
   python vec2chroma.py --granularity paragraphs
   ```
6. **Running Validation and Evaluation:**
   - For the validation and evaluation BleuRT is required. First clone bleuRT:
   ```bash
   git clone https://github.com/google-research/bleurt.git
   ```
   Go in into the subfolder 'bleurt':
   ```bash
   cd bleurt
   ```
   Specifically for *MacOS*: Because `tensorflow` is differently named under MacOS, the install requirements    have to be changed. Go to `bleurt/setup.py` and change in the list variable `install_requires` the entry `tensorflow` to `tensorflow-macos`. It should look like the following:
   ```python
   install_requires = [
    "pandas", "numpy", "scipy", "tensorflow-macos", "tf-slim>=1.1", "sentencepiece"
   ]
   ```

   Save the file.

   Install bleuRT with the following:
   ```bash
   pip install . 
   ```
   - Download the abstract based ChromaDB store (folder called `chroma_store_abstracts`) from [here](https://drive.google.com/drive/folders/1-6FxGDDKGD-sMwT2Pax7VVMLzuZUH0DG). The paragraph based vector database has do be created, it did not fit into the google drive link anymore. Please follow the steps above in '**Loading embeddings to the vector database ChromaDB**' for paragraph based embeddings. This should create the paragraph based ChromaDB store called `chroma_store_paragraphs`.
   Go to `kedronlp/scripts/evaluation`.
   ```bash
   python valid_and_eval.py
   ```

## Project Structure

- **data/**: This directory contains the raw and processed data as well as the model files used by the project.
- **src/**: The source code of the project is organized into modules within this directory.
- **conf/**: Configuration files for Kedro and other tools are stored here. If you want to run the pipelines with different retrievers, or hyperparameters you can change them in the parameters.yml file and they will automatically be broadcasted to all necessary files.
- **scripts/**: Contains scripts we used during developing the project.
- **scripts/evaluation**: Contains files and scripts to perform a validation and evaluation.
- **notebooks/**: Contains tests and analysis notebooks. Have a look here to see our own evaluation of our system
- **docs/**: Documentation related to the project.

## Acknowledgments
We thank Prof. Gertz for this engaging course and Satya for her time to give us helpful advice. 
