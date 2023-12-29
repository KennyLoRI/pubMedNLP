import regex as re
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Extract the abstract from each string
def extract_abstract(context, question):
  pattern = re.compile(r"Abstract: (.+?)(?=\n)")
  abstracts = [re.search(pattern, string).group(1) for string in context]
  input_context = ''.join(abstracts)
  input_dict =  {"context": input_context, "question": question}
  return input_dict

#function to print context information
def print_context_details(context):
  authors_pattern = re.compile(r'Authors: (.+?)\n')
  title_pattern = re.compile(r'Title: (.+?)\n')
  year_pattern = re.compile(r'Year: (\d{4})\n')

  print(f"\n\n{'='*20}\nSources:")
  for data_string in context:
      authors_match = authors_pattern.search(data_string)
      title_match = title_pattern.search(data_string)
      year_match = year_pattern.search(data_string)

      authors = authors_match.group(1) if authors_match else 'N/A'
      title = title_match.group(1) if title_match else 'N/A'
      year = year_match.group(1) if year_match else 'N/A'
      print(f"\nAuthors: {authors}\nTitle: {title}\nYear: {year}\n{'_'*20}")

def instantiate_llm(path = "/Users/Kenneth/PycharmProjects/pubMedNLP/kedronlp/data/06_models/llama-2-7b-chat.Q4_K_M.gguf"):
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llm = LlamaCpp(
        model_path=path,
        temperature=0,
        max_tokens=1000,
        n_ctx=2048,
        top_p=1,
        n_gpu_layers=40,
        n_batch=512,
        callback_manager=callback_manager,
        verbose=True,  # Verbose is required to pass to the callback manager
        )
    return llm