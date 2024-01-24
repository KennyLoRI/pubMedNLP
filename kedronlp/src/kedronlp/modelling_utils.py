import regex as re
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import pandas as pd
from dateutil import parser

# Extract the abstract from each string
def extract_abstract(context, question):
  pattern = re.compile(r"Abstract: (.+?)(?=\n)")
  abstracts = [re.search(pattern, string).group(1) for string in context]
  input_context = ''.join(abstracts)
  input_dict =  {"context": input_context, "question": question}
  return input_dict

#function to print context information
def print_context_details(context, print_context = True):
  authors_pattern = re.compile(r'Authors: (.+?)\n')
  title_pattern = re.compile(r'Title: (.+?)\n')
  year_pattern = re.compile(r'Year: (\d{4})\n')

  print(f"\n\n{'='*20}\nSources:")
  authors_list = []
  title_list = []
  year_list = []
  for data_string in context:
      authors_match = authors_pattern.search(data_string)
      title_match = title_pattern.search(data_string)
      year_match = year_pattern.search(data_string)

      authors = authors_match.group(1) if authors_match else 'N/A'
      title = title_match.group(1) if title_match else 'N/A'
      year = year_match.group(1) if year_match else 'N/A'
      authors_list.append(authors)
      title_list.append(title)
      year_list.append(year)
      if print_context:
        print(f"\nAuthors: {authors}\nTitle: {title}\nYear: {year}\n{'_'*20}")


  context_dict = {
      'Author': authors_list,
      'Title': title_list,
      'Year': year_list
  }

  return context_dict


def instantiate_llm(temperature = 0,
                    max_tokens = 1000,
                    n_ctx = 2048,
                    top_p = 1,
                    n_gpu_layers = 40,
                    n_batch = 512,
                    verbose = True,
                    path='data/06_models/llama-2-7b-chat.Q5_K_M.gguf'):
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llm = LlamaCpp(
        model_path=path,
        temperature=temperature,
        max_tokens=max_tokens,
        n_ctx=n_ctx,
        top_p=top_p,
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        callback_manager=callback_manager,
        verbose=verbose,  # Verbose is required to pass to the callback manager
        )
    return llm

def is_within_range(date_str, after_years, before_years):
    date_obj = parser.parse(date_str, fuzzy=True)
    if after_years and date_obj.year <= max(after_years):
        return False
    if before_years and date_obj.year >= min(before_years):
        return False
    return True