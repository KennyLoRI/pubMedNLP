import regex as re
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import pandas as pd
from dateutil import parser as date_parser
from datetime import datetime

# Extract the abstract from each string
def extract_abstract(context, question):
  pattern = re.compile(r"Abstract: (.+?)(?=\n)")
  abstracts = [re.search(pattern, string).group(1) for string in context]
  input_context = ''.join(abstracts)
  input_dict =  {"context": input_context, "question": question}
  return input_dict

#function to print context information
def get_context_details(context, print_context = True, as_input_dict = False, user_input = None, abstract_only = False):
  abstract_pattern = re.compile(r"Abstract: (.+?)(?=\n)")
  authors_pattern = re.compile(r'Authors: (.+?)\n')
  title_pattern = re.compile(r'Title: (.+?)\n')
  year_pattern = re.compile(r'Year: (\d{4})\n')

  print(f"\n\n{'='*20}\nSources:")
  abstracts_list = []
  authors_list = []
  title_list = []
  year_list = []
  for data_string in context:
      abstracts_match = abstract_pattern.search(data_string)
      authors_match = authors_pattern.search(data_string)
      title_match = title_pattern.search(data_string)
      year_match = year_pattern.search(data_string)

      abstracts = abstracts_match.group(1) if abstracts_match else 'NA'
      authors = authors_match.group(1) if authors_match else 'NA'
      title = title_match.group(1) if title_match else 'NA'
      year = year_match.group(1) if year_match else 'NA'
      abstracts_list.append(abstracts)
      authors_list.append(authors)
      title_list.append(title)
      year_list.append(year)
      if print_context:
        print(f"\nAuthors: {authors}\nTitle: {title}\nYear: {year}\n{'_'*20}")

  if as_input_dict == False:
      context = {
          'Author': authors_list,
          'Title': title_list,
          'Year': year_list
      }

  else:
      # Clean input lists for well formatted input
      if abstract_only:
        filtered_abstr_lst = [value for value in abstracts_list if value != "NA"]
        structured_context = ' '.join(filtered_abstr_lst)
        context = {"context": structured_context, "question": user_input}
      else:
        context_strings = []
        for i, (pub_year, title, abstract) in enumerate(zip(year_list, title_list, abstracts_list)):
            paper_info = f"Paper {i}:\nTitle: {title}\nPublication Year: {pub_year}\nAbstract: {abstract}\n\n"
            context_strings.append(paper_info)
        structured_context = ' '.join(context_strings)
        context = {"context": structured_context, "question": user_input}

  return context


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

def parse_date_with_hints(year, hint):
    if hint.lower() == "before":
        return year - 1
    elif hint.lower() == "after":
        return year + 1
    elif hint.lower() == "since":
        return year
    elif hint.lower() == "later than":
        return year +1
    elif hint.lower() == "no later than":
        return year
    elif hint.lower == "previous to":
        return year -1
    else:
        return year

def extract_date_range(doc):
    start_date = None
    end_date = None

    # Parce dates using linguistic hints
    for token in doc:
        if token.ent_type_ == "DATE" and token.is_digit:
            date_str = token.text
            try:
                year = date_parser.parse(date_str).year
                #search for linguistic hints
                prev_token = doc[token.i-1]
                prev_prev_token = doc[prev_token.i-1]
                if prev_token.text.lower() in ["before", "prior", "until"]:
                    end_date = parse_date_with_hints(year, prev_token.text)
                elif prev_prev_token.text.lower() in ["previous", "prior"] and prev_token.text.lower()=="to":
                    end_date = parse_date_with_hints(year, "previous to")
                elif prev_token.text.lower() in ["after", "since"]:
                    start_date = parse_date_with_hints(year, prev_token.text)
                elif prev_prev_token.text.lower() == "later" and prev_token.text.lower() == "than":
                    if doc[prev_prev_token.i - 1].text == "no":
                        end_date = parse_date_with_hints(year, "no later than")
                    else:
                        start_date = parse_date_with_hints(year, "later than")
                elif prev_token.text.lower() == "between" and doc[token.i + 1].text.lower() == "and" and doc[token.i + 2].is_digit:
                    start_date = year
                    end_date = date_parser.parse(doc[token.i + 2].text).year
                elif prev_token.text.lower() in ["within", "during", "in", "throughout"]:
                    start_date = year
                    end_date = year
            except ValueError:
                pass #perform no filter if structure gets more complicated


    return [start_date, end_date]