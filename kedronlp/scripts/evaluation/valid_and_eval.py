from sentence_transformers import util
from eval_utils import get_predictions
from get_retriever import get_retriever
from modelling_utils import instantiate_llm
from embedding_utils import PubMedBert
from scorer import Scorer
import pandas as pd
import random
import numpy as np
import math
import torch
from tqdm import tqdm
from copy import deepcopy

prompt_template = """
You are a biomedical AI assistant to answer medical questions
mostly about PubMed articles provided as context for you.
As an AI assistant, answer the question accurately,
precisely and concisely. 
Not every article in the provided context is necessarily relevant to the question.
Carefully examine the provided information in the articles and choose the
most likely correct information to answer the question.
If the question is not from the biomedical domain, tell the user that
the question is out of domain and cannot be answered by you.
Only include information in your answer, which is necessary to answer the question.
Be as short as possible with your answer.

Use the following articles to determine the answer: {context}
The question: {question}
Your answer:
"""

temperatures = [0, 0.5]
retrieval_strategies = ["ensemble_retrieval", "similarity", "max_marginal_relevance"]
advanced_dense_retrievers = ["similarity", "mmr"]
granularities = ["paragraphs", "abstracts"]
top_ks = [2, 3]
metadata_strategies = ["parser", "none"]
abstract_only_options = [True, False]

combinations = []
for retrieval_strategy in retrieval_strategies:
    for temperature in temperatures:
        for abstract_only in abstract_only_options:
            for metadata_strategy in metadata_strategies:
                for granularity in granularities:
                    for top_k in top_ks:
                        if granularity == "paragraphs":
                            top_k *= 2
                        combination = {
                                "temperature": temperature,
                                "abstract_only": abstract_only,
                                "metadata_strategy": metadata_strategy,
                                "granularity": granularity,
                                "top_k": top_k,
                                "retrieval_strategy": retrieval_strategy,
                                "advanced_dense_retriever": "none",
                            }
                        if retrieval_strategy != "ensemble_retrieval":
                            combinations.append(combination)
                        elif retrieval_strategy == "ensemble_retrieval":
                            for advanced_dense_retriever in advanced_dense_retrievers:
                                combination_with_advanced = deepcopy(combination)
                                combination_with_advanced["advanced_dense_retriever"] = advanced_dense_retriever
                                combinations.append(combination_with_advanced)

handmade_df = pd.read_csv("eval_questions_handmade.csv")
handmade_columns = list(handmade_df.columns)
semigenerated_df = pd.read_csv("eval_questions_semi_generated.csv")
semigenerated_df = semigenerated_df[handmade_columns]
df = pd.concat([handmade_df, semigenerated_df], ignore_index=True).fillna("")

types = df["Question Type"].unique()
print(f"Available Question Types:\n{types}")
test_set = {}
for a_type in types:
    questions_answers = df.loc[df["Question Type"] == a_type][
        ["Question", "Answer", "Source"]
    ]
    test_set[a_type] = questions_answers.to_dict(orient="records")

breakpoint()
# calculate split for validation set, equals about 40% of all data
splits = [math.ceil(len(test_set[a_type])/2.5) for a_type in test_set.keys()]

validation_set = []
for a_type, split in zip(test_set.keys(), splits):
    if len(test_set[a_type]) > 1:
        for i in range(split):
            qas = test_set[a_type].pop(random.randint(0, len(test_set[a_type]) - 1))
            validation_set.append(qas)

# # for testing
# validation_set = random.sample(validation_set, 10)
# test_set = {a_type: [qa_list[0]] for a_type, qa_list in test_set.items()}

print(f"validation set length: {len(validation_set)}")
print(f"test set length: {sum([len(qas_list) for _, qas_list in test_set.items()])}")

scorer = Scorer()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu") # because cuda out of memory very likely, vectordb and PubMedBert in RAM, LLM is still on GPU

# usage of "last_*" parameters for performing reinitialization only if necessary for efficiency
last_temperature = -1
last_top_k = -1
last_granularity = ""
last_strategy = ""

# validation, find best set of parameters
combination_scores = []
for combination in tqdm(combinations):

    combination["mq_include_original"] = False
    combination["prompt_template"] = prompt_template
    combination["max_tokens"] = 1000
    combination["n_ctx"] = 2048
    combination["top_p"] = 1
    combination["n_gpu_layers"] = -1
    combination["n_batch"] = 512
    combination["verbose"] = True
    combination["spell_checker"] = False

    # temperature change --> reinitiate llm
    if combination["temperature"] != last_temperature:
        llm = None
        llm = instantiate_llm(
            combination["temperature"],
            combination["max_tokens"],
            combination["n_ctx"],
            combination["top_p"],
            combination["n_gpu_layers"],
            combination["n_batch"],
            combination["verbose"],
        )
        last_temperature = combination["temperature"]

    # retrievel strategy change, top_k change or granularity change --> reinitiate retriever
    if (combination["retrieval_strategy"] == "ensemble_retrieval"\
        and (last_strategy != "ensemble_retrieval"\
        or last_top_k != combination["top_k"]))\
        or last_granularity != combination["granularity"]:
        retriever = None
        retriever = get_retriever(combination, device)
        last_top_k = combination["top_k"]
        last_granularity = combination["granularity"]
        last_strategy = combination["retrieval_strategy"]

    print()
    for param, value in combination.items():
        print(f"{param}: {value}", flush=True)
    print()

    questions = [str(qas["Question"]) for qas in validation_set]
    references = [str(qas["Answer"]) for qas in validation_set]
    predictions, _ = get_predictions(llm, questions, combination, combination, retriever)
    scores = scorer.get_scores(predictions=predictions, references=references)
    combination_scores.append(
        {
            "combination": combination,
            "scores": scores,
        }
    )


def weighted_score(scores):
    weighted_score = (
        scores["bleuscore"] * 0.1
        + scores["rougescore"] * 0.1
        + scores["bertscore"] * 0.4
        + scores["bleurtscore"] * 0.4
    )
    return weighted_score


ranked_combinations = sorted(combination_scores, key=lambda x: weighted_score(x["scores"]), reverse=True)

# output evaluation results to files
with open("ranked_combinations.txt", "w") as file:
    relevant_combination_keys = [
        "temperature",
        "abstract_only",
        "metadata_strategy",
        "granularity",
        "top_k",
        "retrieval_strategy",
        "advanced_dense_retriever",
    ]
    for i, combination in enumerate(ranked_combinations):
        file.write(f"combination {i}:\n")
        for key in relevant_combination_keys:
            file.write(f"\t{key}: {combination['combination'][key]}\n")
        file.write(f"scores {i}:\n")
        for score_type, score in combination["scores"].items():
            file.write(f"\t{score_type}: {score:.4f}\n")
        file.write(f"\toverall weighted score: {weighted_score(combination['scores']):.4f}\n")


best_combination = ranked_combinations[0]["combination"]

llm = None
llm = instantiate_llm(
    best_combination["temperature"],
    best_combination["max_tokens"],
    best_combination["n_ctx"],
    best_combination["top_p"],
    best_combination["n_gpu_layers"],
    best_combination["n_batch"],
    best_combination["verbose"],
)
retriever = None
retriever = get_retriever(best_combination, device)

# evaluation end2end
# determine scores for all different question types and overall score
question_types_scores = {}
for question_type, qas_list in test_set.items():
    questions = [str(qas["Question"]) for qas in qas_list]
    references = [str(qas["Answer"]) for qas in qas_list]
    sources = [str(qas["Source"]) for qas in qas_list]
    predictions, _ = get_predictions(llm, questions, best_combination, best_combination, retriever)
    question_types_scores[question_type] = scorer.get_scores(
        predictions=predictions, references=references
    )

ranked_question_types_scores = dict(sorted(question_types_scores.items(), key=lambda x: weighted_score(x[1]), reverse=True))


overall_bleuscore = np.mean([scores["bleuscore"] for _, scores in ranked_question_types_scores.items()])
overall_rougescore = np.mean([scores["rougescore"] for _, scores in ranked_question_types_scores.items()])
overall_bertscore = np.mean([scores["bertscore"] for _, scores in ranked_question_types_scores.items()])
overall_bleurtscore = np.mean([scores["bleurtscore"] for _, scores in ranked_question_types_scores.items()])
overall_weighted_score = weighted_score(
    {
        "bleuscore": overall_bleuscore,
        "rougescore": overall_rougescore,
        "bertscore": overall_bertscore,
        "bleurtscore": overall_bleurtscore,
    })

with open("ranked_question_types_scores.txt", "w") as file:
    for question_type, scores in ranked_question_types_scores.items():
        file.write(f"{question_type}:\n")
        for score_type, score in scores.items():
            file.write(f"\t{score_type}: {score:.4f}\n")
        file.write(f"weighted score: {weighted_score(scores):.4f}")
        file.write("\n\n")
    file.write(f"overall results:\n")
    file.write(f"\tbleuscore: {overall_bleuscore:.4f}\n")
    file.write(f"\trougescore: {overall_rougescore:.4f}\n")
    file.write(f"\tbertscore: {overall_bertscore:.4f}\n")
    file.write(f"\tbleurtscore: {overall_bleurtscore:.4f}\n")
    file.write(f"\tweighted score: {overall_weighted_score:.4f}\n")


# evaluation retriever
# * each qa pair only has one source, so recall can only be 0 or 1 for each pair
# * calculate embedding distance between retrieved source and gold source, keep shortest distance
all_qas = []
for question_type, qas_list in test_set.items():
    all_qas.extend(qas_list)
questions = [str(qas["Question"]) for qas in all_qas if str(qas["Source"]) != "nan"]
references = [str(qas["Answer"]) for qas in all_qas if str(qas["Source"]) != "nan"]
sources = [str(qas["Source"]) for qas in all_qas if str(qas["Source"]) != "nan"]

model = PubMedBert(device=device)
retrievers_results = []
for retrieval_strategy in retrieval_strategies:
    best_combination["retrieval_strategy"] = retrieval_strategy
    for advanced_dense_retriever in advanced_dense_retrievers:
        best_combination["advanced_dense_retriever"] = advanced_dense_retriever
        if retrieval_strategy != "ensemble_retrieval":
            break_after = True
            best_combination["advanced_dense_retriever"] = "none"
        retriever = None
        retriever = get_retriever(best_combination, device)
        _, contexts = get_predictions(llm, questions, best_combination, best_combination, retriever)

        recalls = []
        min_distances = []
        for retrieved_sources, gold_source in zip(contexts, sources):
            in_gold = 0
            for source in retrieved_sources:
                n = 20
                if len(source) <= n:
                    n = len(source)-1
                if source[:n] in gold_source: # use first n characters, should be unique enough
                    in_gold = 1
                    break
            recalls.append(in_gold)

            retrieved_sources_embeddings = model.encode(retrieved_sources)
            gold_source_embedding = model.encode(gold_source)
            distances = util.cos_sim(retrieved_sources_embeddings, gold_source_embedding)
            min_distances.append(torch.min(distances).item())

        recall = np.around(np.mean(recalls), 4)
        min_distance = np.around(np.mean(min_distances), 4)

        retrievers_results.append({
            "retriever_strategy": retrieval_strategy,
            "advanced_dense_retriever": advanced_dense_retriever,
            "avg_recall": recall,
            "avg_cos_distance": min_distance,
        })
        if break_after:
            break

ranked_retrievers_recalls = sorted(retrievers_results, key=lambda x: x["avg_recall"], reverse=True)
ranked_retrievers_distance = sorted(retrievers_results, key=lambda x: x["avg_cos_distance"], reverse=True)

with open("ranked_retrievers_recalls.txt", "w") as file:
    for i, retriever_combination in enumerate(ranked_retrievers_recalls):
        file.write(f"retriever_combination {i}:\n")
        for key, value in retriever_combination.items():
            file.write(f"\t{key}: {value}\n")

with open("ranked_retrievers_distance.txt", "w") as file:
    for i, retriever_combination in enumerate(ranked_retrievers_distance):
        file.write(f"retriever_combination {i}:\n")
        for key, value in retriever_combination.items():
            file.write(f"\t{key}: {value}\n")

print("#"*10)
print("done! Ignore error below! This is due to some deinitialization errors of bleuRT.", flush=True)
print("#"*10)