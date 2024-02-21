from eval_utils import get_predictions
from modelling_utils import instantiate_llm
from scorer import Scorer
import pandas as pd
import random
import numpy as np
import math

prompt_template = """
You are a biomedical AI assistant to answer medical questions 
mostly about PubMed articles based on provided context. 
If the question is not from the biomedical domain, tell the user that 
the question is out of domain and can't be answered by you. 
As an AI assistant, answer the question accurately, 
precisely and concisely. Only include information in your answer, 
which is necessary to answer the question. Be as short and concise as possible. 
Do NOT mention that your answer is based on the provided paper or context. 
Use the following context if applicable: {context} 
The question: {question} 
Your answer: 
"""

temperatures = [0, 0.5]
retrieval_strategies = ["similarity", "max_marginal_relevance"] #"ensemble_retrieval"]
advanced_dense_retrievers = ["similarity", "mmr"]
granularities = ["paragraphs", "abstracts"]
top_ks = [2, 3]
metadata_strategies = ["parser", "none"]
spell_checker_options = [True, False]
abstract_only_options = [True, False]

combinations = []
for temperature in temperatures:
    for spell_checker in spell_checker_options:
        for abstract_only in abstract_only_options:
            for metadata_strategy in metadata_strategies:
                for granularity in granularities:
                    for top_k in top_ks:
                        if granularity == "paragraphs":
                            top_k *= 2
                        for retrieval_strategy in retrieval_strategies:
                            for advanced_dense_retriever in advanced_dense_retrievers:
                                combinations.append(
                                    {
                                        "temperature": temperature,
                                        "spell_checker": spell_checker,
                                        "abstract_only": abstract_only,
                                        "metadata_strategy": metadata_strategy,
                                        "granularity": granularity,
                                        "top_k": top_k,
                                        "retrieval_strategy": retrieval_strategy,
                                        "advanced_dense_retriever": advanced_dense_retriever,
                                    }
                                )
                                if retrieval_strategy != "ensemble_retrieval":
                                    combinations[-1]["advanced_dense_retriever"] = "none"
                                    break

df = pd.read_csv("Evaluation.csv")
types = df["Question Type"].unique()
test_set = {}
for a_type in types:
    questions_answers = df.loc[df["Question Type"] == a_type][
        ["Question", "Answer", "Source"]
    ]
    test_set[a_type] = questions_answers.to_dict(orient="records")

# calculate split for validation set, equals about 40% of all data
splits = [math.ceil(len(test_set[a_type])/4) for a_type in test_set.keys()]

validation_set = []
for a_type, split in zip(test_set.keys(), splits):
    if len(test_set[a_type]) > 1:
        for i in range(split):
            qas = test_set[a_type].pop(random.randint(0, len(test_set[a_type]) - 1))
            validation_set.append(qas)

# for testing
validation_set = random.sample(validation_set, 3)
test_set = {a_type: [qa_list[0]] for a_type, qa_list in test_set.items()}

print(f"validation set length: {len(validation_set)}")
print(f"test set length: {sum([len(qas_list) for _, qas_list in test_set.items()])}")

scorer = Scorer()

# validation, find best set of parameters
combination_scores = []
last_temperature = -1
for combination in combinations:

    combination["mq_include_original"] = False
    combination["prompt_template"] = prompt_template
    combination["max_tokens"] = 1000
    combination["n_ctx"] = 2048
    combination["top_p"] = 1
    combination["n_gpu_layers"] = -1
    combination["n_batch"] = 512
    combination["verbose"] = True

    # temperature change --> reinitiate llm
    if combination["temperature"] != last_temperature:
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

    for param, value in combination.items():
        print(f"{param}: {value}")
    print()

    questions = [qas["Question"] for qas in validation_set]
    references = [qas["Answer"] for qas in validation_set]
    predictions, _ = get_predictions(llm, questions, combination, combination)
    scores = scorer.get_scores(predictions=predictions, references=references)
    combination_scores.append(
        {
            "combination": combination,
            "scores": scores,
        }
    )


def weighted_score(combination_score):
    scores = combination_score["scores"]
    weighted_score = (
        scores["bleuscore"] * 0.15
        + scores["roguescore"] * 0.15
        + scores["bertscore"] * 0.35
        + scores["bleurtscore"] * 0.35
    )
    return weighted_score


ranked_combinations = sorted(combination_scores, key=weighted_score, reverse=True)
best_combination = ranked_combinations[0]["combination"]
llm = instantiate_llm(
    best_combination["temperature"],
    best_combination["max_tokens"],
    best_combination["n_ctx"],
    best_combination["top_p"],
    best_combination["n_gpu_layers"],
    best_combination["n_batch"],
    best_combination["verbose"],
)

# evaluation end2end
# determine scores for all different question types and overall score
question_types_scores = {}
for question_type, qas_list in test_set.items():
    questions = [qas["Question"] for qas in qas_list]
    references = [qas["Answer"] for qas in qas_list]
    sources = [qas["Source"] for qas in qas_list]
    predictions, _ = get_predictions(llm, questions, best_combination, best_combination)
    question_types_scores[question_type] = scorer.get_scores(
        predictions=predictions, references=references
    )


# evaluation retriever
# each qa pair only has one source, so recall can only be 0 or 1 for each pair
all_qas = []
for question_type, qas_list in test_set.items():
    all_qas.extend(qas_list)
questions = [qas["Question"] for qas in all_qas]
references = [qas["Answer"] for qas in all_qas]
sources = [qas["Source"] for qas in all_qas]

retriever_top_k_recalls = []
for top_k in top_ks:
    if best_combination["granularity"] == "paragraphs":
        top_k *= 2
        for retrieval_strategy in retrieval_strategies:
            for advanced_dense_retriever in advanced_dense_retrievers:
                if retrieval_strategy != "ensemble_retrieval":
                    break_after = True
                    best_combination["advanced_dense_retriever"] = "none"
                best_combination["top_k"] = top_k
                best_combination["retrieval_strategy"] = retrieval_strategy
                _, contexts = get_predictions(llm, questions, best_combination, best_combination)
                recall = []
                for retrieved_source, gold_source in zip(contexts, sources):
                    in_gold = 0
                    for passage in retrieved_source:
                        # use only first ten characters as identifier, should be unique enough
                        if passage[:10] in gold_source:
                            in_gold = 1
                            break
                    recall.append(in_gold)
                retriever_top_k_recall = {
                    "retriever_strategy": retrieval_strategy,
                    "advanced_dense_retriever": advanced_dense_retriever,
                    "top_k": top_k,
                    "recall": np.mean(recall)
                }
                if break_after:
                    break

ranked_retriever_top_k_recalls = sorted(retriever_top_k_recalls, key=lambda x: x["recall"], reverse=True)

# output evaluation results to files
with open("ranked_combinations.txt", "w") as file:
    relevant_combination_keys = [
        "temperature",
        "spell_checker",
        "abstract_only",
        "metadata_strategy",
        "granularity",
        "top_k",
        "retrieval_strategy",
        "advanced_dense_retriever",
    ]
    for i, combination in enumerate(ranked_combinations[:20]): # top 20 combinations
        file.write(f"combination {i}:\n")
        for key in relevant_combination_keys:
            file.write(f"\t{key}: {combination['combination'][key]}\n")
        file.write(f"scores {i}:\n")
        for score_type, score in combination["scores"].items():
            file.write(f"\t{score_type}: {score:.4f}\n")
        file.write(f"\toverall weighted score: {weighted_score(combination):.4f}\n")
        file.write("\n\n")

with open("question_types_scores.txt", "w") as file:
    for question_type, scores in question_types_scores.items():
        file.write(f"{question_type}:\n")
        for score_type, score in scores.items():
            file.write(f"\t{score_type}: {score:.4f}\n")
        file.write("\n\n")

with open("retriever_top_k_recalls.txt", "w") as file:
    for i, retriever_combination in enumerate(retriever_top_k_recalls)[:20]: # top 20 combinations
        file.write(f"retriever_combination {i}:\n")
        for key, value in retriever_combination.items():
            file.write(f"\t{key}: {value:.4f}\n")
        file.write("\n\n")