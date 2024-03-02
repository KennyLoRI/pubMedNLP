from scorer import Scorer
import pandas as pd
import numpy as np
import torch

df = pd.read_excel("baseline.xlsx", sheet_name="ChatGPT")
device = torch.device("cpu")
scorer = Scorer(device=device)

types = df["Question Type"].unique()
dataset = {}
for a_type in types:
    questions_answers = df.loc[df["Question Type"] == a_type][
        ["Question", "Answer", "GPTAnswer"]
    ]
    dataset[a_type] = questions_answers.to_dict(orient="records")

question_types_scores = {}
for question_type, qag_list in dataset.items():
    gold_answers = [str(qag["Answer"]) for qag in qag_list]
    gpt_answers = [str(qag["GPTAnswer"]) for qag in qag_list]
    question_types_scores[question_type] = scorer.get_scores(
        predictions=gpt_answers, references=gold_answers
    )

def weighted_score(scores):
    weighted_score = (
        scores["bleuscore"] * 0.1
        + scores["rougescore"] * 0.1
        + scores["bertscore"] * 0.4
        + scores["bleurtscore"] * 0.4
    )
    return weighted_score

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

with open("baseline.txt", "w") as file:
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