import evaluate
from bleurt import score
import numpy as np


class Scorer:
    def __init__(self):
        self.bleu = evaluate.load("bleu")
        self.rouge = evaluate.load("rouge")
        self.bertscore = evaluate.load("bertscore")
        checkpoint = "BLEURT-20"
        self.bleurt = score.BleurtScorer(checkpoint)

    def get_scores(self, predictions, references):
        bleuscore = self.bleu.compute(
            predictions=predictions,
            references=references
        )["bleu"]
        roguescore = sum(self.rouge.compute(
            predictions=predictions,
            references=references,
            use_aggregator=True
        ).values())
        bertscore = np.mean(
            self.bertscore.compute(
                predictions=predictions,
                references=references,
                lang="en"
            )["f1"]
        )
        bleurtscore = np.mean(
            self.bleurt.score(references=references, candidates=predictions)
        )

        return {
            "bleuscore": bleuscore,
            "roguescore": roguescore,
            "bertscore": bertscore,
            "bleurtscore": bleurtscore,
        }
