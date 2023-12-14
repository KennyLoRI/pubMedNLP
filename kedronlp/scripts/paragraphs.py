import torch
import csv
import sys
import ast
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from scipy.signal import argrelextrema
import seaborn as sns
import matplotlib.pyplot as plt
import math
import numpy as np

# user libraries
import scripts_utils
import emb_utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print()

scripts_utils.increase_csv_maxsize()

model = emb_utils.PubMedBert(device=device)

input_csv = open("../data/01_raw/extract_data.csv")
reader = csv.DictReader(input_csv)

output_csv = open("../data/01_raw/paragraphs.csv", "w")
writer = csv.DictWriter(output_csv, fieldnames=["document_base", "paragraphs"])
writer.writeheader()

nlp = spacy.load("en_core_web_sm")

for row in reader:
    row = scripts_utils.preprocess_row(row)
    sentences = nlp(row["Abstract"])
    sentences = [str(sent) for sent in sentences.sents]

    if len(sentences) <= 2:
        continue

    embeddings = model.encode(sentences)
    similarities = cosine_similarity(embeddings)
    values = similarities.diagonal(1)
    for i in range(2, similarities.shape[0]):
        values = np.append(values, similarities.diagonal(i))
    relevant_mean = np.mean(values)
    similarities -= relevant_mean
    heatmap = sns.heatmap(similarities,annot=True).set_title('Cosine similarities matrix')
    heatmap.get_figure().savefig("heatmap.png")

    num_weights = 3
    if len(sentences)-1 < num_weights:
        num_weights = len(sentences)-1

    def sigmoid(x):
        return (1 / (1 + np.exp(-x)))

    y = np.vectorize(sigmoid)
    x = np.linspace(5, -5, num_weights)
    activation_weights = np.pad(y(x), (0, similarities.shape[0]-num_weights))
    sim_rows = [similarities[i, i+1:] for i in range(similarities.shape[0])]
    sim_rows = [np.pad(sim_row, (0, similarities.shape[0]-len(sim_row))) for sim_row in sim_rows]
    sim_rows = np.stack(sim_rows) * activation_weights
    weighted_sums = np.insert(np.sum(sim_rows, axis=1), [0], [0])

    # lets create empty fig for our plor
    fig, ax = plt.subplots()
    ### 6. Find relative minima of our vector. For all local minimas and save them to variable with argrelextrema function
    minmimas = argrelextrema(weighted_sums, np.less) #order parameter controls how frequent should be splits. I would not reccomend changing this parameter.
    # plot the flow of our text with activated similarities
    sns.lineplot(y=weighted_sums, x=range(len(weighted_sums)), ax=ax).set_title('Relative minimas');
    # Now lets plot vertical lines in order to see where we created the split
    plt.vlines(x=minmimas, ymin=min(weighted_sums), ymax=max(weighted_sums), colors='purple', ls='--', lw=1, label='vline_multiple - full height')
    plt.savefig("out.png")
    breakpoint()