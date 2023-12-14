import torch
import csv
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from scipy.signal import argrelextrema
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
writer = csv.DictWriter(output_csv, fieldnames=["doc_info", "paragraphs"])
writer.writeheader()

nlp = spacy.load("en_core_web_sm")

paragraphs_rows = []
batch_size = 500
processed = 0

for row in reader:
    row = scripts_utils.preprocess_row(row)
    doc_info = scripts_utils.get_doc_info(row)
    sentences = nlp(row["Abstract"])
    sentences = [str(sent) for sent in sentences.sents]

    if len(sentences) <= 2:
        paragraphs_row = {"doc_info": doc_info, "paragraphs": [" ".join(sentences)]}
        paragraphs_rows.append(paragraphs_row)

    else:
        embeddings = model.encode(sentences)
        similarities = cosine_similarity(embeddings)
        values = similarities.diagonal(1)
        for i in range(2, similarities.shape[0]):
            values = np.append(values, similarities.diagonal(i))
        relevant_mean = np.mean(values)
        similarities -= relevant_mean

        num_weights = 3
        if len(sentences)-1 < num_weights:
            num_weights = len(sentences)-1

        def sigmoid(x):
            return (1 / (1 + np.exp(-x)))

        sig = np.vectorize(sigmoid)
        x = np.linspace(5, -5, num_weights)
        activation_weights = np.pad(sig(x), (0, similarities.shape[0]-num_weights))
        sim_rows = [similarities[i, i+1:] for i in range(similarities.shape[0])]
        sim_rows = [np.pad(sim_row, (0, similarities.shape[0]-len(sim_row))) for sim_row in sim_rows]
        sim_rows = np.stack(sim_rows) * activation_weights
        weighted_sums = np.insert(np.sum(sim_rows, axis=1), [0], [0])

        minimas = argrelextrema(weighted_sums, np.less)
        split_points = [minima for minima in minimas[0]]

        if split_points:
            paragraphs = []
            start = 0
            for split_point in split_points:
                paragraphs.append(sentences[start:split_point])
                start = split_point
            paragraphs.append(sentences[split_points[-1]:])
            paragraphs = [" ".join(sentence_list) for sentence_list in paragraphs]

            paragraphs_row = {"doc_info": doc_info, "paragraphs": paragraphs}
            paragraphs_rows.append(paragraphs_row)

        else:
            paragraphs_row = {"doc_info": doc_info, "paragraphs": [" ".join(sentences)]}
            paragraphs_rows.append(paragraphs_row)

    if len(paragraphs_rows) >= batch_size:
        writer.writerows(paragraphs_rows)
        processed += len(paragraphs_rows)
        paragraphs_rows = []
        print(f"until now processed {processed} documents")

if paragraphs_rows:
    writer.writerows(paragraphs_rows)
    processed += len(paragraphs_rows)
    paragraphs_rows = []

print(f"processed in total {processed} documents")

    # Visualization

    # import seaborn as sns
    # import matplotlib.pyplot as plt

    # heatmap = sns.heatmap(similarities, annot=True).set_title('Cosine similarities matrix')
    # heatmap.get_figure().savefig("heatmap.png")

    # fig, ax = plt.subplots()
    # sns.lineplot(y=weighted_sums, x=range(len(weighted_sums)), ax=ax).set_title('Relative minimas');
    # plt.vlines(x=minimas, ymin=min(weighted_sums), ymax=max(weighted_sums), colors='purple', ls='--', lw=1, label='vline_multiple - full height')
    # plt.savefig("minimas.png")
    # breakpoint()