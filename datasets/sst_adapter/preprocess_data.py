import torch
import numpy as np
from tqdm import tqdm
import sys

LABEL = "label"
SENTENCE = "sentence"
TASK_NAME = "sst-2-bert"
MODEL_URL = "bert-base-uncased"
import pickle

from transformers import AutoTokenizer, BertModel
from datasets import load_dataset
from datasets import list_datasets
# print(list_datasets())
sst_dataset = load_dataset("glue", "sst2")
print(sst_dataset.keys())
sst_model = BertModel.from_pretrained(MODEL_URL)
sst_tokenizer = AutoTokenizer.from_pretrained(MODEL_URL)
limit = int(sys.argv[1])

embeddings = []
labels = []
i = 0
for row in tqdm(sst_dataset["train"], desc="Processing Samples"):
    inputs = [row["sentence"]]
    # outputs = sst_model(**inputs, output_hidden_states = True)["pooler_output"]
    # outputs = outputs.squeeze(0)
    # # outputs = torch.max(outputs, dim= 0)[0]
    # embeddings.append(outputs.detach().numpy().tolist())
    labels.append(row["label"])
    embeddings.append(inputs)
    i += 1
    if i == limit:
        break

labels = np.array(labels)
print(sum(labels))
with open("features.pkl", "wb") as f:
    pickle.dump(embeddings, f)
np.save("./labels.npy", labels)



