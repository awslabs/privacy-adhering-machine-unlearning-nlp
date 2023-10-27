import torch
import numpy as np
from tqdm import tqdm

LABEL = "label"
SENTENCE = "sentence"
TASK_NAME = "sst-2-bert"
MODEL_URL = "bert-base-uncased"

from transformers import AutoTokenizer, BertModel
from datasets import load_dataset
sst_dataset = load_dataset("glue", "rte")
sst_model = BertModel.from_pretrained(MODEL_URL)
sst_tokenizer = AutoTokenizer.from_pretrained(MODEL_URL)

embeddings = []
labels = []
i = 0
for row in tqdm(sst_dataset["train"], desc="Processing Samples"):
    inputs = sst_tokenizer(row["sentence1"], row["sentence2"], return_tensors="pt")
    outputs = sst_model(**inputs, output_hidden_states = True)["pooler_output"]
    outputs = outputs.squeeze(0)
    # outputs = torch.max(outputs, dim= 0)[0]
    embeddings.append(outputs.detach().numpy().tolist())
    labels.append(row["label"])
    i += 1
    if i == 10000:
        break
embeddings = np.array(embeddings)
print(embeddings.shape)
labels = np.array(labels)

np.save("./features_new.npy", embeddings)
np.save("./labels_new.npy", labels)



