from torch.nn import Module, Linear
from torch.nn.functional import tanh
from transformers import BertModel
import torch.nn as nn

class Model(Module):
    def __init__(self, input_shape, nb_classes, *args, **kwargs):
        super(Model, self).__init__()
        self.bert_model = BertModel.from_pretrained("bert-base-uncased")
        self.linear_layer = nn.Linear(768, nb_classes)

    def forward(self, inputs):
        model_outputs = self.bert_model(**inputs)
        pooled_output = model_outputs["pooler_output"]
        logits = self.linear_layer(pooled_output)

        return logits

