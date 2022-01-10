from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.nn import util
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder
from allennlp.training.metrics import CategoricalAccuracy, FBetaMeasure
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

LABEL_TO_INDEX = {'sadness': 0,
                  'anger': 1,
                  'anticipation': 2,
                  'fear': 3,
                  'joy': 4,
                  'disgust': 5,
                  'surprise': 6,
                  'trust': 7}

@Model.register('sentiment_classifier')
class SentimentClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder):
        super().__init__(vocab)
        self.embedder = embedder
        self.encoder = encoder
        num_labels = len(LABEL_TO_INDEX)
        self.classifier = nn.Linear(encoder.get_output_dim(), num_labels)
        self.accuracy = CategoricalAccuracy()
        self.fbeta = FBetaMeasure(labels=list(LABEL_TO_INDEX.values()), average='weighted')

    def forward(self,
                text: TextFieldTensors,
                label: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        output = {}
        token_embeds = self.embedder(text)
        mask = util.get_text_field_mask(text)
        encoding = self.encoder(token_embeds, mask=mask)
        logits = self.classifier(encoding)
        probs = F.softmax(logits, dim=1)
        output['probs'] = probs
        if label is not None:
            loss = F.cross_entropy(logits, label)
            output['loss'] = loss
            self.fbeta(probs, label)
            self.accuracy(probs, label)
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {"accuracy": self.accuracy.get_metric(
            reset), **self.fbeta.get_metric(reset)}
        return metrics