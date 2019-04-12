from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.nn.util import get_text_field_mask
from allennlp.data.iterators import BucketIterator
from sklearn.model_selection import train_test_split
from allennlp.training.trainer import Trainer
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.common.util import JsonDict
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from allennlp.modules.token_embedders.token_characters_encoder import TokenCharactersEncoder
from typing import Dict
import numpy as np
import torch

@Model.register("net")
class NameNet(Model):
    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 vocab: Vocabulary) -> None:
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.hidden2tag = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                          out_features=vocab.get_vocab_size('labels'))
        self.accuracy = CategoricalAccuracy()   
    def forward(self,
               tokens: Dict[str, torch.Tensor],
               token_characters: Dict[str, torch.Tensor],
               labels: torch.Tensor = None) -> torch.Tensor:
        
        mask = get_text_field_mask(tokens)
        
        embeddings = self.word_embeddings({**tokens,**token_characters})
        
        encoder_out = self.encoder(embeddings, mask)
        
        tag_logits = self.hidden2tag(encoder_out)
        
        output = {"tag_logits": tag_logits}
        
        if labels is not None:
            self.accuracy(tag_logits, labels, mask)
            output["loss"] = sequence_cross_entropy_with_logits(tag_logits, labels, mask)

        return output
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}