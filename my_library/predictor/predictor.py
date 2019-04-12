import numpy as np
from allennlp.predictors import Predictor
from allennlp.common.util import JsonDict


@Predictor.register('predictor')
class LanguagePredictor(Predictor):
    def predict_json(self,inputs: JsonDict) -> JsonDict:
        instance = self._dataset_reader.toInstance(inputs)
        out = self.predict_instance(instance)
        #find maximum score from predictions
        return [self._model.vocab.get_token_from_index(i,'labels') 
                for i in np.argmax(out['tag_logits'], axis=-1)]