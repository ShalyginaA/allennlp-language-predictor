from typing import Iterator, List, Dict
import torch
import torch.optim as optim
import numpy as np

from allennlp.data.dataset_readers import DatasetReader

from allennlp.data import Instance
from allennlp.data.fields import TextField, LabelField, SequenceLabelField

from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, TokenCharactersIndexer
from allennlp.data.tokenizers import Token

import glob
import os
import unicodedata
import string


@DatasetReader.register("data_reader")
class NameDataReader(DatasetReader):
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None,
                char_indexers: Dict[str, TokenCharactersIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.char_indexers = char_indexers or {"token_characters": TokenCharactersIndexer()}
        self.all_letters = string.ascii_letters + " .,;'"
        #the category_lines dictionary, a list of names per language
        self.category_lines = {}
        self.all_categories = []
        self.n_categories = None
        
    # Turn a Unicode string to plain ASCII
    def unicodeToAscii(self, s:str) -> str:
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
            and c in self.all_letters)
    
    # Read a file and split into lines
    def readLines(self, filename:str) -> str:
        lines = open(filename, encoding='utf-8').read().strip().split('\n')
        return [self.unicodeToAscii(line) for line in lines]
    
    #convert inputs corresponding to training example to Instance
    def toInstance(self, names: List[str], categories: List[str] = None) -> Instance:
        token_field = TextField([Token(nm) for nm in names], self.token_indexers)
        
        fields = {"tokens": token_field}
        
        fields["token_characters"] = TextField([Token(nm) for nm in names], self.char_indexers)
        
        if categories:
            fields["labels"] = SequenceLabelField(labels=categories, sequence_field = token_field)
        return Instance(fields)

    #takes a filename and produces a stream of Instances (random training examples)
    def _read(self, file_path: str) -> Iterator[Instance]:
        filenames = glob.glob(file_path)

        for f in filenames:
            category = os.path.splitext(os.path.basename(f))[0]
            self.all_categories.append(category)
            lines = self.readLines(f)
            self.category_lines[category] = lines
        self.n_categories = len(self.all_categories)
        
        name_and_category = []
        for cat in self.all_categories:
            for name in self.category_lines[cat]:
                name_and_category.append((name,cat))
                
        np.random.shuffle(name_and_category)
        
        step = 10
        
        for i in range(0,len(name_and_category),step):
            yield self.toInstance([n[0] for n in name_and_category[i:i+step]], 
                                  [n[1] for n in name_and_category[i:i+step]])
        

