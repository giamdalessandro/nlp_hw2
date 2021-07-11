import os
import re
import json
import collections
import torch

import torchtext
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import Vocab


LAPTOP_TRAIN     = "data/laptops_train.json"
LAPTOP_DEV       = "data/laptops_dev.json"
RESTAURANT_TRAIN = "data/restaurants_train.json"
RESTAURANT_DEV   = "data/restaurants_dev.json"

    
def load_pretrained_embeddings(vocabulary: dict, max_size: int):
    """
    Loads pretrained word embeddings.
    """
    # get GloVe 6B pre-trained word embeddings, of dimension 100
    glove_vec = torchtext.vocab.GloVe(name="6B", dim=100, unk_init=torch.Tensor.normal_)

    pretrained = []
    for k, _ in vocabulary.stoi.items():
        if k == "<PAD>":
            emb = torch.zeros([glove_vec.dim])
        elif k == "<UNK>":
            emb = torch.rand([glove_vec.dim])
        else:
            emb = glove_vec.get_vecs_by_tokens(k, lower_case_backup=True)
        pretrained.append(emb) 

    # return a tensor of size [vocab_size, emb_dim]
    return torch.stack(pretrained, dim=0)


class ABSADataset(Dataset):
    """
    Override of torch base Dataset class to proerly load ABSA restaurants and laptop data.
    """
    def __init__(self, 
            data_path : str=LAPTOP_TRAIN,
            unk_token : str="<UNK>", 
            pad_token : str="<PAD>"
        ):
        self.data_path = data_path
        self._build_vocab(data_path, unk_token=unk_token, pad_token=pad_token)

    def _tokenize_line(self, line: str, pattern='\W'):
        """
        Tokenizes a single line splitting on regex `\W`, i.e. non-word characters 
        (e.g. "The pen is on the table" -> ["the, "pen", "is", "on", "the", "table"]).
        """
        # TODO check nltk tokenize
        # TODO check string not to lower
        return re.split(pattern, line.lower())

    def _build_vocab(self, 
            data_path : str,
            vocab_size: int=3500, 
            unk_token : str="<UNK>", 
            pad_token : str="<PAD>"
        ):
        """
        Reads the dataset and builds a torchtext vocabulary over it. It adds the following 
        attributes to the class: 
            - self.distinct_words   # number of distinct words
            - self.distinct_tgts    # number of distinct targets words
            - self.vocabulary       # torchtext.Vocab vocabulary over data
    
        Args:
            - `vocab_size`: size of the vocabolary;
            - `unk_token` : token to associate with unknown words;
            - `pad_token` : token to indicate padding;
        """       
        print(f"\n[INFO]: Loading data from '{data_path}'...")
        sentences = []
        labels    = []
        words_list   = []
        targets_list = []

        with open(data_path, "r") as f:
            json_data = json.load(f)
            for entry in json_data:
                # tokenize data sentences
                text = self._tokenize_line(entry["text"])
                words_list.extend(text)

                # get target words
                targets = entry["targets"]
                if len(targets) > 0:
                    for tgt in targets:
                        targets_list.append(tgt[1])

                sentences.append(entry)
                labels.append(targets)
                
        assert len(sentences) == len(labels)
        print("sentence pairs:",len(sentences))
        print("labels:",len(labels))

        print("\n[dataset]: building vocabulary ...")
        # count words occurency and frequency            
        word_counter = collections.Counter(words_list)
        self.distinct_words = len(word_counter)
        print(f"Number of distinct words: {len(word_counter)}")
        
        # count target words occurency and frequency
        tgts_counter = collections.Counter(targets_list)
        self.distinct_tgts = len(tgts_counter)
        print(f"Number of distinct targets: {len(tgts_counter)}")

        # load pretrained GloVe word embeddings
        glove_vec = torchtext.vocab.GloVe(name="6B", dim=100, unk_init=torch.Tensor.normal_)
        self.vocabulary = Vocab(
            counter=word_counter,                # (word,freq) mapping
            max_size=vocab_size,                 # vocabulary max size
            specials=[pad_token,unk_token],      # special tokens
            vectors=glove_vec                    # pre-trained embeddings
        )
        # ensure pad_token embedding is a zeros tensor
        self.vocabulary.vectors[0] = torch.zeros([glove_vec.dim])
        print("Embedding vectors:", self.vocabulary.vectors.size())
        
        self.samples = zip(sentences,labels)
        return sentences, labels

    def __len__(self):
        # returns the number of samples in our dataset
      return len(self.samples)

    def __getitem__(self, idx):
        # returns the idx-th sample
        return self.samples[idx]


class ABSADataModule(pl.LightningDataModule):
    """ TODO
    Override of pl.LightningDataModule class to easly handle ABSADataset for training and evaluation.
    """
    def __init__(self, 
            train_path: str=LAPTOP_TRAIN, 
            dev_path  : str=LAPTOP_DEV,
            batch_size: int=32,
            mode      : str="restaurants"
        ):
        super().__init__()
        self.train_path = train_path
        self.dev_path = dev_path
        self.batch_size = batch_size
        self.mode = mode

    def setup(self):
        """
        Initialize train and eval datasets from training
        """
        # TODO check if need both dataset together
        self.train_laptop = ABSADataset(data_path=LAPTOP_TRAIN)
        self.eval_laptop  = ABSADataset(data_path=LAPTOP_DEV)
        self.train_restaurant = ABSADataset(data_path=RESTAURANT_TRAIN)
        self.eval_restaurant  = ABSADataset(data_path=RESTAURANT_DEV)

    def test_setup(self, test_data: list):
        """
        Initialize test data for testing.
        """
        test_dataset = None
        return

    def train_dataloader(self, *args, **kwargs):
        train_dataset = self.train_restaurant if self.mode == "restaurants" else self.train_laptop
        return DataLoader(train_dataset, batch_size=self.batch_size)

    def eval_dataloader(self, *args, **kwargs):
        eval_dataset = self.eval_restaurant if self.mode == "restaurants" else self.eval_laptop
        return DataLoader(eval_dataset, batch_size=self.batch_size)

