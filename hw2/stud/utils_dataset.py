import os
import re
import json
import collections

import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import Vocab


LAPTOP_TRAIN     = "data/laptops_train.json"
LAPTOP_DEV       = "data/laptops_dev.json"
RESTAURANT_TRAIN = "data/restaurants_train.json"
RESTAURANT_DEV   = "data/restaurants_dev.json"


class ABSADataset(Dataset):
    """
    Override of torch base Dataset class to proerly load ABSA data.
    """
    def __init__(self, 
            train_path: str=LAPTOP_TRAIN, 
            dev_path  : str=LAPTOP_DEV, 
            batch_size: int=32,
        ):
        self.train_path  = train_path
        self.dev_path    = dev_path
        self.batch_size  = batch_size

        self.__build_vocab(train_path)


    def __tokenize_line(self, line: str, pattern='\W'):
        """
        Tokenizes a single line (e.g. "The pen is on the table" -> 
        ["the, "pen", "is", "on", "the", "table"]).
        """
        # TODO check nltk tokenize
        # TODO check string not to lower
        return re.split(pattern, line.lower())

    def __build_vocab(self, 
            data_path: str,
            vocab_size: int=1000, 
            unk_token: str="<UNK>", 
            pad_token: str="<PAD>"
        ):
        """
        Reads the dataset and builds a torchtext vocabulary over it. It adds the following 
        attributes to the class: 
            - self.distinct_words   # number of distinct words
            - self.distinct_tgts    # number of distinct targets words
    
        Args:
            - vocab_size: size of the vocabolary;
            - unk_token : token to associate with unknown words;
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
                text = self.__tokenize_line(entry["text"])
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

        print("\n[INFO]: building vocabulary ...")
        # count words occurency and frequency            
        word_counter = collections.Counter(words_list)
        self.distinct_words = len(word_counter)
        print(f"Number of distinct words: {len(word_counter)}")
        
        # count target words occurency and frequency
        tgts_counter = collections.Counter(targets_list)
        self.distinct_tgts = len(tgts_counter)
        print(f"Number of distinct targets: {len(tgts_counter)}")

        self.vocabulary = Vocab(word_counter, max_size=vocab_size, specials=[pad_token,unk_token])

        return sentences, labels

    def __len__(self):
        # returns the number of samples in our dataset
      return len(self.samples)

    def __getitem__(self, idx):
        # returns the idx-th sample
        return self.samples[idx]


class ABSADataModule(pl.LightningDataModule):
    """
    Override of pl.LightningDataModule class to easly handle ABSADataset for training and evaluation.
    """
    def __init__(self, 
            train_path: str=LAPTOP_TRAIN, 
            dev_path  : str=LAPTOP_DEV,
        ):
        super().__init__()
        self.train_path  = train_path
        self.dev_path    = dev_path

    def setup(self):
        """
        Initialize train and eval datasets from training
        """
        self.train_dataset = None
        self.eval_dataset  = None

    def test_setup(self, test_data: list):
        """
        Initialize test data for testing.
        """
        test_dataset = None
        return

    def train_dataloader(self, *args, **kwargs):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def eval_dataloader(self, *args, **kwargs):
        return DataLoader(self.eval_dataset, batch_size=self.batch_size)

