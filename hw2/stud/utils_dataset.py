import os
import json

from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


SEMCOR_XML  = "data/wsd_data/semcor.data_bis.xml"
SEMCOR_JSON = "data/wsd_data/semcor.data.json"

TRAIN_PATH = "data/wsd_data/wsd_train.json"
DEV_PATH   = "data/dev.jsonl"
DEV_LABELS = "data/dev_wsd.txt"


class ABSADataset(Dataset):
    """
    Override of torch base Dataset class to proerly load WSD data.
    """
    def __init__(self, 
            train_path: str=TRAIN_PATH, 
            dev_path  : str=DEV_PATH, 
            batch_size: int=32,
        ):
        self.train_path  = train_path
        self.dev_path    = dev_path
        self.batch_size  = batch_size

    def _preprocess_data(self):
        """
        Preprocess sentence pairs dataset to obtain the (sent,gloss) couples (i.e. for each sentence 
        match its (lemma,pos_tag) with all the gloss of all word senses available in WordNet).
        """
        return

    def _preprocess_test_data(self):
        """
        Preprocess test sentence pairs to obtain the (sent,gloss) couples (i.e. for each sentence 
        match its (lemma,pos_tag) with all the word sense glosses available in WordNet).
        """
        return

    def __len__(self):
        # returns the number of samples in our dataset
      return len(self.samples)

    def __getitem__(self, idx):
        # returns the idx-th sample
        return self.samples[idx]


class ABSADataModule(pl.LightningDataModule):
    """
    Override of pl.LightningDataModule class to easly handle WSDDataset for training and evaluation.
    """
    def __init__(self, 
            train_path: str=TRAIN_PATH, 
            dev_path  : str=DEV_PATH
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

