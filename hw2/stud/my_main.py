import os

import pytorch_lightning as pl
pl.seed_everything(42, workers=True) 

from utils_dataset import ABSADataset


TRAIN      = False
NUM_EPOCHS = 20
BATCH_SIZE = 32



train_dataset = ABSADataset()