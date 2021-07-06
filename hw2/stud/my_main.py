import os
#import nltk

import pytorch_lightning as pl
pl.seed_everything(42, workers=True) 

TRAIN      = False
NUM_EPOCHS = 20
BATCH_SIZE = 32
