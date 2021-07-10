import os

import pytorch_lightning as pl
pl.seed_everything(42, workers=True) 

from utils_dataset import ABSADataset, LAPTOP_TRAIN, RESTAURANT_TRAIN


TRAIN      = False
NUM_EPOCHS = 20
BATCH_SIZE = 32


train_laptop = ABSADataset(data_path=LAPTOP_TRAIN)
print(train_laptop.vocabulary)

train_restaurant = ABSADataset(data_path=RESTAURANT_TRAIN)
print(train_restaurant.vocabulary)