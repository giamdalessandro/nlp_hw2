import os

import pytorch_lightning as pl
import torch
pl.seed_everything(42, workers=True) 

from utils_dataset import ABSADataModule, LAPTOP_TRAIN, LAPTOP_DEV
from utils_classifier import TaskAModel, ABSALightningModule

TRAIN      = False
NUM_EPOCHS = 20
BATCH_SIZE = 32

#train_laptop = ABSADataset(data_path=LAPTOP_TRAIN)
#vocab_laptop = train_laptop.vocabulary
#train_restaurant = ABSADataset(data_path=RESTAURANT_TRAIN)
#vocab_restaurant = train_restaurant.vocabulary

#### Load train and eval data
print("\n[INFO]: Loading datasets ...")
data_module  = ABSADataModule(train_path=LAPTOP_TRAIN, dev_path=LAPTOP_DEV)
vocab_laptop = data_module.vocabulary
# instanciate dataloaders
train_dataloader = data_module.train_dataloader()
eval_dataloader = data_module.eval_dataloader()

#### set model hyper parameters
hparams = {
	"embedding_dim" : 100,
	"vocab_size" : len(vocab_laptop),
	"hidden_dim" : 128,
	"output_dim" : 10,
	"bidirectional" : True,
	"num_layers" : 1,
	"dropout" : 0.0
}

print("\n[INFO]: Building model ...")
model = TaskAModel(hparams=hparams, embeddings=vocab_laptop.vectors)