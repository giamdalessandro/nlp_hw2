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
# create dataloaders
train_dataloader = data_module.train_dataloader()
eval_dataloader = data_module.eval_dataloader()

#### set model hyper parameters
class HParams():
	embedding_dim = 100
	embeddings    = vocab_laptop.vectors
	vocab_size    = len(vocab_laptop)
	hidden_dim    = 128
	output_dim    = 10
	bidirectional = True
	num_layers = 1
	dropout = 0.0

	def __str__(self) -> str:
		return f"Hyper-parameters: \n\tembedding_dim : {self.embedding_dim} \n     \
	vocab_size    : {self.vocab_size} \n\trnn_layers    : {self.num_layers} \n     \
	bidirectional : {self.bidirectional} \n\thidden_dim    : {self.hidden_dim} \n  \
	output_dim    : {self.output_dim} \n"


print("\n[INFO]: Building model ...")
hparams = HParams()
model = TaskAModel(hparams=hparams, embeddings=vocab_laptop.vectors)