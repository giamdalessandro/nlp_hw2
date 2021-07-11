import os

import pytorch_lightning as pl
import torch
pl.seed_everything(42, workers=True) 

from utils_dataset import ABSADataset, LAPTOP_TRAIN, RESTAURANT_TRAIN
from utils_classifier import TaskAModel

TRAIN      = False
NUM_EPOCHS = 20
BATCH_SIZE = 32


# Load train datasets
train_laptop = ABSADataset(data_path=LAPTOP_TRAIN)
vocab_laptop = train_laptop.vocabulary

train_restaurant = ABSADataset(data_path=RESTAURANT_TRAIN)
vocab_restaurant = train_restaurant.vocabulary

# set model hyper parameters
class HParams():
	embedding_dim = 100
	embeddings    = vocab_laptop.vectors
	vocab_size    = len(vocab_laptop)
	hidden_dim    = 128
	output_dim    = 10
	bidirectional = False
	num_layers = 1
	dropout  = 0.0

	def __str__(self) -> str:
		return f"Hyper-parameters: \n  \
	embedding_dim : {self.embedding_dim} \n  \
	vocab_size    : {self.vocab_size} \n  \
	rnn_layers    : {self.num_layers} \n  \
	bidirectional : {self.bidirectional} \n  \
	hidden_dim    : {self.hidden_dim} \n  \
	output_dim    : {self.output_dim} \n"


print("\n[INFO]: Building model ...")
hparams = HParams()
model = TaskAModel(hparams=hparams, embeddings=vocab_laptop.vectors)