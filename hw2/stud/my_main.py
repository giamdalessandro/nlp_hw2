import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
pl.seed_everything(42, workers=True) 

from utils_dataset import ABSADataModule, LAPTOP_TRAIN, LAPTOP_DEV
from utils_classifier import TaskAModel, ABSALightningModule, rnn_collate_fn

TRAIN      = False
NUM_EPOCHS = 5
BATCH_SIZE = 32



#### Load train and eval data
print("\n[INFO]: Loading datasets ...")
data_module  = ABSADataModule(train_path=LAPTOP_TRAIN, dev_path=LAPTOP_DEV, collate_fn=rnn_collate_fn)
vocab_laptop = data_module.vocabulary
# instanciate dataloaders
train_dataloader = data_module.train_dataloader()
eval_dataloader = data_module.eval_dataloader()

#### set model hyper parameters
hparams = {
	"embedding_dim" : 100,
	"vocab_size" : len(vocab_laptop),
	"hidden_dim" : 128,
	"output_dim" : 4,         # num of BILOU tags to predict
 	"bidirectional" : True,
	"num_layers" : 1,
	"dropout" : 0.0
}

print("\n[INFO]: Building model ...")
# instanciate task-specific model
task_model = TaskAModel(hparams=hparams, embeddings=vocab_laptop.vectors.float())
# instanciate pl.LightningModule for training
model = ABSALightningModule(task_model)

#### set up Trainer and callbacks
# checkpoint callback for pl.Trainer()
ckpt_clbk = ModelCheckpoint(
    monitor='train_loss',
    mode='min',
    dirpath="./model/pl_checkpoints/",
    save_last=True,
    save_top_k=3
)
logger = pl.loggers.TensorBoardLogger(save_dir='logs/')

#### Training
trainer = pl.Trainer(
    gpus=1,
    max_epochs=NUM_EPOCHS,
    logger=logger,
    callbacks=[ckpt_clbk],
    progress_bar_refresh_rate=20
)              
trainer.fit(model, train_dataloader, eval_dataloader)