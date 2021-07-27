import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
pl.seed_everything(42, workers=True) 

from utils_dataset import ABSADataModule, BIO_TAGS, LAPTOP_TRAIN, LAPTOP_DEV, RESTAURANT_DEV, RESTAURANT_TRAIN
from utils_classifier import TaskAModel, ABSALightningModule, rnn_collate_fn

TRAIN      = False
NUM_EPOCHS = 20
BATCH_SIZE = 32

def micro_macro_precision(model:pl.LightningModule, l_dataset:DataLoader, l_label_vocab=BIO_TAGS):
    model.freeze()
    all_predictions = []
    all_labels = []

    for elem in l_dataset:
        idx_in  = elem[0]
        idx_out = elem[1]
        _, predictions = model(idx_in)
        predictions = predictions #.view(-1)
        labels = idx_out.view(-1)
        valid_label = labels != 0
        
        valid_predictions = predictions[valid_label]
        valid_labels = labels[valid_label]
        
        all_predictions.extend(valid_predictions.tolist())
        all_labels.extend(valid_labels.tolist())

    # global precision. Does take class imbalance into account.
    micro_precision = precision_score(all_labels, all_predictions, average="micro", zero_division=0)
    # precision per class and arithmetic average of them. Does not take into account class imbalance.
    macro_precision = precision_score(all_labels, all_predictions, average="macro", zero_division=0)
    class_precision = precision_score(all_labels, all_predictions, labels = list(range(len(l_label_vocab))), 
                                        average=None, zero_division=0)
    model.unfreeze()
    return {"micro_precision":micro_precision,
            "macro_precision":macro_precision, 
            "per_class_precision":class_precision}



#### Load train and eval data
print("\n[INFO]: Loading datasets ...")
data_module  = ABSADataModule(train_path=RESTAURANT_TRAIN, dev_path=LAPTOP_DEV, collate_fn=rnn_collate_fn)
vocab_laptop = data_module.vocabulary
# instanciate dataloaders
train_dataloader = data_module.train_dataloader()
eval_dataloader = data_module.eval_dataloader()

#### set model hyper parameters
hparams = {
	"embedding_dim" : 100,                 # embedding dimension
	"vocab_size"    : len(vocab_laptop),   # vocab length
	"lstm_dim"      : 128,                 # LSTM hidden layer dim
    "hidden_dim"    : 128,                  # hidden linear layer dim
	"output_dim"    : len(BIO_TAGS),       # num of BILOU tags to predict
 	"bidirectional" : True,                # if biLSTM
	"rnn_layers"    : 1,
	"dropout"       : 0.3
}

print("\n[INFO]: Building model ...")
# instanciate task-specific model
task_model = TaskAModel(hparams=hparams, embeddings=vocab_laptop.vectors.float())
# instanciate pl.LightningModule for training
model = ABSALightningModule(task_model)

#### Trainer
# checkpoint callback for pl.Trainer()
ckpt_clbk = ModelCheckpoint(
    monitor="macro_f1",
    mode="max",
    dirpath="./model/pl_checkpoints/",
    save_last=True,
    save_top_k=2
)
early_clbk = EarlyStopping(
    monitor="macro_f1",
    patience=5,
    verbose=True,
    mode="max",
    check_on_train_epoch_end=True
)
logger = pl.loggers.TensorBoardLogger(save_dir='logs/')

# training loop
trainer = pl.Trainer(
    gpus=1,
    max_epochs=NUM_EPOCHS,
    logger=logger,
    callbacks=[ckpt_clbk,early_clbk],
    progress_bar_refresh_rate=20
)
trainer.fit(model, train_dataloader, eval_dataloader)


#### compute performances
"""
precisions = micro_macro_precision(model, eval_dataloader)
per_class_precision = precisions["per_class_precision"]
print(f"Micro Precision: {precisions['micro_precision']}")
print(f"Macro Precision: {precisions['macro_precision']}")
print("Per class Precision:")
for idx_class, precision in sorted(enumerate(per_class_precision), key=lambda elem: -elem[1]):
    label = BIO_TAGS[idx_class]
    print(label, precision)
"""