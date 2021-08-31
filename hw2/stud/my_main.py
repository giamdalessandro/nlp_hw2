import pytorch_lightning as pl
pl.seed_everything(42, workers=True)

import torch, gc
if torch.cuda.is_available():
  gc.collect()
  torch.cuda.empty_cache()

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from transformers import BertTokenizer, DistilBertTokenizer

from tasks_metrics import *
from utils_dataset import ABSADataModule, BIO_TAGS, POLARITY_TAGS, \
                        LAPTOP_TRAIN, LAPTOP_DEV, RESTAURANT_DEV, RESTAURANT_TRAIN, read_json_data
from utils_classifier import TaskAModel, TaskATransformerModel, TaskBTransformerModel, \
                        ABSALightningModule, seq_collate_fn,  raw_collate_fn, get_preds_terms

DEVICE     = "cpu"
TRAIN      = False
NUM_EPOCHS = 20
BATCH_SIZE = 32

TASK       = "A"  # A, B, C or D
METRICS    = True
SAVE_NAME  = f"BERT_t{TASK}_2FFh_gelu_eps" # test config name



#### set model hyper parameters
hparams = {
    "embedding_dim" : 768,            # embedding dimension -> (100 GloVe | 768 BertModel)
    "cls_hidden_dim": 64,             # hidden linear layer dim
    "cls_output_dim": len(BIO_TAGS) if TASK == "A" else len(POLARITY_TAGS), # num of labels to predict
    "bidirectional" : True,           # if biLSTM or LSTM
    "lstm_dim"      : 128,            # LSTM hidden layer dim
    "rnn_layers"    : 1,
    "dropout"       : 0.5
}

#### Load train and eval data
print("\n[INFO]: Loading datasets ...")
if TASK == "A":
    #tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-cased")
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    collate_fn = raw_collate_fn 
elif TASK == "B":
    tokenizer = None
    collate_fn = seq_collate_fn

data_module = ABSADataModule(train_path=LAPTOP_TRAIN, dev_path=RESTAURANT_DEV, task=TASK, 
                            collate_fn=collate_fn, tokenizer=tokenizer)
train_vocab = data_module.vocabulary
hparams["vocab_size"] = len(train_vocab) # vocab length

train_dataloader = data_module.train_dataloader(num_workers=8)
eval_dataloader  = data_module.eval_dataloader(num_workers=8)


print("\n[INFO]: Building model ...")
# instanciate task-specific model
if TASK == "A":
    #task_model = TaskAModel(hparams=hparams, embeddings=train_vocab.vectors.float())
    task_model = TaskATransformerModel(hparams=hparams, tokenizer=tokenizer, device=DEVICE)

elif TASK == "B":
    task_model = TaskBTransformerModel(hparams=hparams, device=DEVICE)


if TRAIN:
    #### Trainer
    # instanciate pl.LightningModule for training with task model
    model = ABSALightningModule(task_model, device=DEVICE)

    # checkpoint and early stopping callback for pl.Trainer()
    ckpt_clbk = ModelCheckpoint(
        monitor="macro_f1",   # macro_f1 -> taskA
        mode="max",
        dirpath="./model/pl_checkpoints/",
        save_last=True,
        save_top_k=3
    )
    early_clbk = EarlyStopping(
        monitor="macro_f1",   # macro_f1 -> taskA
        patience=3, #5
        verbose=True,
        mode="max",
        check_on_train_epoch_end=True
    )
    logger = pl.loggers.TensorBoardLogger(save_dir='logs/', name=SAVE_NAME)

    # training loop
    trainer = pl.Trainer(
        gpus=1 if DEVICE == "cuda" else 0,
        max_epochs=NUM_EPOCHS,
        logger=logger,
        callbacks=[ckpt_clbk,early_clbk],
        progress_bar_refresh_rate=10
    )

    # execute training loop
    trainer.fit(model, train_dataloader, eval_dataloader)

else:
    LOAD_NAME = "BERT_tA_lap2res_2FFh_gelu_eps"
    print(f"\n[INFO]: Loading saved model '{LOAD_NAME}' ...")
    model = ABSALightningModule(test=True).load_from_checkpoint(
        checkpoint_path=F"model/to_save/task{TASK}/{LOAD_NAME}.ckpt",
        model=task_model
    )



#### compute performances -----------------------------
label_dict = BIO_TAGS if TASK == "A" else POLARITY_TAGS

if METRICS:
    print("\n[INFO]: precison metrics ...")
    precisions = precision_metrics(model, eval_dataloader, label_dict)
    evaluate_precision(precisions=precisions, task=TASK)

    #print("\n[INFO]: evaluate extraction  ...")
    #evaluate_extraction(model, eval_dataloader)

    #print("\n[INFO]: evaluate sentiment  ...")
    #samples = read_json_data(LAPTOP_DEV)
    #predictions = predict_taskB(model, samples=samples)
    #evaluate_sentiment(samples, predictions, mode="Aspect Sentiment")