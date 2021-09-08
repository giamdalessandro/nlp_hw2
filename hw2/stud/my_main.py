import pytorch_lightning as pl
pl.seed_everything(42, workers=True)

import torch, gc
if torch.cuda.is_available():
  gc.collect()
  torch.cuda.empty_cache()

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from transformers import BertTokenizer, DistilBertTokenizer

from utils_metrics import *
from utils_classifier import *
from utils_dataset import CATEGORY_TAGS, BIO_TAGS, POLARITY_TAGS, POLARITY_2_TAGS, \
                        LAPTOP_TRAIN, LAPTOP_DEV, RESTAURANT_DEV, RESTAURANT_TRAIN, \
                        ABSADataModule, read_json_data
from utils_general import raw_collate_fn, raw2_collate_fn, seq_collate_fn, cat_collate_fn

DEVICE     = "cpu"
TRAIN      = False
NUM_EPOCHS = 20
BATCH_SIZE = 32

TASK       = "AB"  # A, B, C, D, AB or CD
METRICS    = True
SAVE_NAME  = f"RoBERTa_tttttt{TASK}_2FFh_gelu_eps" # test config name


#### set model hyper parameters
if TASK == "A" or TASK == "AB":
    num_classes = len(BIO_TAGS)
elif TASK == "B":
    num_classes = len(POLARITY_TAGS) 
elif TASK == "C" or TASK == "CD":
    num_classes = len(CATEGORY_TAGS)
elif TASK == "D":
    # no need for dummy "un-polarized" label
    num_classes = len(POLARITY_2_TAGS)

hparams = {
    "task"          : TASK,
    "embedding_dim" : 768,           # embedding dimension -> (100 GloVe | 768 BertModel)
    "cls_hidden_dim": 64,            # hidden linear layer dim
    "cls_output_dim": num_classes,   # num of labels to predict
    #"bidirectional" : True,         # if biLSTM or LSTM
    #"lstm_dim"      : 128,          # LSTM hidden layer dim
    #"rnn_layers"    : 1,
    "dropout"       : 0.2
}

#### Load train and eval data
print("\n[INFO]: Loading datasets ...")
if TASK == "A":
    #tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-cased")
    #tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    collate_fn = raw_collate_fn if TRAIN else raw2_collate_fn
else: 
    # B,C or D
    tokenizer = None
    collate_fn = cat_collate_fn if TASK == "C" else seq_collate_fn
    
if (TASK != "AB" and TASK != "CD"): 
    data_module = ABSADataModule(train_path=RESTAURANT_TRAIN, dev_path=LAPTOP_DEV, task=TASK, 
                                collate_fn=collate_fn, tokenizer=tokenizer, test=(not TRAIN))
    train_vocab = data_module.vocabulary
    hparams["vocab_size"] = len(train_vocab) # vocab length

    # need the eval_dataloader also for testing
    eval_dataloader  = data_module.eval_dataloader(num_workers=8)


print("\n[INFO]: Building model ...")
# instanciate task-specific model
if TASK == "A":
    #task_model = TaskAModel(hparams=hparams, embeddings=train_vocab.vectors.float())
    task_model = TaskATermExtracrionModel(hparams=hparams, tokenizer=tokenizer, device=DEVICE)
elif TASK == "B":
    task_model = TaskBAspectSentimentModel(hparams=hparams, device=DEVICE)
elif TASK == "C":
    task_model = TaskCCategoryExtractionModel(hparams=hparams, device=DEVICE)
elif TASK == "D":
    task_model = TaskDCategorySentimentModel(hparams=hparams, device=DEVICE)


if TRAIN:
    #### Trainer
    train_dataloader = data_module.train_dataloader(num_workers=8)
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
    if TASK == "AB":
        model = TaskABModel(hparams=hparams, device=DEVICE)
    elif TASK == "CD":
        model = TaskCDModel(hparams=hparams, device=DEVICE)
    else:
        LOAD_NAME = "BERT_tA_res2res_2FFh_gelu_eps"
        print(f"\n[INFO]: Loading saved model '{LOAD_NAME}' ...")
        model = ABSALightningModule(test=True).load_from_checkpoint(
            checkpoint_path=F"model/to_save/task{TASK}/{LOAD_NAME}.ckpt",
            model=task_model
        )



#### compute performances -----------------------------
if TASK == "A":
    label_dict = BIO_TAGS 
elif TASK == "B":
    label_dict = POLARITY_TAGS 
elif TASK == "C":
    label_dict = CATEGORY_TAGS 
elif TASK == "D":
    label_dict = POLARITY_2_TAGS 


if METRICS:
    print("\n[INFO]: precison metrics ...")
    if TASK != "C" and (TASK != "AB" and TASK != "CD"):
        precisions = precision_metrics(model, eval_dataloader, label_dict, task=TASK)
        evaluate_precision(precisions=precisions, task=TASK)

    if TASK == "A":
        print("\n[INFO]: evaluate extraction  ...")
        evaluate_extraction(model, eval_dataloader)

    else:
        print(f"\n[INFO]: evaluate sentiment on task '{TASK}' ...")
        samples = read_json_data(RESTAURANT_DEV)
        if TASK == "B":
            predictions = predict_taskB(model, samples=samples)
            evaluate_sentiment(samples, predictions, mode="Aspect Sentiment")

        elif TASK == "C":
            # TODO
            predictions = predict_taskC(model, samples=samples)
            evaluate_sentiment(samples, predictions, mode="Category Extraction")

        elif TASK == "D":
            predictions = predict_taskD(model, samples=samples)
            evaluate_sentiment(samples, predictions, mode="Category Sentiment")
            
        elif TASK == "AB":
            model.predict(samples=samples)
        
        elif TASK == "CD":
            model.predict(samples=samples)
            