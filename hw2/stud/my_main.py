import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
pl.seed_everything(42, workers=True) 

from utils_dataset import ABSADataModule, BIO_TAGS, IDX2LABEL, \
                        LAPTOP_TRAIN, LAPTOP_DEV, RESTAURANT_DEV, RESTAURANT_TRAIN
from utils_classifier import TaskAModel, TaskATransformerModel, ABSALightningModule, \
                        rnn_collate_fn,  raw_collate_fn, get_preds_terms

TRAIN      = False
NUM_EPOCHS = 20
BATCH_SIZE = 32
# testing config name
SAVE_NAME = "transf_allRnn_res2lap_2FF64_BIO"

def precisions_scores(model: pl.LightningModule, l_dataset: DataLoader, l_label_vocab):
    model.freeze()
    all_predictions = []
    all_labels = []
    for indexed_elem in l_dataset:
        indexed_in, _, indexed_labels, _, _ = indexed_elem
        predictions, _ = model(indexed_in)
        predictions = torch.argmax(predictions, -1).view(-1)
        labels = indexed_labels.view(-1)
        valid_indices = labels != 0
        
        valid_predictions = predictions[valid_indices]
        valid_labels = labels[valid_indices]
        
        all_predictions.extend(valid_predictions.tolist())
        all_labels.extend(valid_labels.tolist())

    # global precision. Does take class imbalance into account.
    micro_precision = precision_score(all_labels, all_predictions, average="micro", zero_division=0)
    # precision per class and arithmetic average of them. Does not take into account class imbalance.
    macro_precision = precision_score(all_labels, all_predictions, average="macro", zero_division=0)
    per_class_precision = precision_score(all_labels, all_predictions, labels=list(range(len(l_label_vocab))),
                                         average=None, zero_division=0)
    model.unfreeze()
    return {"micro_precision":micro_precision,
            "macro_precision":macro_precision, 
            "per_class_precision":per_class_precision}

def evaluate_precision(precisions: dict, label_d: dict=IDX2LABEL):
    per_class_precision = precisions["per_class_precision"]
    print(f"Micro Precision: {precisions['micro_precision']}")
    print(f"Macro Precision: {precisions['macro_precision']}")

    print("Per class Precision:")
    print("\tlabel\tscore")
    for idx_class, precision in sorted(enumerate(per_class_precision), key=lambda elem: -elem[1]):
        label = label_d[idx_class]
        print(f"\t{label}\t{precision:.4f}")

    return

def evaluate_extraction(model: pl.LightningModule, l_dataset: DataLoader):
    model.freeze()
    scores = {"tp": 0, "fp": 0, "fn": 0}
    for elem in l_dataset:
        inputs, _, labels, tokens, l_terms = elem
        _, preds = model(inputs)

        t_preds = get_preds_terms(preds, tokens)
        #print(t_preds)
        ll = []
        for b in l_terms:
            for l in b:
                ll.append(l)

        pred_terms  = {i for i in t_preds}
        label_terms = {t for t in ll}

        scores["tp"] += len(pred_terms & label_terms)
        scores["fp"] += len(pred_terms - label_terms)
        scores["fn"] += len(label_terms - pred_terms)

    precision = 100*scores["tp"] / (scores["tp"] + scores["fp"])
    recall = 100*scores["tp"] / (scores["tp"] + scores["fn"])
    f1 = 2 * precision*recall / (precision+recall)

    print(f"Aspect Extraction Evaluation")
    print(f"\tAspects\t TP: {scores['tp']};\tFP: {scores['fp']};\tFN: {scores['fn']}")
    print(f"\t\tprecision: {precision:.2f};\trecall: {recall:.2f};\tf1: {f1:.2f}")


#### Load train and eval data
print("\n[INFO]: Loading datasets ...")
data_module = ABSADataModule(train_path=RESTAURANT_TRAIN, dev_path=LAPTOP_DEV, collate_fn=raw_collate_fn)
train_vocab = data_module.vocabulary
# instanciate dataloaders
train_dataloader = data_module.train_dataloader()
eval_dataloader  = data_module.eval_dataloader()

#### set model hyper parameters
hparams = {
	"embedding_dim" : 768,               # embedding dimension, 100 GloVe, 768 BertModel
	"vocab_size"    : len(train_vocab),  # vocab length
	"lstm_dim"      : 128,               # LSTM hidden layer dim
    "hidden_dim"    : 64,                # hidden linear layer dim
	"output_dim"    : len(BIO_TAGS),     # num of BILOU tags to predict
 	"bidirectional" : True,              # if biLSTM or LSTM
	"rnn_layers"    : 1,
	"dropout"       : 0.3
}

print("\n[INFO]: Building model ...")
# instanciate task-specific model
#task_model = TaskAModel(hparams=hparams, embeddings=train_vocab.vectors.float())
task_model = TaskATransformerModel(hparams=hparams)

# instanciate pl.LightningModule for training with task model
model = ABSALightningModule(task_model)


#### Trainer
# checkpoint and early stopping callback for pl.Trainer()
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
logger = pl.loggers.TensorBoardLogger(save_dir='logs/', name=SAVE_NAME)

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
print("\n[INFO]: precison metrics ...")
precisions = precisions_scores(model, eval_dataloader, BIO_TAGS)
evaluate_precision(precisions=precisions)

print("\n[INFO]: evaluate extraction  ...")
evaluate_extraction(model, eval_dataloader)