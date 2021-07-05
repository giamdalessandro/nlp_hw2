import os
#import nltk

import pytorch_lightning as pl
pl.seed_everything(42, workers=True) 

'''
TRAIN      = False
NUM_EPOCHS = 20
BATCH_SIZE = 32
TRAINSIZE  = None

print("\n[INFO]: Building dataset...")
data_module = WSDDataModule(batch_size=BATCH_SIZE)

# Initialize train and eval dataloader
train_dataloader = data_module.train_dataloader()
eval_dataloader = data_module.eval_dataloader(collate_fn=rnn_collate_fn)

# Set checkpoint and earlyStop callbacks for pl.Trainer()
ckpt_clbk = ModelCheckpoint(
    monitor='eval_WiC_acc_A',
    mode='max',
    dirpath="./model/pl_checkpoints/",
    save_last=True,
    save_top_k=3
)
early_clbk = EarlyStopping(
    monitor='eval_WiC_acc_A',
    patience=5,
    verbose=True,
    mode='max',
    check_on_train_epoch_end=True
)
logger = pl.loggers.TensorBoardLogger(save_dir='logs/')

if TRAIN:
    print("\n[INFO]: Building model...")
    model = WordSensesClassifier(input_dim=768, hidden_dim=128)
    
else:
    print("\n[INFO]: Loading saved model ...")
    model = WordSensesClassifier().load_from_checkpoint(checkpoint_path="model/BERT_bestResult.ckpt")

trainer = pl.Trainer(
    gpus=1,
    max_epochs=NUM_EPOCHS,
    logger=logger,
    callbacks=[ckpt_clbk, early_clbk],
    progress_bar_refresh_rate=20
)              
trainer.fit(model, train_dataloader, eval_dataloader)
'''