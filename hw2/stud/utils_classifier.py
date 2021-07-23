import torch
import torchmetrics
from torch import nn
import pytorch_lightning as pl


def print_hparams(hparam: dict):
    print("\n[model]: hyperparameters ...")
    for k, v in hparam.items():
        print(f"{k}:\t{v}")
    print()

def rnn_collate_fn(data_elements: list):
    """
    Override the collate function in order to deal with the different sizes of the input 
    index sequences. (data_elements is a list of (x, y) tuples)
    """
    X = []
    x_lens = []
    y = []
    for elem in data_elements:
        X.append(torch.Tensor(elem[0]))
        x_lens.append(len(elem[0]))
        y.append(torch.Tensor(elem[1]))

    X = torch.nn.utils.rnn.pad_sequence(X, batch_first=True)
    x_lens = torch.Tensor(x_lens)
    y = torch.nn.utils.rnn.pad_sequence(y, batch_first=True)
    return X, x_lens, y


class TaskAModel(nn.Module):
    # we provide the hyperparameters as input
    def __init__(self, hparams: dict, embeddings = None):
        super().__init__()
        print_hparams(hparams)
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(hparams["dropout"])
        
        # 
        self.word_embedding = nn.Embedding.from_pretrained(embeddings)

        # Recurrent layer
        self.lstm = nn.LSTM(
            input_size=hparams["embedding_dim"], 
            hidden_size=hparams["hidden_dim"], 
            bidirectional=hparams["bidirectional"],
            num_layers=hparams["num_layers"], 
            dropout=hparams["dropout"] if hparams["num_layers"] > 1 else 0,
            batch_first=True
        )

        # classifier head
        lstm_output_dim = hparams["hidden_dim"] if hparams["bidirectional"] is False else hparams["hidden_dim"]*2
        self.classifier = nn.Linear(lstm_output_dim, hparams["output_dim"])
    
    def forward(self, x):
        embeddings = self.word_embedding(x.long())
        embeddings = self.dropout(embeddings)
        o, (h, c) = self.lstm(embeddings)
        o = self.dropout(o)
        output = self.classifier(o)
        #output = self.softmax(output)
        return output


class ABSALightningModule(pl.LightningModule):
    """
    LightningModule to easly handle training and evaluation loops with a given nn.Module.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.loss_function = nn.CrossEntropyLoss(ignore_index=0)
        self.f1_score = torchmetrics.F1(mdmc_average="global")
        return

    def forward(self, x):
        # may add final one-hot encoding heres
        logits = self.model(x)
        predictions = torch.argmax(logits, -1)
        return logits, predictions

    def training_step(self, train_batch, batch_idx):
        x, x_lens, y = train_batch
        # We receive one batch of data and perform a forward pass:
        logits, preds = self.forward(x)
        logits = logits.view(-1, logits.shape[-1])
        labels = y.view(-1).long()

        # Compute the loss:
        loss = self.loss_function(logits, labels)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, x_lens, y = val_batch
        logits, preds = self.forward(x)
        logits = logits.view(-1, logits.shape[-1])
        labels = y.view(-1).long()

        # Compute F1 score
        #print("logits:", logits.size())        
        #print("labels:", labels.size())        
        f1 = self.f1_score(preds, y.int())
        self.log('f1_score', f1, prog_bar=True)

        # Compute validation loss 
        sample_loss = self.loss_function(logits, labels)
        self.log('valid_loss', sample_loss, prog_bar=True)
        return

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

