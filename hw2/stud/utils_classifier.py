import torch
from torch import nn
import pytorch_lightning as pl


class TaskAModel(nn.Module):
    # we provide the hyperparameters as input
    def __init__(self, hparams, embeddings = None):
        super().__init__()
        print(hparams)
        self.dropout = nn.Dropout(hparams.dropout)
        
        # word embeddings
        print("initializing embeddings from pretrained ...")
        self.word_embedding = nn.Embedding.from_pretrained(embeddings)

        # Recurrent layer
        self.lstm = nn.LSTM(
            input_size=hparams.embedding_dim, 
            hidden_size=hparams.hidden_dim, 
            bidirectional=hparams.bidirectional,
            num_layers=hparams.num_layers, 
            dropout=hparams.dropout if hparams.num_layers > 1 else 0
        )

        # classifier head
        lstm_output_dim = hparams.hidden_dim if hparams.bidirectional is False else hparams.hidden_dim * 2
        self.classifier = nn.Linear(lstm_output_dim, hparams.output_dim)
    
    def forward(self, x):
        embeddings = self.word_embedding(x)
        embeddings = self.dropout(embeddings)
        o, (h, c) = self.lstm(embeddings)
        o = self.dropout(o)
        output = self.classifier(o)
        return output


class ABSALightningModule(pl.LightningModule):
    """
    LightningModule to easly handle training and evaluation loops with a given nn.Module.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        return    

    def forward(self, x):
        logits = self.model(x)
        predictions = torch.argmax(logits, -1)
        return logits, predictions

    def training_step(self, train_batch, batch_idx):
        # x, y = train_batch
        return

    def validation_step(self, val_batch, batch_idx):
        return

    # TODO may be needed
    ##def backward(self, loss, optimizer, optimizer_idx, *args, **kwargs):
    ##    return super().backward(loss, optimizer, optimizer_idx, retain_graph=True, *args, **kwargs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

