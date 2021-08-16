import torch
import torchmetrics
from torch import nn

import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
                        BertTokenizer, BertModel

def print_hparams(hparam: dict):
    print("\n[model]: hyperparameters ...")
    for k, v in hparam.items():
        print(f"{k}:\t{v}")
    print()

def rnn_collate_fn(data_elements: list):
    """
    Override the collate function in order to deal with the different sizes of the input 
    index sequences. (data_elements is a list of (x, y, token, terms) tuples)
    """
    X = []
    x_lens = []
    y = []
    tokens = []
    terms = []
    for elem in data_elements:
        X.append(torch.Tensor(elem[0]))
        x_lens.append(len(elem[0]))
        y.append(torch.Tensor(elem[1]))
        tokens.append(elem[2])
        terms.append(elem[3])

    X = torch.nn.utils.rnn.pad_sequence(X, batch_first=True)
    x_lens = torch.Tensor(x_lens)
    y = torch.nn.utils.rnn.pad_sequence(y, batch_first=True)
    return X, x_lens, y, tokens, terms

def raw_collate_fn(data_elements: list):
    """
    Override the collate function in order to deal with the different sizes of the input 
    index sequences. (data_elements is a list of (x, y) tuples, where x is raw input text)
    """
    X = []
    y = []
    for elem in data_elements:
        X.append(elem[0])
        y.append(elem[1])

    return X, y

def get_preds_terms(preds, tokens):
    """
    Extract predicted aspect terms from predicted tags sequence (batch-wise).
    """
    #print("\npreds:",preds.size())
    #print("tokens:", len(tokens))
    pred_terms = []
    for b in range(len(preds)):
        #print("preds:", preds[b])
        for p in range(len(tokens[b])): # len(tokens)
            if preds[b][p] != 0 and preds[b][p] != 4:
                pred_terms.append(tokens[b][p])

    return pred_terms

def remove_batch_padding(rnn_out: torch.Tensor, lenghts):
    # useless if not averaging rnn output
    clean_batch = []
    last_idxs = lenghts - 1
    rnn_out = rnn_out[0]
    print("rnn out size:", rnn_out.size())
    batch_size = rnn_out.size(0)

    for i in range(batch_size):
        words_output = torch.split(rnn_out[i], last_idxs[i])[0]
        #print("words out:", words_output.size())
        clean_batch.append(words_output)
        
    vectors = clean_batch # torch.stack(clean_batch)
    #print("vectors out:", vectors.size())
    return vectors


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
            hidden_size=hparams["lstm_dim"], 
            bidirectional=hparams["bidirectional"],
            num_layers=hparams["rnn_layers"], 
            dropout=hparams["dropout"] if hparams["rnn_layers"] > 1 else 0,
            batch_first=True
        )

        # classifier head
        lstm_output_dim = hparams["lstm_dim"] if hparams["bidirectional"] is False else hparams["lstm_dim"]*2
        self.hidden = nn.Linear(lstm_output_dim, hparams["hidden_dim"])
        self.output = nn.Linear(hparams["hidden_dim"], hparams["output_dim"])
    
    def forward(self, x, x_lens):
        embeddings = self.word_embedding(x.long())
        embeddings = self.dropout(embeddings)
        rnn_out, (h, c)  = self.lstm(embeddings)        
        o = self.dropout(rnn_out)
        hidden = self.hidden(o)
        output = self.output(hidden)
        return output

class TaskATransformerModel(nn.Module):
    # we provide the hyperparameters as input
    def __init__(self, hparams: dict):
        super().__init__()
        print_hparams(hparams)
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(hparams["dropout"])

        self.tokenizer = BertTokenizer.from_pretrained("ykacer/bert-base-cased-imdb-sequence-classification")
        self.transfModel = BertModel.from_pretrained("ykacer/bert-base-cased-imdb-sequence-classification")

        # Recurrent layer
        self.lstm = nn.LSTM(
            input_size=hparams["embedding_dim"], 
            hidden_size=hparams["lstm_dim"], 
            bidirectional=hparams["bidirectional"],
            num_layers=hparams["rnn_layers"], 
            dropout=hparams["dropout"] if hparams["rnn_layers"] > 1 else 0,
            batch_first=True
        )

        # classifier head
        lstm_output_dim = hparams["lstm_dim"] if hparams["bidirectional"] is False else hparams["lstm_dim"]*2
        self.hidden = nn.Linear(lstm_output_dim, hparams["hidden_dim"])
        self.output = nn.Linear(hparams["hidden_dim"], hparams["output_dim"])
    
    def forward(self, x):
        # x -> (raw_sentence,raw_targets)
        tokens = self.tokenizer(x, return_tensors='pt', padding=True, truncation=True)
        transf_out = self.transfModel(**tokens)
        #transf_out = self.dropout(transf_out.last_hidden_state)
        o, (h, c) = self.lstm(transf_out.last_hidden_state)
        o = self.dropout(o)

        hidden = self.hidden(o)
        output = self.output(hidden)
        return output


class ABSALightningModule(pl.LightningModule):
    """
    LightningModule to easly handle training and evaluation loops with a given nn.Module.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

        # task A metrics
        self.loss_function = nn.CrossEntropyLoss(ignore_index=0)
        self.micro_f1 = torchmetrics.F1(
            num_classes=5,
            average="micro",
            mdmc_average="global",
            ignore_index=0
        )
        self.macro_f1 = torchmetrics.F1(
            num_classes=5,
            average="macro",
            mdmc_average="global",
            ignore_index=0
        )
        return

    def forward(self, x):
        # add output processing
        logits = self.model(x)
        predictions = torch.argmax(logits, -1)
        return logits, predictions

    def training_step(self, train_batch, batch_idx):
        #x, x_lens, y, _, _ = train_batch
        x, y = train_batch
        # We receive one batch of data and perform a forward pass:
        logits, preds = self.forward(x)
        logits = logits.view(-1, logits.shape[-1])
        labels = y.view(-1).long()

        # Compute the loss:
        loss = self.loss_function(logits, labels)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        #x, x_lens, y, _, _ = val_batch
        x, y = val_batch
        logits, preds = self.forward(x)
        logits = logits.view(-1, logits.shape[-1])
        labels = y.view(-1).long()
        #print("logits:", logits.size())        
        #print("labels:", labels.size())        

        # Compute F1 scores
        micro_f1 = self.micro_f1(preds, y.int())
        self.log('micro_f1', micro_f1, prog_bar=True)

        macro_f1 = self.macro_f1(preds, y.int())
        self.log('macro_f1', macro_f1, prog_bar=True)

        # Compute validation loss 
        sample_loss = self.loss_function(logits, labels)
        self.log('valid_loss', sample_loss, prog_bar=True, on_epoch=True)
        return

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

