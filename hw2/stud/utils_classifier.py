import torch
import torchmetrics
from torch import nn

import pytorch_lightning as pl
from transformers import BertForTokenClassification, BertTokenizer, BertModel, \
                        DistilBertForTokenClassification, DistilBertTokenizer
from transformers.utils.dummy_pt_objects import DistilBertForSequenceClassification


def print_hparams(hparam: dict):
    print("\n[model]: hyperparameters ...")
    for k, v in hparam.items():
        print(f"{k}:\t{v}")
    print()

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

def get_label_tokens(targets: dict, tokenizer):
    """
    Commento sbagliato come un negroni ma senza negroni.
    """
    for tgt in targets:
        if len(tgt[1]) > 0:
            tokenizer.tokenize(tgt[1])

    return

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
        y.append(torch.Tensor(elem[1]))

    y = torch.nn.utils.rnn.pad_sequence(y, batch_first=True)
    return X, y

### Task specific models
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
    def __init__(self, hparams: dict, tokenizer=None):
        super().__init__()
        print_hparams(hparams)

        self.tokenizer   = tokenizer
        self.transfModel = BertForTokenClassification.from_pretrained(
            "bert-base-cased",
            num_labels=hparams["cls_output_dim"]
        )
        # custom classifier head
        classifier_head = nn.Sequential(
            nn.Linear(hparams["embedding_dim"], hparams["cls_hidden_dim"]),
            nn.ReLU(),
            nn.Linear(hparams["cls_hidden_dim"], hparams["cls_output_dim"]),
        )
        self.transfModel.classifier = classifier_head
        self.transfModel.dropout = nn.Dropout(hparams["dropout"])

    def forward(self, x, y=None, test: bool=False):
        # x -> raw_input
        tokens = self.tokenizer(x, return_tensors='pt', padding=True, truncation=True)
        #for k, v in tokens.items():
        #    if not test:   
        #        tokens[k] = v.cuda()

        y = y.long() if y is not None else None
        output = self.transfModel(**tokens, labels=y)
        return output

class TaskBTransformerModel(nn.Module):
    # we provide the hyperparameters as input
    def __init__(self, hparams: dict, tokenizer=None):
        super().__init__()
        print_hparams(hparams)

        self.tokenizer   = tokenizer
        self.transfModel = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-cased",
            num_labels=hparams["cls_output_dim"]
        )
        # custom classifier head
        classifier_head = nn.Sequential(
            nn.Linear(hparams["embedding_dim"], hparams["cls_hidden_dim"]),
            nn.ReLU(),
            nn.Linear(hparams["cls_hidden_dim"], hparams["cls_output_dim"]),
        )
        self.transfModel.classifier = classifier_head
        self.transfModel.dropout = nn.Dropout(hparams["dropout"])

    def forward(self, x, y=None, test: bool=False):
        # x -> raw_input
        tokens = self.tokenizer(x, return_tensors='pt', padding=True, truncation=True)
        #for k, v in tokens.items():
        #    if not test:   
        #        tokens[k] = v.cuda()

        y = y.long() if y is not None else None
        output = self.transfModel(**tokens, labels=y)
        return output


### pl.LightningModule
class ABSALightningModule(pl.LightningModule):
    """
    LightningModule to easly handle training and evaluation loops with a given nn.Module.
    """
    def __init__(self, model: nn.Module=None):
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

    def forward(self, x, y=None):
        """ Perform model forward pass. """
        output = self.model(x, y)
        return output

    def training_step(self, train_batch, batch_idx):
        # Base -> x, x_lens, y, _, _ = train_batch
        # Bert -> x, y = train_batch 
        x, y = train_batch
        output = self.forward(x, y)
        logits = output.logits    
    
        # Compute the loss:
        #labels = y.view(-1).long()
        #loss = self.loss_function(logits, labels)
        loss = output.loss
        #output.loss.backward(retain_graph=True)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        # Base -> x, x_lens, y, _, _ = train_batch
        # Bert -> x, y = train_batch
        x, y = val_batch
        output = self.forward(x, y)
        logits = output.logits
        preds = torch.argmax(logits, -1)

        # Compute F1 scores
        micro_f1 = self.micro_f1(preds, y.int())
        self.log('micro_f1', micro_f1, prog_bar=True)

        macro_f1 = self.macro_f1(preds, y.int())
        self.log('macro_f1', macro_f1, prog_bar=True)

        # Compute validation loss
        #labels = y.view(-1).long()
        #sample_loss = self.loss_function(logits, labels)
        val_loss = output.loss
        self.log('valid_loss', val_loss, prog_bar=True, on_epoch=True)
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-5, eps=1e-8)
        return optimizer

    def backward(self, loss, optimizer, optimizer_idx, *args, **kwargs):
        return super().backward(loss, optimizer, optimizer_idx, retain_graph=True, *args, **kwargs)

