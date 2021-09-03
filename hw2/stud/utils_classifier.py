import torch
import torchmetrics
from torch import nn

from typing import List, Dict

import pytorch_lightning as pl
from transformers import BertForTokenClassification, BertTokenizer, BertForSequenceClassification, \
                        RobertaForSequenceClassification, RobertaTokenizer

try:
    from utils_general import *
    from utils_dataset import _read_data_taskB
except:
    from stud.utils_general import *
    from stud.utils_dataset import _read_data_taskB

POLARITY_INV = {
	0 : "un-polarized",   # dummy label for sentences with no target
    1 : "positive",
    2 : "negative",
    3 : "neutral",
    4 : "conflict"
}


### RNNs collate functions
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
    index sequences. (data_elements is a list of (x, y, toks) tuples, where `x` is raw input text, 
    `y` the ground truth label and `toks` the tokenized input)
    """
    X = []
    y = []
    for elem in data_elements:
        X.append(elem[0])
        y.append(torch.Tensor(elem[1]))

    y = torch.nn.utils.rnn.pad_sequence(y, batch_first=True)
    return X, y, None

def raw2_collate_fn(data_elements: list):
    """
    Override the collate function in order to deal with the different sizes of the input 
    index sequences. (data_elements is a list of (x, y, toks) tuples, where `x` is raw input text, 
    `y` the ground truth label and `toks` the tokenized input)
    """
    X = []
    y = []
    toks  = []
    terms = []
    for elem in data_elements:
        X.append(elem[0])
        y.append(torch.Tensor(elem[1]))
        terms.append(elem[2])
        toks.append(elem[3])

    y = torch.nn.utils.rnn.pad_sequence(y, batch_first=True)
    return X, y, terms, toks

def cat_collate_fn(data_elements: list):
    """
    Override the collate function in order to deal with the different sizes of the input 
    index sequences. (data_elements is a list of (x, y) tuples, where x is raw input text)
    """
    X = []
    y = []
    cats = []
    for elem in data_elements:
        X.append(elem[0])
        y.append(torch.Tensor(elem[1]))
        cats.append(elem[2])

    y = torch.nn.utils.rnn.pad_sequence(y, batch_first=True)
    return X, y, cats

def seq_collate_fn(data_elements: list):
    """
    Override the collate function in order to deal with the different sizes of the input 
    index sequences. (data_elements is a list of (x, y, terms) tuples, where "x" is raw input text and 
    "term" a list of aspect terms or categories)
    """
    X = []
    y = []
    terms = []
    for elem in data_elements:
        #print(elem)
        X.extend(elem[0])
        y.extend(elem[1])
        terms.extend(elem[2])

    y = torch.Tensor(y)
    return X, y, terms

class CustomRobertaClassificationHead(nn.Module):
    """
    Override of `RobertaClassificationHead` module to customize 
    classification head for sentence-level classification tasks.
    """
    def __init__(self, hparams: dict):
        super().__init__()
        self.activation = nn.GELU() # nn.Tanh()
        self.dense      = nn.Linear(hparams["embedding_dim"], hparams["cls_hidden_dim"])
        self.dropout    = nn.Dropout(hparams["dropout"])
        self.out_proj   = nn.Linear(hparams["cls_hidden_dim"], hparams["cls_output_dim"])

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


## predict utils
def predict_taskB(model, samples: List[Dict], step_size: int=32, label_tags: Dict=POLARITY_INV, verbose=False):
    """
    Perform prediction for task B, step_size element at a time.
    """
    print("[preds]: predicting on task B ...")
    model.freeze()
    predicted = []  # List[Dict] for output

    # pre-processing data
    data_elems = _read_data_taskB(test=True, test_samples=samples)

    for step in range(0,len(data_elems), step_size):
        # test step_size samples at a time
        if step+step_size <= len(data_elems):
            step_batch = data_elems[step:step+step_size]
        else:
            step_batch = data_elems[step:]

        if verbose: print("batch_size:", len(step_batch))

        # use collate_fn to input step_size samples into the model
        x, y, gt_terms = seq_collate_fn(step_batch)
        with torch.no_grad():
            # predict with model
            out = model(x)
            logits = out.logits   
            pred_labels = torch.argmax(logits, -1)

        # build (term,aspect) couples to produce correct output for the metrics
        preds = []
        for i in range(len(gt_terms)): 
            text = x[i] if isinstance(x[i], str) else x[i][0]
            if i != len(gt_terms)-1:
                next_text = x[i+1] if isinstance(x[i+1], str) else x[i+1][0]
            
            if verbose:
                print("\ntext:", text)
                print(f"values: term: {gt_terms[i]}, pred aspect: {label_tags[int(pred_labels[i])]}")

            if gt_terms[i] != "":   # 0 -> "un-polarized"         
                # there is a prediction only if there is a ground truth term 
                # and the related polarity.  
                preds.append((gt_terms[i],label_tags[int(pred_labels[i])]))
                if verbose: print("[LOFFA]:", preds)

            if next_text != text or i == len(gt_terms)-1:
                # when input text changes we are dealing with another set of targets,
                # i.e. another prediction.
                if verbose: print("[CACCA]:", preds)
                predicted.append({"targets":preds})
                next_text = text
                preds = []

    print("Num predictions:", len(predicted))
    return predicted



### Task specific models
## task A,B
class TaskAModel(nn.Module):
    """
    Torch nn.Module to perform task A (aspect term extraction).
    """
    # we provide the hyperparameters as input
    def __init__(self, hparams: dict, embeddings = None):
        super().__init__()
        print_hparams(hparams)
        self.hparams = hparams
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

class TaskATermExtracrionModel(nn.Module):
    """
    Torch nn.Module to perform task A (aspect term extraction) with the help of a tranformer.
    """
    def __init__(self, hparams: dict, tokenizer=None, device: str="cpu"):
        super().__init__()
        self.hparams = hparams
        self.device = device
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
        if self.device == "cuda":
            for k, v in tokens.items():
                if not test:   
                    tokens[k] = v.cuda()

        y = y.long() if y is not None else None
        output = self.transfModel(**tokens, labels=y)
        return output

class TaskBAspectSentimentModel(nn.Module):
    """
    Torch nn.Module to perform task B (aspect sentiment classification) with the help of a tranformer.
    """
    def __init__(self, hparams: dict, device: str="cpu"):
        super().__init__()
        self.device  = device
        self.hparams = hparams
        print_hparams(hparams)

        self.tokenizer   = BertTokenizer.from_pretrained("bert-base-cased")
        self.transfModel = BertForSequenceClassification.from_pretrained(
            "bert-base-cased",
            num_labels=hparams["cls_output_dim"]
        )
        # custom classifier head
        classifier_head = nn.Sequential(
            nn.Linear(hparams["embedding_dim"], hparams["cls_hidden_dim"]),
            nn.GELU(), #nn.ReLU
            nn.Linear(hparams["cls_hidden_dim"], hparams["cls_output_dim"]),
        )
        self.transfModel.classifier = classifier_head
        self.transfModel.dropout = nn.Dropout(hparams["dropout"])

    def forward(self, x, y=None, test: bool=False):
        # x -> raw_input
        tokens = self.tokenizer([x[i][0] for i in range(len(x))], [x[i][1] for i in range(len(x))], 
                                return_tensors='pt', padding=True, truncation=True)
        if self.device == "cuda":
            for k, v in tokens.items():
                if not test:   
                    tokens[k] = v.cuda()

        y = None if (y is None or test) else y.long()
        output = self.transfModel(**tokens, labels=y)
        return output

    def predict(self, samples: List[Dict]):
        return predict_taskB(self, samples=samples)

class TaskABModel(nn.Module):
    def __init__(self, hparams: dict, device: str="cpu"):
        super().__init__()
        self.device  = device
        self.hparams = hparams
        print_hparams(hparams)

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        # load best task-A model
        self.A_model = ABSALightningModule(test=True).load_from_checkpoint(
            checkpoint_path="model/to_save/taskA/BERT_tA_res2res_2FFh_gelu_eps.ckpt",
            model=TaskATermExtracrionModel(hparams=hparams, tokenizer=self.tokenizer, device=device)
        )
        # load best task-B model
        self.B_model = ABSALightningModule(test=True).load_from_checkpoint(
            checkpoint_path="model/to_save/taskB/BERT_tB_res2res_2FFh_gelu3_toktok_f1.ckpt",
            model=TaskBAspectSentimentModel(hparams=hparams, device=device)
        )

    #def forward(self, x, y=None, test: bool=False):
    #    out_A = self.A_model(x)
    #    out_B = self.B_model(x)
    #    return out_A, out_B

## task C,D
class TaskCCategoryExtractionModel(nn.Module):
    """
    Torch nn.Module to perform task C (category extraction) with the help of a tranformer.
    """
    def __init__(self, hparams: dict, device: str="cpu"):
        super().__init__()
        self.device  = device
        self.hparams = hparams
        print_hparams(hparams)

        self.tokenizer   = RobertaTokenizer.from_pretrained("roberta-base")
        self.transfModel = RobertaForSequenceClassification.from_pretrained(
            "roberta-base",
            num_labels=hparams["cls_output_dim"],
            problem_type="multi_label_classification"
        )
        # custom classifier head
        self.transfModel.classifier = CustomRobertaClassificationHead(hparams)
        self.transfModel.dropout = nn.Dropout(hparams["dropout"])

    def forward(self, x, y=None, test: bool=False):
        tokens = self.tokenizer(x, return_tensors='pt', padding=True, truncation=True)
        if self.device == "cuda":
            for k, v in tokens.items():
                if not test:   
                    tokens[k] = v.cuda()

        y = None if (y is None or test) else y.float()
        output = self.transfModel(**tokens, labels=y)
        return output

class TaskDCategorySentimentModel(nn.Module):
    """
    Torch nn.Module to perform task D (category sentiment classification) with the help of a tranformer.
    """
    def __init__(self, hparams: dict, device: str="cpu"):
        super().__init__()
        self.device  = device
        self.hparams = hparams
        print_hparams(hparams)

        #self.tokenizer   = BertTokenizer.from_pretrained("bert-base-cased")
        #self.transfModel = BertForSequenceClassification.from_pretrained(
        #    "bert-base-cased",
        #    num_labels=hparams["cls_output_dim"]
        #)
        #classifier_head = nn.Sequential(
        #    ## nn.Dropout(hparams["dropout"]),
        #    nn.Linear(hparams["embedding_dim"], hparams["cls_hidden_dim"]),
        #    nn.GELU(), #nn.ReLU
        #    #nn.Dropout(hparams["dropout"]),
        #    nn.Linear(hparams["cls_hidden_dim"], hparams["cls_output_dim"]),
        #)
        #self.transfModel.classifier = classifier_head
        # custom classifier head
        self.tokenizer   = RobertaTokenizer.from_pretrained("roberta-base")
        self.transfModel = RobertaForSequenceClassification.from_pretrained(
            "roberta-base",
            num_labels=hparams["cls_output_dim"]
        )
        self.transfModel.classifier = CustomRobertaClassificationHead(hparams)
        self.transfModel.dropout = nn.Dropout(hparams["dropout"])

    def forward(self, x, y=None, test: bool=False):
        # x -> raw_input
        tokens = self.tokenizer([x[i][0] for i in range(len(x))], [x[i][1] for i in range(len(x))], 
                                return_tensors='pt', padding=True, truncation=True)
        if self.device == "cuda":
            for k, v in tokens.items():
                if not test:   
                    tokens[k] = v.cuda()

        y = None if (y is None or test) else y.long()
        output = self.transfModel(**tokens, labels=y)
        return output



### pl.LightningModule
class ABSALightningModule(pl.LightningModule):
    """
    LightningModule to easly handle training and evaluation loops with a given nn.Module.
    """
    def __init__(self, model: nn.Module=None, test : bool=False, device : str="cpu"):
        super().__init__()
        self.model  = model.cuda() if device == "cuda" else model
        if not test:
            task = self.model.hparams["task"]
            num_classes = self.model.hparams["cls_output_dim"]
        else:
            # just to initialize metrics when testing
            task = None
            num_classes = 2

        # task A metrics
        self.loss_function = nn.CrossEntropyLoss(ignore_index=None if task == "D" else 0)
        self.micro_f1 = torchmetrics.F1(
            num_classes=num_classes,
            average="micro",
            mdmc_average="global",
            ignore_index=None if (task=="D" or task=="C") else 0
        )
        self.macro_f1 = torchmetrics.F1(
            num_classes=num_classes,
            average="macro",
            mdmc_average="global",
            ignore_index=None if (task=="D" or task=="C") else 0
        )

        # task B metrics
        self.accuracy_fn = torchmetrics.Accuracy(
            num_classes=num_classes,
            ignore_index=None if (task=="D" or task=="C") else 0,   # ignore dummy "un-polarized" label
            subset_accuracy=True if task=="C" else False
        )
        return

    def forward(self, x, y=None):
        """ Perform model forward pass. """
        output = self.model(x, y)
        return output

    def training_step(self, train_batch, batch_idx):
        # Base -> x, x_lens, y, _, _ = train_batch
        # Bert -> x, y, terms = train_batch 
        x, y, _ = train_batch
        output = self.forward(x, y)

        # Training accuracy
        logits = output.logits   
        logits = torch.argmax(logits, -1) 
        train_acc = self.accuracy_fn(logits, y.int())
        self.log('train_acc', train_acc, prog_bar=True, on_epoch=True)

        # Training loss:
        loss = output.loss
        output.loss.backward(retain_graph=True)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        # Base -> x, x_lens, y, _, _ = train_batch
        # Bert -> x, y, terms = train_batch
        x, y, _ = val_batch
        output = self.forward(x, y)

        # Validation accuracy
        logits = output.logits
        logits = torch.argmax(logits, -1)
        val_acc = self.accuracy_fn(logits, y.int())
        self.log('val_acc', val_acc, prog_bar=True, on_epoch=True)

        # Micro-macro F1 scores
        micro_f1 = self.micro_f1(logits, y.int())
        self.log('micro_f1', micro_f1, prog_bar=True)

        macro_f1 = self.macro_f1(logits, y.int())
        self.log('macro_f1', macro_f1, prog_bar=True)

        # Validation loss
        val_loss = output.loss
        self.log('val_loss', val_loss, prog_bar=True, on_epoch=True)
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=3e-5, eps=1e-8)
        return optimizer

    def backward(self, loss, optimizer, optimizer_idx, *args, **kwargs):
        return super().backward(loss, optimizer, optimizer_idx, retain_graph=True, *args, **kwargs)

    def predict(self, samples: List[Dict]):
        return self.model.predict(samples)



############################
#labels = y.view(-1).long()
#loss = self.loss_function(logits, labels)
#labels = y.view(-1).long()
#sample_loss = self.loss_function(logits, labels)