import torch
import torchmetrics
from torch import nn

from typing import List, Dict

import pytorch_lightning as pl
from transformers import BertForTokenClassification, BertTokenizer, BertForSequenceClassification, \
                        RobertaForSequenceClassification, RobertaTokenizer


try:
    from utils_general import *
    from utils_metrics import CATEGORY_INV, IDX2LABEL, POLARITY_2_INV, POLARITY_INV
    from utils_dataset import _read_data_taskA, _read_data_taskB, _read_data_taskC, _read_data_taskD
except:
    from stud.utils_general import *
    from stud.utils_dataset import _read_data_taskA, _read_data_taskB, _read_data_taskC, _read_data_taskD
    from stud.utils_metrics import CATEGORY_INV, IDX2LABEL, POLARITY_2_INV, POLARITY_INV

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


### task predict
def predict_taskAB(model, samples: List[Dict], tokenizer=None, step_size: int=32, label_tags: Dict=POLARITY_INV, verbose=False):
    """ TODO
    Perform prediction for task A+B, step_size element at a time.
    """
    print("[preds]: predicting on task A+B ...")
    #model.freeze()
    predicted = []  # List[Dict] for output

    # pre-process data
    dataA_elems = _read_data_taskA(tokenizer=tokenizer, test=True, test_samples=samples)
    #dataB_elems = _read_data_taskB(test=True, test_samples=samples)

    for step in range(0,len(dataA_elems), step_size):
        # test step_size samples at a time
        if step+step_size <= len(dataA_elems):
            step_batch_A = dataA_elems[step:step+step_size]
            #step_batch_B = dataB_elems[step:step+step_size]
        else:
            step_batch_A = dataA_elems[step:]
            #step_batch_B = dataB_elems[step:]

        if verbose: print("batch_size:", len(step_batch_A))

        # use collate_fn to input step_size samples into the model
        x_A, _, _, tokens = raw2_collate_fn(step_batch_A)
        #print(x_A)
        #x_B, _, gt_terms = seq_collate_fn(step_batch_B)
        with torch.no_grad():
            # predict with modelAB
            out_A = model.A_model(x_A[0])

            logits_A = out_A.logits   
            pred_tokens = torch.argmax(logits_A, -1)
            #print(pred_tokens)
            _, pred_terms = get_preds_terms(pred_tokens, tokens, roberta=True)

            #logits_B = out_B.logits   
            #pred_sents = torch.argmax(logits_B, -1)


        # build (term,aspect) couples to produce correct output for the metrics
        preds = []
        for i in range(len(pred_terms)): 
            if verbose:
                print("\npred terms:", pred_terms[i])
                #print(f"label term: {l_terms[i]}")

            for j in pred_terms[i]:
                # for each predicted term build a couple
                out_B = model.B_model([x_A[i],pred_terms[i]])
                logits_B = out_B.logits   
                pred_sents = torch.argmax(logits_B, -1)
                  
                preds.append((pred_terms[i],label_tags[int(pred_sents)]))
                if verbose: print("[LOFFA]:", preds)

            if verbose: print("[CACCA]:", preds)
            predicted.append({"targets":preds})
            preds = []

    print("Num predictions:", len(predicted))
    return predicted

def predict_taskB(model,  samples: List[Dict], step_size: int=32, label_tags: Dict=POLARITY_INV, verbose=False):
    """
    Perform prediction for task B, step_size element at a time.
    """
    print("[preds]: predicting on task B ...")
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
        x, _, gt_terms = seq_collate_fn(step_batch)
        with torch.no_grad():
            # predict with model
            out = model(x)
            logits = out.logits   
            pred_labels = torch.argmax(logits, -1)

        # build (term,aspect) couples to produce correct output for the metrics
        preds = []
        next_text = ""
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

def predict_taskC(model,  samples: List[Dict], step_size: int=32, label_tags: Dict=CATEGORY_INV, verbose=False):
    """
    Perform prediction for task C, step_size element at a time.
    """
    print("[preds]: predicting on task C ...")
    model.freeze()
    predicted = []  # List[Dict] for output

    # to correctly get the output labels
    sigmoid = torch.nn.Sigmoid()
    threshold = torch.tensor([0.5])


    # pre-processing data
    data_elems = _read_data_taskC(test=True, test_samples=samples)
    for step in range(0,len(data_elems), step_size):
        # test step_size samples at a time
        if step+step_size <= len(data_elems):
            step_batch = data_elems[step:step+step_size]
        else:
            step_batch = data_elems[step:]

        if verbose: print("batch_size:", len(step_batch))

        # use collate_fn to input step_size samples into the model
        x, _, gt_cats = cat_collate_fn(step_batch)
        with torch.no_grad():
            # predict with model
            out = model(x)
            logits = out.logits   
            logits = sigmoid(logits)
            pred_labels = (logits>threshold).float()*1
            if verbose:
                print("logits:", logits)
                print("preds: ", pred_labels)

        # build (term,aspect) couples to produce correct output for the metrics
        preds = []
        for i in range(len(gt_cats)): 
            # for each elem in collate batch
            text = x[i]
            if i != len(gt_cats)-1:
                next_text = x[i+1]
            
            if verbose:
                print("\ntext:", text)
                print(f"values: cat: {gt_cats[i]}, pred cat: {pred_labels[i]}")

            for p in range(len(pred_labels[0])):
                # for each prediction append the categories that scores 1.
                if pred_labels[i][p] == 1:   
                    preds.append((label_tags[p], "dummy-polarity"))
                    if verbose: print("[LOFFA]:", preds)

            if next_text != text or i == len(gt_cats)-1:
                # when input text changes we are dealing with another set of targets,
                # i.e. another prediction.
                if verbose: print("[CACCA]:", preds)
                predicted.append({"categories":preds})
                next_text = text
                preds = []

    print("Num predictions:", len(predicted))
    return predicted

def predict_taskD(model,  samples: List[Dict], step_size: int=32, label_tags: Dict=POLARITY_2_INV, verbose=False):
    """
    Perform prediction for task D, step_size element at a time.
    """
    print("[preds]: predicting on task D ...")
    model.freeze()
    predicted = []  # List[Dict] for output

    # pre-processing data
    data_elems = _read_data_taskD(test=True, test_samples=samples)
    for step in range(0,len(data_elems), step_size):
        # test step_size samples at a time
        if step+step_size <= len(data_elems):
            step_batch = data_elems[step:step+step_size]
        else:
            step_batch = data_elems[step:]

        if verbose: print("batch_size:", len(step_batch))

        # use collate_fn to input step_size samples into the model
        x, _, gt_cats = seq_collate_fn(step_batch)
        with torch.no_grad():
            # predict with model
            out = model(x)
            logits = out.logits   
            pred_labels = torch.argmax(logits, -1)

        # build (term,aspect) couples to produce correct output for the metrics
        preds = []
        for i in range(len(gt_cats)): 
            # for each elem in collate batch
            text = x[i][0]
            if i != len(gt_cats)-1:
                next_text = x[i+1][0]
            
            if verbose:
                print("\ntext:", text)
                print(f"values: cat: {gt_cats[i]}, pred sent: {label_tags[int(pred_labels[i])]}")

            # there is a prediction only if there is a ground truth term 
            # and the related polarity.  
            preds.append((gt_cats[i],label_tags[int(pred_labels[i])]))
            if verbose: print("[LOFFA]:", preds)

            if next_text != text or i == len(gt_cats)-1:
                # when input text changes we are dealing with another set of targets,
                # i.e. another prediction.
                if verbose: print("[CACCA]:", preds)
                predicted.append({"categories":preds})
                next_text = text
                preds = []

    print("Num predictions:", len(predicted))
    return predicted

def predict_taskCD(model, samples: List[Dict], step_size: int=32, label_tags: Dict=CATEGORY_INV, verbose=False):
    """
    Perform prediction for task C+D, step_size element at a time.
    """
    print("[preds]: predicting on task C+D ...")
    #model.freeze()
    predicted = []  # List[Dict] for output

    # to correctly get the output labels
    sigmoid = torch.nn.Sigmoid()
    threshold = torch.tensor([0.5])


    # pre-processing data
    data_elems = _read_data_taskC(test=True, test_samples=samples)
    for step in range(0,len(data_elems), step_size):
        # test step_size samples at a time
        if step+step_size <= len(data_elems):
            step_batch = data_elems[step:step+step_size]
        else:
            step_batch = data_elems[step:]

        if verbose: print("batch_size:", len(step_batch))

        # use collate_fn to input step_size samples into the model
        x_C, _, _ = cat_collate_fn(step_batch)
        with torch.no_grad():
            # predict with model
            for i in range(len(x_C)): 
                out_C = model.C_model(x_C[i])

                logits_C = out_C.logits   
                logits_C = sigmoid(logits_C)
                pred_cats = (logits_C>threshold).float()*1

                if verbose:
                    print("logits:", logits_C)
                    print("preds: ", pred_cats)

                # build (term,aspect) couples to produce correct output for the metrics
                preds = []
                #for i in range(len(x_C)): 
                
                text = x_C[i]
                #print(text)
                if verbose:
                    print("\ntext:", text)
                    print(f"values pred cat: {pred_cats[0]}")

                for p in range(5):  # 
                    # for each category append the categories that scores 1,
                    # with relative predicted polarity.
                    if int(pred_cats[0][p]) == 1:   
                        # execute D model to get polarity 
                        #print("lt", label_tags[p])
                        out_D = model.D_model([[text,label_tags[p]]])
                        logits_D = out_D.logits   
                        pred_sents = torch.argmax(logits_D, -1)
                        

                        preds.append((label_tags[p], POLARITY_2_INV[int(pred_sents)]))
                        if verbose:
                            print("ps:", pred_sents) 
                            print("[LOFFA]:", preds)

                # for each elem in collate batch we output a prediction
                if verbose: print("[CACCA]:", preds)
                predicted.append({"categories":preds})
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
        hparams["cls_output_dim"] = 4
        self.A_model = ABSALightningModule(test=True).load_from_checkpoint(
            checkpoint_path="model/to_docker/BERT_tA_res2res_2FFh_gelu_eps.ckpt",
            model=TaskATermExtracrionModel(hparams=hparams, tokenizer=self.tokenizer, device=device)
        )
        # load best task-B model
        hparams["cls_output_dim"] = 5
        self.B_model = ABSALightningModule(test=True).load_from_checkpoint(
            checkpoint_path="model/to_docker/BERT_tB_res2res_2FFh_gelu3_toktok_f1.ckpt",
            model=TaskBAspectSentimentModel(hparams=hparams, device=device)
        )

    def forward(self, x_A, x_B, y=None, test: bool=False):
        out_A = self.A_model(x_A)
        out_B = self.B_model(x_B)
        return out_A, out_B

    def predict(self, samples: List[Dict]):
        return predict_taskAB(self, samples=samples, tokenizer=self.tokenizer)

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

class TaskCDModel(nn.Module):
    def __init__(self, hparams: dict, device: str="cpu"):
        super().__init__()
        self.device  = device
        self.hparams = hparams
        print_hparams(hparams)

        # load best task-C model
        hparams["cls_output_dim"] = 5
        self.C_model = ABSALightningModule(test=True).load_from_checkpoint(
            checkpoint_path="model/to_docker/RoBERTa_tC_res2res_2FFh_gelu.ckpt",
            model=TaskCCategoryExtractionModel(hparams=hparams, device=device)
        )
        # load best task-D model
        hparams["cls_output_dim"] = 4
        self.D_model = ABSALightningModule(test=True).load_from_checkpoint(
            checkpoint_path="model/to_docker/version_5_lr16_drop06_710.ckpt",
            model=TaskDCategorySentimentModel(hparams=hparams, device=device)
        )

    def forward(self, x_C, x_D, y=None, test: bool=False):
        out_C = self.C_model(x_C)
        out_D = self.D_model(x_D)
        return out_C, out_D

    def predict(self, samples: List[Dict]):
        return predict_taskCD(self, samples=samples)



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
        self.freeze()
        return self.model.predict(samples)



############################
#labels = y.view(-1).long()
#loss = self.loss_function(logits, labels)
#labels = y.view(-1).long()
#sample_loss = self.loss_function(logits, labels)