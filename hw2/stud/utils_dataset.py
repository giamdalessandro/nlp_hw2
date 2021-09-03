import os
import re
import json
import collections
from nltk import tag
from nltk.util import pr

import torch
from torch.utils import data
import torchtext
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torchtext import vocab
from torchtext.vocab import Vocab
from transformers import BertTokenizer, DistilBertTokenizer

LAPTOP_TRAIN     = "data/laptops_train.json"
LAPTOP_DEV       = "data/laptops_dev.json"
RESTAURANT_TRAIN = "data/restaurants_train.json"
RESTAURANT_DEV   = "data/restaurants_dev.json"

BIO_TAGS = {
    "pad": 0,
    "B"  : 1,
    "I"  : 2,
    "O"  : 3
}

CATEGORY_TAGS = {
    "anecdotes/miscellaneous" : 0,
    "price"                   : 1,
    "food"                    : 2,
    "ambience"                : 3,
    "service"                 : 4
}

POLARITY_TAGS = {
    "un-polarized" : 0,
    "positive"     : 1,
    "negative"     : 2,
    "neutral"      : 3,
    "conflict"     : 4,
}

POLARITY_2_TAGS = {
    "positive"     : 0,
    "negative"     : 1,
    "neutral"      : 2,
    "conflict"     : 3
}

### utils
def read_json_data(data_path: str):
    """ Load dataset from JSON file to Dict."""
    f = open(data_path, "r")
    return json.load(f)

def load_pretrained_embeddings(vocabulary: dict, max_size: int):
    """
    Loads pretrained word embeddings.
    """
    # get GloVe 6B pre-trained word embeddings, of dimension 100
    glove_vec = torchtext.vocab.GloVe(name="6B", dim=100, unk_init=torch.Tensor.normal_)

    pretrained = []
    for k, _ in vocabulary.stoi.items():
        if k == "<PAD>":
            emb = torch.zeros([glove_vec.dim])
        elif k == "<UNK>":
            emb = torch.rand([glove_vec.dim])
        else:
            emb = glove_vec.get_vecs_by_tokens(k, lower_case_backup=True)
        pretrained.append(emb) 

    # return a tensor of size [vocab_size, emb_dim]
    return torch.stack(pretrained, dim=0)

def _read_data_taskA(data_path: str, tokenizer, 
        bert: bool=False, 
        mode: str="tokenize", 
        tagger=None, 
        test: bool=False, 
        test_samples=None
    ):
    """
    Reads the dataset and analyze words and targets frequencies.
    """
    print(f"\n[dataset]: Loading data from '{data_path}'...")
    sentences = []
    labels    = []
    tok_list  = []
    words_list   = []
    targets_list = []
    target_final = []

    data_dict = read_json_data(data_path)# if not test else test_samples
    for entry in data_dict:
        t_list = []
        # tokenize data sentences
        if bert:
            tokens = tokenizer.tokenize(entry["text"])
            tokens.insert(0, "[CLS]")  # RoBERTa "<s>" <-> BERT "[CLS]" 
            tokens.append("[SEP]")     # RoBERTa "</s>" <-> BERT "[SEP]"
        else:
            tokens = tokenizer(entry["text"])
        words_list.extend(tokens)
        tok_list.append(tokens)

        # count target words
        targets = entry["targets"]
        tgt_list = []
        if len(targets) > 0:
            t_list.append(targets)
            for tgt in targets:
                targets_list.append(tgt[1])
                tgt_list.append(tgt[1])
        else:
            t_list.append([])

        # tag input tokens
        b_tok = tokenizer if bert else None
        tags = tagger(targets, tokens, bert_tokenizer=b_tok)
        #print(tags)
        labels.append(tags)
        target_final.append(tgt_list)
        
        if mode == "tokenize":
            sentences.append(tokens)
        elif mode == "raw":
            sentences.append(entry["text"])
            

    assert len(sentences) == len(labels)
    print("sentences:",len(sentences))
    print("labels:",len(labels))

    # count words occurency and frequency            
    word_counter = collections.Counter(words_list)
    distinct_words = len(word_counter)
    print(f"Number of distinct words: {distinct_words}")
    
    # count target words occurency and frequency
    tgts_counter = collections.Counter(targets_list)
    distinct_tgts = len(tgts_counter)
    print(f"Number of distinct targets: {distinct_tgts}")

    if not test:
        return sentences, labels, targets_list, word_counter
    else:
        return sentences, labels, target_final, tok_list

def _read_data_taskB(data_path: str="path", test: bool=False, test_samples=None):
    """
    Reads the dataset and analyze words and targets frequencies.
    """
    sentences = []
    labels    = []
    targets_list = []

    data_dict = read_json_data(data_path) if not test else test_samples
    for entry in data_dict:
        text    = entry["text"]
        targets = entry["targets"]

        sent_term  = []
        pol_labels = []
        term_list  = []
        if len(targets) > 0:
            for tgt in targets:
                term = tgt[1]
                sent_term.append([text,term])
                term_list.append(term)

                polarity = "un-polarized" if test else tgt[2]
                pol_labels.append(POLARITY_TAGS[polarity])

        else:
            polarity = "un-polarized"
            sent_term.append([text,""])
            pol_labels.append(POLARITY_TAGS[polarity])
            term_list.append("")

        sentences.append(sent_term)
        labels.append(pol_labels)
        targets_list.append(term_list)

    assert len(sentences) == len(labels)
    if not test:
        return sentences, labels, targets_list, None
    else:
        return list(zip(sentences,labels,targets_list))

def _read_data_taskC(data_path: str="path", test: bool=False, test_samples=None, cat2id: dict=CATEGORY_TAGS):
    """
    Reads the dataset and analyze words and targets frequencies.
    """
    sentences = []
    labels    = []
    targets_list = []

    data_dict = read_json_data(data_path) if not test else test_samples
    for entry in data_dict:
        text = entry["text"]
        categories = entry["categories"]

        # sent_cat = []
        cats_list  = []
        vec_y = [0 for i in range(len(cat2id))]
        for cat in categories:
            category = cat[0]
            vec_y[cat2id[category]] = 1
            cats_list.append(category)

        #for cat in cat2id.keys():
        #    sent_cat.append([text,cat])
        sentences.append(text)
        labels.append(vec_y)
        targets_list.append(cats_list)

    assert len(sentences) == len(labels)
    if not test:
        return sentences, labels, targets_list, None
    else:
        return list(zip(sentences,labels,targets_list))

def _read_data_taskD(data_path: str="path", test: bool=False, test_samples=None):
    """
    Reads the dataset and analyze words and targets frequencies.
    """
    sentences = []
    labels    = []
    targets_list = []

    data_dict = read_json_data(data_path) if not test else test_samples
    for entry in data_dict:
        text = entry["text"]
        categories = entry["categories"]

        sent_cats  = []
        pol_labels = []
        cats_list  = []
        for cat in categories:
            category = cat[0]
            polarity = cat[1]

            sent_cats.append([text,cat])
            pol_labels.append(POLARITY_2_TAGS[polarity])
            cats_list.append(category)

        sentences.append(sent_cats)
        labels.append(pol_labels)
        targets_list.append(cats_list)

    assert len(sentences) == len(labels)
    if not test:
        return sentences, labels, targets_list, None
    else:
        return list(zip(sentences,labels,targets_list))


class ABSADataset(Dataset):
    """
    Override of torch base Dataset class to proerly load ABSA restaurants or laptop data.
    """
    def __init__(self, 
            data_path : str=LAPTOP_TRAIN,
            unk_token : str="<UNK>", 
            pad_token : str="<PAD>",
            mode      : str="tokenize",
            task      : str="A",
            test      : bool=False,
            tokenizer=None,
            vocab=None
        ):
        self.data_path = data_path
        self.test = test
        self.task = task
        self.bert_tokenizer = tokenizer
        self._build_vocab(data_path, unk_token=unk_token, pad_token=pad_token, mode=mode, vocab=vocab, test=test)

    def _tokenize_line(self, line: str, pattern='\W'):
        """
        Tokenizes a single line splitting on regex `\W`, i.e. non-word characters 
        (e.g. "The pen is on the table" -> ["the, "pen", "is", "on", "the", "table"]).
        """
        # TODO check nltk tokenize
        # TODO check string not to lower
        line = re.sub("[.,;:]", " ", line)
        return re.split(pattern, line.lower())

    def _tag_tokens(self, targets: list, tokens: list, tags: dict=BIO_TAGS, bert_tokenizer=None, verbose: bool=False):
        """
        Matches each token of the input text to the corresponding BILOU tag, 
        and returns the tags vector/list.
        """
        if bert_tokenizer is not None:
            tokenizer = bert_tokenizer

        if len(targets) > 0:
            tags_list = []
            for tgt in targets:
                t_list = []
                inside = False
                found  = False
                if bert_tokenizer is not None:
                    tgt_terms = tokenizer.tokenize(tgt[1]) 
                else:
                    tgt_terms = self._tokenize_line(tgt[1])

                if verbose:
                    print(tgt_terms)

                for i in range(len(tokens)):
                    if tokens[i] == tgt_terms[0] and not found: 
                        # token is the beginning (B) of target terms sequence
                        t_list.append(tags["B"])
                        if len(tgt_terms) > 1 and tokens[i:i+len(tgt_terms)] == tgt_terms:
                            # check if the matching token is not a repetition of the term
                            # and is the actual target term, if so the correct sequence is found 
                            inside = True
                            found  = True

                    elif inside == True:
                        # multi words terms
                        if tokens[i] in tgt_terms[1:-1] and len(tgt_terms) > 2:
                            # token is inside (I) the target terms sequence
                            t_list.append(tags["I"])

                        elif tokens[i] == tgt_terms[-1]:
                            # token is the last (L) target term
                            t_list.append(tags["I"]) # tags["L"] 
                            inside = False

                        # when the last tgt_word is repeated inside the tgt_terms 
                        inside = False

                    else:
                        # token is outside (O) the target terms sequence
                        t_list.append(tags["O"])

                tags_list.append(torch.Tensor(t_list))

            # merge tags
            tags_tensor = torch.stack(tags_list)
            res = torch.min(tags_tensor, dim=0)
            if verbose:
                print("targets:", targets)
                print("tokens:", tokens, "-- len:", len(tokens))
                print("tags:", tags_list)
                #print("tags:", tags_tensor.size())
                #print("res:", res.values.size())
            
            return res.values

        else:
            return [tags["O"] for t in tokens]

    def _read_data(self, 
        data_path : str, 
        bert : bool=False, 
        mode : str="tokenize", 
        task : str="A"
        ):
        """
        Reads the dataset and analyze words and targets frequencies.

        Args:
        - `mode` : whether to tokenize ("tokenize") or get the raw ("raw") input text.
        - `task` : preprocess data to perform the particular task {"A", "B", "C" or "D"}.
        """
        print(f"\n[dataset]: Loading data from '{data_path}'...")
        print(f"[dataset]: performing task '{task}' preprocessing ...")
        if task == "A":
            tokenizer = self._tokenize_line if mode == "tokenize" else self.bert_tokenizer
            return _read_data_taskA(data_path, tokenizer, bert, mode, tagger=self._tag_tokens, test=self.test)

        elif task == "B":
            return _read_data_taskB(data_path, test=False)
    
        elif task == "C":
            return _read_data_taskC(data_path, test=False)

        elif task == "D":
            return _read_data_taskD(data_path, test=False)

    def _build_vocab(self, 
            data_path : str,
            vocab_size: int=3500, 
            unk_token : str="<UNK>", 
            pad_token : str="<PAD>",
            mode : str="tokenize",
            test : bool=False,
            vocab=None
        ):
        """
        Builds a torchtext vocabulary over read data. It adds the following 
        attributes to the class: 
            - self.distinct_words   # number of distinct words
            - self.distinct_tgts    # number of distinct targets words
            - self.vocabulary       # torchtext.Vocab vocabulary over data
    
        Args:
            - `vocab_size`: size of the vocabolary;
            - `unk_token` : token to associate with unknown words;
            - `pad_token` : token to indicate padding;
        """       
        # read data form file
        sentences, labels, targets_list, word_counter = self._read_data(data_path, mode=mode, bert=True, task=self.task)

        # build vocabulary on data if none is given
        if vocab is None:
            print("\n[dataset]: building vocabulary ...")
            # load pretrained GloVe word embeddings
            glove_vec = torchtext.vocab.GloVe(name="6B", dim=100, unk_init=torch.FloatTensor.normal_)
            self.vocabulary = Vocab(
                counter=word_counter,             # (word,freq) mapping
                max_size=vocab_size,              # vocabulary max size
                specials=[pad_token,unk_token],   # special tokens
                vectors=glove_vec                 # pre-trained embeddings
            )
            # ensure pad_token embedding is a zeros tensor
            self.vocabulary.vectors[0] = torch.zeros([glove_vec.dim]).float()
            print("Embedding vectors:", self.vocabulary.vectors.size())

        else:
            print("\n[dataset]: (dev) using train vocabulary ...")
            self.vocabulary = vocab

        # create data samples -> (x, y)
        self.samples = []

        if mode == "tokenize":
            for toks, tags, terms in zip(sentences,labels,targets_list):
                tokens_idxs = []
                for t in toks:
                    try:
                        idx = self.vocabulary.stoi[t]
                    except:
                        idx = self.vocabulary.stoi[unk_token]

                    assert len(toks) == len(tags)
                    tokens_idxs.append(idx)

                #print(toks, tags)
                self.samples.append((tokens_idxs,tags,toks,self._tokenize_line(terms)))

        elif mode == "raw":
            # use raw text as input (required by transformers)
            if not test:
                for s, l, tgt in zip(sentences,labels,targets_list):
                    self.samples.append((s,l,tgt))
            else:
                for s, l, tgt, tok in zip(sentences,labels,targets_list, word_counter):
                    self.samples.append((s,l,tgt,tok))
        return

    def __len__(self):
        # returns the number of samples in our dataset
      return len(self.samples)

    def __getitem__(self, idx):
        # returns the idx-th sample
        return self.samples[idx]


class ABSADataModule(pl.LightningDataModule):
    """ TODO
    Override of pl.LightningDataModule class to easly handle the ABSADataset for training and evaluation.
    """
    def __init__(self, 
            train_path : str=LAPTOP_TRAIN, 
            dev_path   : str=LAPTOP_DEV,
            batch_size : int=32,
            in_mode    : str="raw",
            task       : str="A",
            test       : bool=False,
            collate_fn=None,
            tokenizer=None
        ):
        super().__init__()
        self.train_path = train_path
        self.dev_path   = dev_path
        self.batch_size = batch_size
        self.in_mode    = in_mode
        self.task       = task
        self.collate_fn = collate_fn
        self.tokenizer  = tokenizer

        if not test:
            self.setup()
        else:
            self.test_setup()

    def setup(self):
        """
        Initialize train and eval datasets from training
        """
        # TODO check if need both dataset together
        self.train_dataset = ABSADataset(data_path=self.train_path, mode=self.in_mode, task=self.task, 
                                        tokenizer=self.tokenizer, vocab="bert")
        self.vocabulary = self.train_dataset.vocabulary

        self.eval_dataset = ABSADataset(data_path=self.dev_path, mode=self.in_mode, task=self.task,
                                        tokenizer=self.tokenizer, vocab=self.vocabulary)
        #self.train_restaurant = ABSADataset(data_path=RESTAURANT_TRAIN)
        #self.eval_restaurant  = ABSADataset(data_path=RESTAURANT_DEV)

    def test_setup(self, test_data: list=None):
        """
        Initialize test data for testing.
        """
        print("[dataset]: using test setup ...")
        self.vocabulary = ["empty"]
        self.eval_dataset = ABSADataset(data_path=self.dev_path, mode=self.in_mode, task=self.task,
                                        tokenizer=self.tokenizer, vocab="bert", test=True)
        return

    def train_dataloader(self, *args, **kwargs):
        #train_dataset = self.train_restaurant if self.mode == "restaurants" else self.train_laptop
        return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn, *args)

    def eval_dataloader(self, *args, **kwargs):
        #eval_dataset = self.eval_restaurant if self.mode == "restaurants" else self.eval_laptop
        return DataLoader(self.eval_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn, *args)

