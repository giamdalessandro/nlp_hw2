import os
import re
import json
import collections
from nltk import tag
from nltk.util import pr

import torch
import torchtext
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torchtext import vocab
from torchtext.vocab import Vocab
from transformers import BertTokenizer

LAPTOP_TRAIN     = "data/laptops_train.json"
LAPTOP_DEV       = "data/laptops_dev.json"
RESTAURANT_TRAIN = "data/restaurants_train.json"
RESTAURANT_DEV   = "data/restaurants_dev.json"

BIO_TAGS = {
    "pad": 0,
    "B"  : 1,
    "I"  : 2,
    "L"  : 3,
    "O"  : 4
}

IDX2LABEL = {
    0 : "pad",
    1 : "B",
    2 : "I",
    3 : "L",
    4 : "O"
}


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


class ABSADataset(Dataset):
    """
    Override of torch base Dataset class to proerly load ABSA restaurants or laptop data.
    """
    def __init__(self, 
            data_path : str=LAPTOP_TRAIN,
            unk_token : str="<UNK>", 
            pad_token : str="<PAD>",
            mode : str="tokenize",
            vocab=None
        ):
        self.data_path = data_path
        self._build_vocab(data_path, unk_token=unk_token, pad_token=pad_token, mode=mode, vocab=vocab)

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
                    tgt_terms = tokenizer.tokenize(tgt[1], add_special_tokens=True) 
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

    def _read_data(self, data_path : str, mode: str="tokenize", bert: bool=False):
        """
        Reads the dataset and analyze words and targets frequencies.
        Args:
        - `mode` : whether to tokenize ("tokenize") or get the raw ("raw") input text.
        """
        print(f"\n[dataset]: Loading data from '{data_path}'...")
        sentences = []
        labels    = []
        words_list   = []
        targets_list = []
        tokenizer = None
        if bert:
            tokenizer = BertTokenizer.from_pretrained(
                "ykacer/bert-base-cased-imdb-sequence-classification")

        with open(data_path, "r") as f:
            json_data = json.load(f)
            for entry in json_data:
                t_list = []
                # tokenize data sentences
                if bert:
                    tokens = tokenizer.tokenize(entry["text"])
                    tokens.insert(0, "[CLS]")
                    tokens.append("[SEP]")
                else:
                    tokens = self._tokenize_line(entry["text"])
                words_list.extend(tokens)

                # count target words
                targets = entry["targets"]
                if len(targets) > 0:
                    t_list.append(targets)
                    for tgt in targets:
                        targets_list.append(tgt[1])
                else:
                    t_list.append([])

                # tag input tokens
                tags = self._tag_tokens(targets, tokens, bert_tokenizer=tokenizer)
                labels.append(tags)
                
                if mode == "tokenize":
                    sentences.append(tokens)
                elif mode == "raw":
                    sentences.append(entry["text"])
                

        assert len(sentences) == len(labels)
        print("sentences:",len(sentences))
        print("labels:",len(labels))

        # count words occurency and frequency            
        word_counter = collections.Counter(words_list)
        self.distinct_words = len(word_counter)
        print(f"Number of distinct words: {len(word_counter)}")
        
        # count target words occurency and frequency
        tgts_counter = collections.Counter(targets_list)
        self.distinct_tgts = len(tgts_counter)
        print(f"Number of distinct targets: {len(tgts_counter)}")

        return sentences, labels, targets_list, word_counter
    
    def _build_vocab(self, 
            data_path : str,
            vocab_size: int=3500, 
            unk_token : str="<UNK>", 
            pad_token : str="<PAD>",
            mode : str="tokenize",
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
        sentences, labels, targets_list, word_counter = self._read_data(data_path, mode=mode, bert=True)

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
            for s, l in zip(sentences,labels):
                self.samples.append((s,l))
        
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
            train_path: str=LAPTOP_TRAIN, 
            dev_path  : str=LAPTOP_DEV,
            batch_size: int=32,
            in_mode : str="raw",
            test : bool=False,
            collate_fn=None
        ):
        super().__init__()
        self.train_path = train_path
        self.dev_path   = dev_path
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.in_mode = in_mode

        if not test:
            self.setup()
        else:
            self.test_setup()

    def setup(self):
        """
        Initialize train and eval datasets from training
        """
        # TODO check if need both dataset together
        self.train_dataset = ABSADataset(data_path=self.train_path, mode=self.in_mode, vocab="bert")
        self.vocabulary = self.train_dataset.vocabulary

        self.eval_dataset = ABSADataset(data_path=self.dev_path, mode=self.in_mode, vocab=self.vocabulary)
        #self.train_restaurant = ABSADataset(data_path=RESTAURANT_TRAIN)
        #self.eval_restaurant  = ABSADataset(data_path=RESTAURANT_DEV)

    def test_setup(self, test_data: list):
        """
        Initialize test data for testing.
        """
        test_dataset = None
        return

    def train_dataloader(self, *args, **kwargs):
        #train_dataset = self.train_restaurant if self.mode == "restaurants" else self.train_laptop
        return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)

    def eval_dataloader(self, *args, **kwargs):
        #eval_dataset = self.eval_restaurant if self.mode == "restaurants" else self.eval_laptop
        return DataLoader(self.eval_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)

