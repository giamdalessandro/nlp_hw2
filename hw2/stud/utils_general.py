import torch


### utils functions
def print_hparams(hparam: dict):
    print("\n[model]: hyperparameters ...")
    for k, v in hparam.items():
        print(f"{k}:\t{v}")
    print()

def clean_tokens_RoBERTa(pred_tokens: str):
    """
    Remove special characters from predicted tokens to correctly analyze model performances.
    """
    clean = ""
    for pt in pred_tokens:
        if pt.startswith("Ġ"):
            # "Ġ" is RoBERTa special character for space (" ")    
            clean += " " + pt[1:]
        else:
            clean += pt

    res = clean.split(" ")
    return res

def clean_tokens_BERT(pred_tokens: str):
    """
    Remove special characters from predicted tokens to correctly analyze model performances.
    """
    clean = ""
    for i in range(len(pred_tokens)):
        pt = pred_tokens[i]
        if pt.startswith("##"):
            # "##" is a BERT special character    
            clean += pt[2:]
        elif i != 0 and (pt in ["-","'","_"] or pred_tokens[i-1] in ["-","'","_"]):
            clean += pt
        else:
            clean += " " + pt

    res = clean #.split(" ")
    return res[1:]

def get_preds_terms(preds, tokens, roberta: bool=False, verbose: bool=False):
    """
    Extract predicted aspect terms from predicted tags sequence (batch-wise).
    """
    if verbose:
        print("\npreds:",preds.size())
        print("tokens:", len(tokens))

    pred_terms = []
    sent_terms = []
    for b in range(len(preds)):
        if verbose: print("preds:", preds[b])
        preds = []
        term = []

        inside = False
        for p in range(len(tokens[b])): # len(tokens)
            if not inside:
                if (preds[b][p] != 0 and preds[b][p] != 3):
                    term.append(tokens[b][p])
                    inside = True
            else:
                if (preds[b][p] != 0 and preds[b][p] != 3):
                    term.append(tokens[b][p])
                else:
                    if tokens[b][p].startswith("##"):
                        term.append(tokens[b][p])

                    ff = clean_tokens_BERT(term)
                    pred_terms.append(ff)
                    preds.append(ff)
                    inside = False
                    term = []

        sent_terms.append(preds)    

    return pred_terms, sent_terms

def get_label_tokens(targets: dict, tokenizer):
    """
    Commento sbagliato come un negroni, ma senza negroni.
    """
    for tgt in targets:
        if len(tgt[1]) > 0:
            tokenizer.tokenize(tgt[1])

    return

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
