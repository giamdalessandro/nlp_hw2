# -------------------------------------------------------- #
#  To better test my models I have implemented (locally) the same 
#  evaluation functions provided in the docker test. 
# ---------------------------------------------------------#
import pytorch_lightning as pl

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score

POLARITY_INV = {
    0 : "positive",
    1 : "negative",
    2 : "neutral ",
    3 : "conflict",
	4 : "un-polarized"
}
#    3 : "L",
IDX2LABEL = {
    0 : "pad",
    1 : "B",
    2 : "I",
    3 : "O"
}


def precision_metrics(model: pl.LightningModule, l_dataset: DataLoader, l_label_vocab):
    model.freeze()
    all_predictions = []
    all_labels = []
    for indexed_elem in l_dataset:
        #indexed_in, _, indexed_labels, _, _ = indexed_elem
        indexed_in, indexed_labels = indexed_elem
        outputs = model(indexed_in)
        predictions = torch.argmax(outputs.logits, -1).view(-1)
        labels = indexed_labels.view(-1)
        valid_indices = labels != 9
        
        valid_predictions = predictions[valid_indices]
        valid_labels = labels[valid_indices]
        
        all_predictions.extend(valid_predictions.tolist())
        all_labels.extend(valid_labels.tolist())

    # global precision. Does take class imbalance into account.
    micro_precision = precision_score(all_labels, all_predictions, average="micro", zero_division=0)
    # precision per class and arithmetic average of them. Does not take into account class imbalance.
    macro_precision = precision_score(all_labels, all_predictions, average="macro", zero_division=0)
    per_class_precision = precision_score(all_labels, all_predictions, labels=list(range(len(l_label_vocab))),
                                         average=None, zero_division=0)
    model.unfreeze()
    return {"micro_precision":micro_precision,
            "macro_precision":macro_precision, 
            "per_class_precision":per_class_precision}

def evaluate_precision(precisions: dict, label_d: dict=POLARITY_INV):
    per_class_precision = precisions["per_class_precision"]
    print(f"Micro Precision: {precisions['micro_precision']}")
    print(f"Macro Precision: {precisions['macro_precision']}")

    print("Per class Precision:")
    print("\tlabel\tscore")
    for idx_class, precision in sorted(enumerate(per_class_precision), key=lambda elem: -elem[1]):
        label = label_d[idx_class]
        print(f"\t{label}\t{precision:.4f}")

    return

def evaluate_extraction(model: pl.LightningModule, l_dataset: DataLoader):
    model.freeze()
    scores = {"tp": 0, "fp": 0, "fn": 0}
    for elem in l_dataset:
        #inputs, _, labels, tokens, l_terms = elempreds
        inputs, labels = elem
        outs = model(inputs)

        preds = torch.argmax(outs.logits, -1).view(-1)
        t_preds = get_preds_terms(preds, tokens)
        #print(t_preds)
        ll = []
        for b in l_terms:
            for l in b:
                ll.append(l)

        pred_terms  = {i for i in t_preds}
        label_terms = {t for t in ll}

        scores["tp"] += len(pred_terms & label_terms)
        scores["fp"] += len(pred_terms - label_terms)
        scores["fn"] += len(label_terms - pred_terms)

    precision = 100*scores["tp"] / (scores["tp"] + scores["fp"])
    recall = 100*scores["tp"] / (scores["tp"] + scores["fn"])
    f1 = 2 * precision*recall / (precision+recall)

    print(f"Aspect Extraction Evaluation")
    print(f"\tAspects\t TP: {scores['tp']};\tFP: {scores['fp']};\tFN: {scores['fn']}")
    print(f"\t\tprecision: {precision:.2f};\trecall: {recall:.2f};\tf1: {f1:.2f}")

def evaluate_sentiment(samples, predictions_b, mode="Aspect Sentiment"):
    scores = {}
    if mode == 'Category Extraction':
        sentiment_types = ["anecdotes/miscellaneous", "price", "food", "ambience", "service"]
    else:
        sentiment_types = ["positive", "negative", "neutral", "conflict"]

    scores = {sent: {"tp": 0, "fp": 0, "fn": 0} for sent in sentiment_types + ["ALL"]}
    for label, pred in zip(samples, predictions_b):
        for sentiment in sentiment_types:
            if mode == "Aspect Sentiment":
                pred_sent = {(term_pred[0], term_pred[1]) for term_pred in pred["targets"] if term_pred[1] == sentiment}
                gt_sent = {(term_pred[1], term_pred[2]) for term_pred in label["targets"] if term_pred[2] == sentiment}

            elif mode == 'Category Extraction' and "categories" in label:
                pred_sent = {(term_pred[0]) for term_pred in pred["categories"] if term_pred[0] == sentiment}
                gt_sent = {(term_pred[0]) for term_pred in label["categories"] if term_pred[0] == sentiment}

            elif "categories" in label:
                pred_sent = {(term_pred[0], term_pred[1]) for term_pred in pred["categories"] if term_pred[1] == sentiment}
                gt_sent = {(term_pred[0], term_pred[1]) for term_pred in label["categories"] if term_pred[1] == sentiment}

            else:
                continue

            scores[sentiment]["tp"] += len(pred_sent & gt_sent)
            scores[sentiment]["fp"] += len(pred_sent - gt_sent)
            scores[sentiment]["fn"] += len(gt_sent - pred_sent)

    # Compute per sentiment Precision / Recall / F1
    for sent_type in scores.keys():
        if scores[sent_type]["tp"]:
            scores[sent_type]["p"] = 100 * scores[sent_type]["tp"] / (scores[sent_type]["fp"] + scores[sent_type]["tp"])
            scores[sent_type]["r"] = 100 * scores[sent_type]["tp"] / (scores[sent_type]["fn"] + scores[sent_type]["tp"])
        else:
            scores[sent_type]["p"], scores[sent_type]["r"] = 0, 0

        if not scores[sent_type]["p"] + scores[sent_type]["r"] == 0:
            scores[sent_type]["f1"] = 2 * scores[sent_type]["p"] * scores[sent_type]["r"] / (scores[sent_type]["p"] + scores[sent_type]["r"])
        else:
            scores[sent_type]["f1"] = 0

    # Compute micro F1 Scores
    tp = sum([scores[sent_type]["tp"] for sent_type in sentiment_types])
    fp = sum([scores[sent_type]["fp"] for sent_type in sentiment_types])
    fn = sum([scores[sent_type]["fn"] for sent_type in sentiment_types])

    if tp:
        precision = 100 * tp / (tp + fp)
        recall = 100 * tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)

    else:
        precision, recall, f1 = 0, 0, 0

    scores["ALL"]["p"] = precision
    scores["ALL"]["r"] = recall
    scores["ALL"]["f1"] = f1
    scores["ALL"]["tp"] = tp
    scores["ALL"]["fp"] = fp
    scores["ALL"]["fn"] = fn

    # Compute Macro F1 Scores
    scores["ALL"]["Macro_f1"] = sum([scores[ent_type]["f1"] for ent_type in sentiment_types])/len(sentiment_types)
    scores["ALL"]["Macro_p"]  = sum([scores[ent_type]["p"] for ent_type in sentiment_types])/len(sentiment_types)
    scores["ALL"]["Macro_r"]  = sum([scores[ent_type]["r"] for ent_type in sentiment_types])/len(sentiment_types)

    print(f"{mode} Evaluation\n")

    print(f"\tALL\t TP: {scores['ALL']['tp']};\tFP: {scores['ALL']['fp']};\tFN: {scores['ALL']['fn']}")
    print(f"\t\t(m avg): precision: {precision:.2f};\trecall: {recall:.2f};\tf1: {f1:.2f} (micro)")
    print(f"\t\t(M avg): precision: {scores['ALL']['Macro_p']:.2f};\trecall: {scores['ALL']['Macro_r']:.2f};\tf1: {scores['ALL']['Macro_f1']:.2f} (Macro)\n")

    for sent_type in sentiment_types:
        print("\t{}: \tTP: {};\tFP: {};\tFN: {};\tprecision: {:.2f};\trecall: {:.2f};\tf1: {:.2f};\t{}".format(
            sent_type,
            scores[sent_type]["tp"],
            scores[sent_type]["fp"],
            scores[sent_type]["fn"],
            scores[sent_type]["p"],
            scores[sent_type]["r"],
            scores[sent_type]["f1"],
            scores[sent_type]["tp"] +
            scores[sent_type][
                "fp"]))

    return scores, precision, recall, f1