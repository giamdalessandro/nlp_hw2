"""
class TaskATransformerModel(nn.Module):
    # we provide the hyperparameters as input
    def __init__(self, hparams: dict):
        super().__init__()
        print_hparams(hparams)
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(hparams["dropout"])

        #self.tokenizer = DistilBertTokenizer.from_pretrained(
        #    "distilbert-base-cased")
        self.tokenizer = BertTokenizer.from_pretrained("ykacer/bert-base-cased-imdb-sequence-classification")
        self.transfModel = BertForTokenClassification.from_pretrained(
            "bert-base-cased", 
            num_labels=5)

        # Recurrent layer
        #self.lstm = nn.LSTM(
        #    input_size=hparams["embedding_dim"], 
        #    hidden_size=hparams["lstm_dim"], 
        #    bidirectional=hparams["bidirectional"],
        #    num_layers=hparams["rnn_layers"], 
        #    dropout=hparams["dropout"] if hparams["rnn_layers"] > 1 else 0,
        #    batch_first=True
        #)

        # classifier head
        #lstm_output_dim = hparams["lstm_dim"] if hparams["bidirectional"] is False else hparams["lstm_dim"]*2
        #self.hidden = nn.Linear(lstm_output_dim, hparams["hidden_dim"])
        #self.output = nn.Linear(hparams["hidden_dim"], hparams["output_dim"])
    
    def forward(self, x, test: bool=False):
        # x -> (raw_sentence,tokenized_targets)
        tokens = self.tokenizer(x, return_tensors='pt', padding=True, truncation=True)
        for k, v in tokens.items():
            if not test:   
                tokens[k] = v.cuda()

        transf_out = self.transfModel(**tokens)
        #transf_out = self.dropout(transf_out.last_hidden_state)
        #o, (h, c) = self.lstm(transf_out.last_hidden_state)
        #o = self.dropout(o)

        #hidden = self.hidden(o)
        #output = self.output(hidden)
        return transf_out #output
"""
