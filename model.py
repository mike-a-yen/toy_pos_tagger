import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderRNN(nn.Module):
    def __init__(self,emb_size,rnn_size,rnn_layers,dropout=0.0,vocab_size=2500,bidirectional=True):
        nn.Module.__init__(self)
        self.drop = nn.Dropout(dropout)
        self.embedding = nn.Embedding(vocab_size,emb_size)
        self.rnn = nn.LSTM(emb_size,rnn_size,rnn_layers,dropout,bidirectional=bidirectional)

        self.emb_size = emb_size
        self.rnn_size = rnn_size
        self.rnn_layers = rnn_layers
        self.vocab_size = vocab_size
        self.bidirectional = bidirectional

    def forward(self,word_ids,hidden):
        embed = self.embedding(word_ids)
        encoded,hidden = self.rnn(self.drop(embed))
        return encoded,hidden

    def init_hidden(self,batch_size):
        bi = 1+self.bidirectional
        h_state = torch.zeros(bi*self.rnn_layers,batch_size,self.rnn_size)
        c_state = torch.zeros(bi*self.rnn_layers,batch_size,self.rnn_size)
        return h_state.to(device),c_state.to(device)

class POSTagger(nn.Module):
    def __init__(self,out_size,emb_size,rnn_size,rnn_layers,dropout=0.0,vocab_size=2500,bidirectional=True):
        nn.Module.__init__(self)
        self.encoder = EncoderRNN(emb_size,rnn_size,rnn_layers,dropout,vocab_size,bidirectional)
        bi = 1+bidirectional
        self.decoder = nn.Linear(bi*rnn_size,out_size)
        self.out_size = out_size
        self.output_activation = nn.LogSoftmax(dim=2)

    def forward(self,word_ids,hidden):
        encoded,hidden = self.encoder(word_ids,hidden)
        logits = self.decoder(encoded)
        log_probs = self.output_activation(logits)
        return log_probs,hidden
