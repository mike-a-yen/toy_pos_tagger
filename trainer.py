import json

import torch
import torch.nn as nn

from model import POSTagger
from vocabulary import Vocabulary, LabelVocabulary, BOS, EOS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(path,batch_size):
    with open(path,'r') as fo:
        long_sent = json.load(fo)
        tokens = []
        tags = []
        for tok,tag in long_sent:
            tokens.append(tok)
            tags.append(tag)
        tokens = torch.LongTensor(tokens).view(-1,1).contiguous()
        tags = torch.LongTensor(tags).view(-1,1).contiguous()

    return tokens.to(device),tags.to(device)

def detach_hidden(hidden):
    if isinstance(hidden,torch.Tensor):
        return hidden.detach()
    else:
        return tuple(detach_hidden(h) for h in hidden)

class Trainer(object):
    def __init__(self,model):
        self.model = model
        self.loss = nn.NLLLoss()
        self.opt = torch.optim.Adagrad(self.model.parameters(),0.001)

        self.bptt = 30

    def forward_batch(self,x_data,hidden):
        log_probs, hidden = self.model(x_data,hidden)
        return log_probs, hidden

    def train_batch(self,x_data,hidden,y_data):
        log_probs,hidden = self.forward_batch(x_data,hidden)
        log_probs_flat = log_probs.view(-1,self.model.out_size)
        y_flat = y_data.view(-1)
        cost = self.loss(log_probs_flat,y_flat)
        cost.backward()
        self.opt.step()
        n_samples = y_flat.size(0)

        return {"cost":cost.item(),
                "n_samples":n_samples,
                "hidden":hidden,
                "log_probs":log_probs}

    def train_epoch(self,X,Y):
        self.model.train()
        n_batches = X.size(0)//self.bptt+1
        hidden = self.model.encoder.init_hidden(1)
        epoch_loss = 0.0
        n_samples = 0
        for batch_id in range(n_batches):
            hidden = detach_hidden(hidden)
            x_batch = self.get_next_batch(X,batch_id)
            y_batch = self.get_next_batch(Y,batch_id)
            batch_results = self.train_batch(x_batch,hidden,y_batch)
            batch_cost = batch_results['cost']*batch_results['n_samples']
            n_samples += batch_results['n_samples']
            epoch_loss += batch_cost
        epoch_loss /= n_samples
        print('Epoch Train Loss: {:0.4f}'.format(epoch_loss))

    def evaluate(self,X,Y):
        self.model.eval()
        n_batches = (X.size(0)-1)//self.bptt+1
        hidden = self.model.encoder.init_hidden(1)
        eval_loss = 0.0
        n_samples = 0
        for batch_id in range(n_batches):
            hidden = detach_hidden(hidden)
            x_batch = self.get_next_batch(X,batch_id)
            y_batch = self.get_next_batch(Y,batch_id)
            log_probs,hidden = self.forward_batch(x_batch,hidden)
            log_probs_flat = log_probs.view(-1,self.model.out_size)
            y_flat = y_batch.view(-1)
            cost = self.loss(log_probs_flat,y_flat)
            batch_samples = y_flat.size(0)
            batch_cost = cost.item()*batch_samples
            n_samples += batch_samples
            eval_loss += batch_cost
            self.model.zero_grad()
        eval_loss /= n_samples
        print('Evaluated Loss: {:0.4f}'.format(eval_loss))

    def get_next_batch(self,data,batch_id):
        offset = batch_id*self.bptt
        seq_len = min(self.bptt,data.size(0)-offset)
        chunk = data[offset:offset+seq_len]
        return chunk

if __name__ == '__main__':
    import numpy as np

    train_tokens,train_tags = load_data('data/train.json',1)
    test_tokens,test_tags = load_data('data/test.json',1)
    print('Data loaded')
    vocab = Vocabulary.load('models/vocab-2408.pklb')
    label_vocab = LabelVocabulary.load('models/label_vocab.pklb')
    print('Vocabulary loaded')

    emb_size = 32
    rnn_size = 128
    rnn_layers = 1
    dropout = 0.4
    model = POSTagger(label_vocab.size,emb_size,rnn_size,rnn_layers,dropout,vocab.size+100).to(device)
    print('Model defined on',device)
    trainer = Trainer(model)
    for epoch in range(64):
        print('='*5,'epoch',epoch,'='*5)
        trainer.train_epoch(train_tokens,train_tags)
        trainer.evaluate(test_tokens,test_tags)

    model.eval()
    sentence = [BOS]+['the','lazy','brown','fox','jumped','over','the','lazy','dog','.']+[EOS]
    encoded = torch.LongTensor([vocab[t] for t in sentence]).view(-1,1).to(device)
    hidden = model.encoder.init_hidden(1)
    log_probs,hidden = model(encoded,hidden)
    log_probs = log_probs.squeeze().detach().cpu().numpy()
    probs = np.exp(log_probs)
    argmax = np.argmax(probs,axis=-1)
    solution = list(zip(sentence,argmax))
    for word,tag in solution:
        print(word,label_vocab.idx_to_token[tag])
