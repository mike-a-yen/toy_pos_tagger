import pickle

BOS = "<BOS>"
EOS = "<EOS>"
UNK = "<UNK>"
PAD = "<PAD>"

class Vocabulary(object):
    def __init__(self):
        self.token_to_idx = dict()
        self.idx_to_token = dict()
        self.add_token(BOS)
        self.add_token(EOS)
        self.add_token(UNK)
        self.add_token(PAD)

    def add_token(self,token):
        if token not in self.token_to_idx:
            idx = len(self.token_to_idx)
            self.token_to_idx[token] = idx
            self.idx_to_token[idx] = token
            return 1
        else:
            return 0

    @property
    def size(self):
        return len(self.token_to_idx)

    def __getitem__(self,token):
        return self.token_to_idx.get(token,self.token_to_idx[UNK])

    def save(self,path):
        with open(path,'wb') as fw:
            pickle.dump(self,fw)

    @classmethod
    def load(cls,path):
        with open(path,'rb') as fo:
            return pickle.load(fo)

class LabelVocabulary(Vocabulary):

    def __init__(self):
        self.token_to_idx = dict()
        self.idx_to_token = dict()
        self.add_token(UNK)

    def __getitem__(self,token):
        return self.token_to_idx[token]
