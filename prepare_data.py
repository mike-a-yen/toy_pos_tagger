import json
from collections import Counter
from sklearn.model_selection import train_test_split
from penn_treebank_data import sents
from vocabulary import Vocabulary, LabelVocabulary, BOS, EOS, UNK


max_size = 3000
min_count = 5
token_count = Counter([w.lower() for sent in sents for w,_ in sent])
tokens = token_count.most_common(max_size)

vocab = Vocabulary()
for token,count in tokens:
    if count >= min_count:
        vocab.add_token(token)

label_vocab = LabelVocabulary()
labels = set([tag for sent in sents for _,tag in sent])
for label in labels:
    label_vocab.add_token(label)

print('Vocab size:',vocab.size)
print('Label size:',label_vocab.size)
vocab.save('models/vocab-{}.pklb'.format(vocab.size))
label_vocab.save('models/label_vocab.pklb')
print('Vocab saved')

encoded_sents = [[(vocab[BOS],label_vocab[UNK])]\
                  +[(vocab[token.lower()],label_vocab[tag]) for token,tag in sent]\
                  +[(vocab[EOS],label_vocab[UNK])]
                    for sent in sents]

with open('data/encoded_sents.json','w') as fw:
    json.dump(encoded_sents,fw)

train,test = train_test_split(encoded_sents,test_size=0.2)
with open('data/train.json','w') as fw:
    long_sent = [pair for sent in train for pair in sent]
    json.dump(long_sent,fw)
with open('data/test.json','w') as fw:
    long_sent = [pair for sent in test for pair in sent]
    json.dump(long_sent,fw)
