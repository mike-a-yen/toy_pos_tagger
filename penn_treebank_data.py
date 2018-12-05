import nltk
from nltk.corpus import treebank

sents = treebank.tagged_sents(tagset='universal')
n_sents = len(sents)
n_words = sum(map(len,sents))
n_tags = len(set([tag for sent in sents for word,tag in sent]))
print('N sents:',n_sents)
print('N words:',n_words)
print('N tags:',n_tags)
