from morphoclass import MorphoToken
from collections import Counter
import pickle


def basetopic(dataset):
    """Standard topic tiling algorithm, open pos, stopwords not accounted for"""
    for i in range(len(dataset)):
        newdoc = []
        for sent in dataset[i]:
            newdoc.extend([token.lemma for token in sent if token.pos in {'NOUN', 'VERB', 'ADV', 'ADJ'}])
        dataset[i] = newdoc
    return dataset


def standardtopic(dataset):
    """Standard topic tiling algorithm, open pos, minus 100 most frequent words"""
    plain = []
    for doc in dataset:
        for sent in doc:
            plain.extend([token.lemma for token in sent if token.pos in {'NOUN', 'VERB', 'ADV', 'ADJ'}])
    freq = {token for token, count in Counter(plain).most_common(100)}
    del plain
    for i in range(len(dataset)):
        newdoc = []
        for sent in dataset[i]:
            newdoc.extend([token.lemma for token in sent
                           if token.pos in {'NOUN', 'VERB', 'ADV', 'ADJ'} and token.lemma not in freq])
        dataset[i] = newdoc
    return dataset


def simplefreq(dataset):
    """Blunt most frequent tokens"""
    plain = []
    for doc in dataset:
        for sent in doc:
            plain.extend([token.lemma for token in sent])
    freq = {token for token, count in Counter(plain).most_common(5000)}
    del plain
    for i in range(len(dataset)):
        newdoc = []
        for sent in dataset[i]:
            newdoc.extend([token.lemma for token in sent if token.lemma in freq])
        dataset[i] = newdoc
    return dataset


def nonwords(dataset):
    """Only non-word tokens"""
    for i in range(len(dataset)):
        newdoc = []
        for sent in dataset[i]:
            newdoc.extend([token.lemma for token in sent if token.category != 'word'])
        dataset[i] = newdoc
    return dataset


def closedclass(dataset):
    """Only words not in NOUN, VERB, ADJ, ADV"""
    for i in range(len(dataset)):
        newdoc = []
        for sent in dataset[i]:
            newdoc.extend([token.lemma for token in sent
                           if token.category == 'word' and token.pos not in {'NOUN', 'VERB', 'ADV', 'ADJ', 'PUNCT'}])
            # sometimes rnnmorph defines as punctuation what tokenizer defined as a word
        dataset[i] = newdoc
    return dataset


def closedclassplus(dataset):
    """Like previous but with emoji"""
    for i in range(len(dataset)):
        newdoc = []
        for sent in dataset[i]:
            newdoc.extend([token.lemma for token in sent
                           if token.category == 'emoji'
                           or token.category == 'word' and token.pos not in {'NOUN', 'VERB', 'ADV', 'ADJ', 'PUNCT'}])
        dataset[i] = newdoc
    return dataset


def closedclassandnonwords(dataset):
    """Like previous but with all punct and emoji"""
    for i in range(len(dataset)):
        newdoc = []
        for sent in dataset[i]:
            newdoc.extend([token.lemma for token in sent
                           if token.category == 'word' and token.pos not in {'NOUN', 'VERB', 'ADV', 'ADJ'}
                           or token.category != 'word'])
        dataset[i] = newdoc
    return dataset


def closedclassadvs(dataset):
    """Only words not in NOUN, VERB, ADJ, ADV"""
    for i in range(len(dataset)):
        newdoc = []
        for sent in dataset[i]:
            newdoc.extend([token.lemma for token in sent
                           if token.category == 'word' and token.pos not in {'NOUN', 'VERB', 'ADJ', 'PUNCT'}])
            # sometimes rnnmorph defines as punctuation what tokenizer defined as a word
        dataset[i] = newdoc
    return dataset


def nonounsverbs(dataset):
    """Only words not in NOUN, VERB, ADJ, ADV"""
    for i in range(len(dataset)):
        newdoc = []
        for sent in dataset[i]:
            newdoc.extend([token.lemma for token in sent
                           if token.category == 'word' and token.pos not in {'NOUN', 'VERB', 'PUNCT'}])
            # sometimes rnnmorph defines as punctuation what tokenizer defined as a word
        dataset[i] = newdoc
    return dataset


def pronsandother(dataset):
    """Pronouns and some other"""
    for i in range(len(dataset)):
        newdoc = []
        for sent in dataset[i]:
            newdoc.extend([token.lemma for token in sent
                           if token.category == 'word' and token.pos in {'PRON', 'PART', 'ADV', 'INTJ', 'NUM', 'PUNCT'}
                           or token.category != 'word'])
        dataset[i] = newdoc
    return dataset


def justprons(dataset):
    """Pronouns"""
    for i in range(len(dataset)):
        newdoc = []
        for sent in dataset[i]:
            newdoc.extend([token.lemma for token in sent if token.pos == 'PRON'])
        dataset[i] = newdoc
    return dataset
