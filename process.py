from morphoclass import MorphoToken
from collections import Counter


def testtopic(dataset):
    """Standard topic tiling algorithm, open pos"""
    for i in range(len(dataset)):
        newdoc = []
        for sent in dataset[i]:
            newdoc.extend([token.lemma for token in sent if token.pos in {'NOUN', 'VERB', 'ADV', 'ADJ'}])
        dataset[i] = newdoc
    return dataset


def stupidfreq(dataset):
    plain = []
    for doc in dataset:
        for sent in doc:
            plain.extend([token.lemma for token in sent])
    freq = {token for token, count in Counter(plain).most_common(3000)}
    del plain
    for i in range(len(dataset)):
        newdoc = []
        for sent in dataset[i]:
            newdoc.extend([token.lemma for token in sent if token.lemma in freq])
        dataset[i] = newdoc
    return dataset


def nonwords(dataset):
    result = []
    for doc in dataset:
        newdoc = []
        for sent in doc:
            newdoc.extend([token.lemma for token in sent if token.category != 'word'])
        if newdoc:
            result.append(newdoc)
    del dataset
    return result
