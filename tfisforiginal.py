import pickle
import os
import pandas as pd

from collections import defaultdict, Counter
from morphoclass import MorphoToken
from numpy.linalg import norm
from numpy import dot


def data_loader(data):
    p = '/home/al/PythonFiles/files/disser/readydata/morpho'
    files = os.listdir(p)
    for f in files:
        if os.path.splitext(f)[1] or os.path.isdir(f):
            continue
        fullp = os.path.join(p, f)
        dataset = pickle.load(open(fullp, 'rb'))
        for doc in dataset:
            for sent in doc:
                data[f].extend([token.lemma for token in sent
                                if token.category != 'punct' and token.pos != 'NOUN'])


def dictcreator(data):
    wordcount = Counter()
    for key in data:
        wordcount += Counter(data[key])
    dictionary = {token for token, freq in sorted(wordcount.items(), key=lambda x: -x[1])[:10000]}
    return sorted(dictionary)


def vectors(data, allwords):
    vec = {}
    for key in data:
        vec[key] = dict.fromkeys(allwords, 0)
        for token in data[key]:
            if token in vec[key]:
                vec[key][token] += 1
    for key in vec:
        vec[key] = [value for token, value in sorted(vec[key].items())]
    return vec


def cosines(vecs):
    cosine = defaultdict(dict)
    norms = {}
    for seg in vecs:
        norms[seg] = norm(vecs[seg])
    for seg in vecs:
        segcos = {}
        for others in vecs:
            if vecs[others] is not vecs[seg]:
                segcos[others] = round(dot(vecs[others], vecs[seg]) / (norms[others] * norms[seg]), 4)
        cosine[seg].update(segcos)
    return cosine


def main():
    fulldata = defaultdict(list)
    data_loader(fulldata)
    allwords = dictcreator(fulldata)
    vecs = vectors(fulldata, allwords)
    results = cosines(vecs)
    df = pd.DataFrame.from_dict(results, orient='index')
    df.to_excel('cosines_10000f.xlsx')


if __name__ == '__main__':
    main()
