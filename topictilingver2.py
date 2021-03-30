import pickle
import json

from numpy.linalg import norm
from numpy import dot


def create_vector(win, ntops):
    """ Creates a vector for a window span. Ntops - number of topics """
    vec = [0] * ntops
    wintoks = []
    for sent in win:
        wintoks.extend(sent)
    for token in wintoks:
        if token in ldamodel.keys():
            vec[ldamodel[token] - 1] += 1
    return vec, norm(vec)


def load_data():
    with open('/home/al/Documents/PythonFiles/files/disser/readydata/partial.json', 'r', encoding='utf8') as file:
        data = json.load(file)[:2000]
    return data


def movement(doc, winsize, ntops):
    if len(doc) < winsize:
        # print(f'Some doc has less than {winsize} sents!')
        return
    vecs = []
    start = 0
    end = winsize
    for i in range(len(doc) - winsize):
        vecs.append(create_vector(doc[start:end], ntops))
        vecs.append(create_vector(doc[end:winsize], ntops))
        start += 1
        end += 1
    return vecs


def cosine(vecs):
    dists = []
    for i in range(1, len(vecs), 2):
        dists.append(dot(vecs[i][0], vecs[i - 1][0]) / (vecs[i][1] * vecs[i - 1][1] + 1))
    return dists


def splitter(cos, doc, winsize):
    minimum = cos.index(min(cos))
    crack = (minimum + 1) * winsize
    doc1 = doc[:crack + 1]
    doc2 = doc[crack + 1:]
    return [doc1, doc2]


print('Loading LDA model...')
ldamodel = pickle.load(open('/home/al/Documents/PythonFiles/files/disser/readydata/20k/lemmatized/LDA', 'rb'))
print('Loading data...')
dataset = load_data()
windowsize = int(input('Window size: '))
topics = int(input('Quantity of topics: '))
print('Calculating vectors and cosine distances...')
finals = []
num = 1
lost = 0
finale = open('/home/al/Documents/PythonFiles/files/disser/readydata/testsplitted.txt', 'a', encoding='utf8')
with open('/home/al/Documents/PythonFiles/files/disser/readydata/partial_clean3.json', 'r', encoding='utf8') as jison:
    original = json.load(jison)
for k in range(len(dataset)):
    vectors = movement(dataset[k], windowsize, topics)
    if vectors:
        cosines = cosine(vectors)
        if not cosines:
            print(f'Text No {num} hasn\'t got enough sents!')
            lost += 1
        else:
            print(original[k], file=finale)
            print('_____', file=finale)
            print(*splitter(cosines, dataset[k], windowsize), sep='\n\n', end='\n\n\n', file=finale)
            print('~' * 10, file=finale)
            # finals.append(splitter(cosines, document, windowsize))
    num += 1
print(f'{lost} texts shorter than windowsize')
print('Done. Writing to file...')
finale.close()
# with open('/home/al/Documents/PythonFiles/files/disser/readydata/testsplitted.txt', 'w', encoding='utf8') as finale:
#     for elem in finals:
#         print(*elem, sep='\n\n', end='\n\n\n', file=finale)
#         print('~' * 10, file=finale)
