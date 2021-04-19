import pickle
import json
import os

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
        data = json.load(file)[:100]
    return data


def movement(doc, winsize, ntops):
    if len(doc) <= winsize:
        # print(f'Some doc has less than {winsize} sents!')
        return
    vecs = []
    start = 0
    end = winsize
    # end = 1
    for i in range(len(doc) - winsize):
        vecs.append(create_vector(doc[start:end], ntops))
        vecs.append(create_vector(doc[end:end + winsize], ntops))
        start += 1
        end += 1
    return vecs


def cosine(vecs):
    dists = []
    for i in range(1, len(vecs), 2):
        dists.append(dot(vecs[i][0], vecs[i - 1][0]) / (vecs[i][1] * vecs[i - 1][1] + 1))
    return dists


def splitter(cos, doc, winsize):
    segmented = []
    mean = sum(cos) / len(cos)
    sigma = (sum((x - mean) ** 2 for x in cos) / len(cos)) ** 0.5
    threshold = mean - sigma / 2
    breakerpoints = []
    for i in range(len(cos)):
        hl = cos[i]
        j = i
        while True:
            if j == 0:
                break
            if cos[j - 1] > cos[j]:
                hl = cos[j - 1]
                j -= 1
            else:
                break
        hr = cos[i]
        r = i
        while True:
            if r == len(cos) - 1:
                break
            if cos[r + 1] > cos[j]:
                hr = cos[r + 1]
                r += 1
            else:
                break
        dscore = 0.5 * (hl + hr - 2 * cos[i])
        if dscore > threshold:
            breakerpoints.append(i + winsize)
    if not breakerpoints:
        return None
    starter = 0
    for index in breakerpoints:
        if index != starter:
            segmented.append(doc[starter:index])
        starter = index
    if starter != len(doc) - 1:
        segmented.append(doc[starter:])
    return segmented


print('Loading LDA model...')
ldamodel = pickle.load(open('/home/al/Documents/PythonFiles/files/disser/readydata/20k/lemmatized/LDAver', 'rb'))
print('Loading data...')
dataset = load_data()
windowsize = int(input('Window size: '))
topics = int(input('Quantity of topics: '))
print('Calculating vectors and cosine distances...')
num = 1
lost = 0
if os.path.exists('/home/al/Documents/PythonFiles/files/disser/readydata/testsplitpuncstop.txt'):
    os.remove('/home/al/Documents/PythonFiles/files/disser/readydata/testsplitpuncstop.txt')
finale = open('/home/al/Documents/PythonFiles/files/disser/readydata/testsplitpuncstop.txt', 'a', encoding='utf8')
with open('/home/al/Documents/PythonFiles/files/disser/readydata/partial_clean3.json', 'r', encoding='utf8') as jison:
    original = json.load(jison)
for k in range(len(dataset)):
    vectors = movement(dataset[k], windowsize, topics)
    if vectors:
        cosines = cosine(vectors)
        print(original[k], file=finale)
        print('_____', file=finale)
        # print('Vector:', vectors)
        # print('Cosines: ', cosines)
        print(cosines, file=finale)
        res = splitter(cosines, dataset[k], windowsize)
        if res:
            print(*res, sep='\n\n', end='\n\n\n', file=finale)
        else:
            print('ZERO BREAKPOINTS FOUND', file=finale)
        # if splitter:
        #     print('Result:')
        #     print(*splitter(cosines, dataset[k], windowsize), sep='\n\n', end='\n\n\n')
        # else:
        #     print('Couldn\'t split text!')
        print('~' * 10, file=finale)
    else:
        # print(f'Text No {num} hasn\'t got enough sents!')
        lost += 1
    num += 1
print(f'{lost} texts shorter than windowsize')
finale.close()
