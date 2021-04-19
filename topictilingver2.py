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
    return vec, norm(vec)  # so we have both a vector itself and its norm in a tuple


def load_data():
    """ Loads data... suddenly """
    with open(r'F:\CODE\files\disser\readydata\partial.json', 'r', encoding='utf8') as file:  # no ready json
        data = json.load(file)[:100]  # size restricted for testing purposes
    return data


def movement(doc, winsize, ntops):
    """ Calculates vectors for sents winsize, sort of sent 1-2 * 3-4, then 2-3 * 4-5 and so on. Be careful:
    first span of a winsize isn't being split, so you can't get a split between 1st and 2nd sents if your
    winsize is 2 """
    if len(doc) <= winsize:  # needs to have at least winsize length
        return
    vecs = []
    start = 0
    end = winsize
    # end = 1 - to get this to work you'd have to make several smart changes
    for i in range(len(doc) - winsize):
        # This way we have a list twice the size of intervals, so that calculating cosines we'd go with step = 2
        vecs.append(create_vector(doc[start:end], ntops))
        vecs.append(create_vector(doc[end:end + winsize], ntops))
        start += 1
        end += 1
    return vecs


def cosine(vecs):
    """ Calculate cosine distances for vecs in a list """
    dists = []
    for i in range(1, len(vecs), 2):
        dists.append(dot(vecs[i][0], vecs[i - 1][0]) / (vecs[i][1] * vecs[i - 1][1] + 1))
    return dists


def splitter(cos, doc, winsize):
    """ Segment a doc by d-scores """
    segmented = []
    mean = sum(cos) / len(cos)
    sigma = (sum((x - mean) ** 2 for x in cos) / len(cos)) ** 0.5
    threshold = mean - sigma / 2  
    # d-scores less than threshold are ok: d-score for a local max is 0, the greater d-score, the deeper local min
    breakerpoints = []
    for i in range(len(cos)):
        # calculating HL (highest left)
        hl = cos[i]
        j = i
        while True:
            if j == 0:
                break
            if cos[j - 1] > cos[j]:  
                # we break if previous cosine equals current. 
                # Flats are not considered, this way we lower d-scores for sequences of zeros
                hl = cos[j - 1]
                j -= 1
            else:
                break
        # calculating HR (highest right)
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
            breakerpoints.append(i + winsize)  # the algorithm works so that index + winsize gives the sent we need
    if not breakerpoints:  # no d-score appeared high enough to pass the threshold
        return None
    # Segmenting work goes here
    starter = 0
    for index in breakerpoints:
        if index != starter:
            segmented.append(doc[starter:index])
        starter = index
    if doc[starter:]:  # considers tail
        segmented.append(doc[starter:])
    return segmented


print('Loading LDA model...')
ldamodel = pickle.load(open(r'F:\CODE\files\disser\LDAs\LDAmodel', 'rb'))  # LDA MODEL LOAD check it if you change model
print('Loading data...')
dataset = load_data()
windowsize = int(input('Window size: '))
topics = int(input('Quantity of topics: '))  # I guess I could code it into the model but I was too lazy
print('Calculating vectors and cosine distances...')
lost = 0  # to count texts lost due to their incoherent length
pathtores = r'F:\CODE\files\disser\readydata\testsplitpuncstop.txt'  # path to resulting text file
if os.path.exists(pathtores):  # to prevent script from endlessly adding to file
    os.remove(pathtores)
finale = open(pathtores, 'a', encoding='utf8')
with open(r'F:\CODE\files\disser\readydata\partial_clean3.json', 'r', encoding='utf8') as jison:
    original = json.load(jison)
for k in range(len(dataset)):
    vectors = movement(dataset[k], windowsize, topics)
    if vectors:
        cosines = cosine(vectors)
        print(original[k], file=finale)
        print('_____', file=finale)
        res = splitter(cosines, dataset[k], windowsize)
        if res:
            print(*res, sep='\n\n', end='\n\n\n', file=finale)
        else:
            print('ZERO BREAKPOINTS FOUND', file=finale)
        print('~' * 10, file=finale)
    else:
        lost += 1
print(f'{lost} texts shorter than windowsize')
finale.close()
