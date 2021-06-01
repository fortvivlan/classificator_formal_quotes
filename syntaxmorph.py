import pyconll
import pickle
from nltk import ngrams
from numpy.linalg import norm
from numpy import dot


def process_data(path):
    listofdocs = []
    rawlist = []
    with open(path, 'r', encoding='utf8') as file:
        doctoparse = ''
        doc = []
        rawdoc = []
        for line in file:
            if '</text>' in line and doctoparse:
                parse = pyconll.load_from_string(doctoparse)
                for sentence in parse:
                    sent = []
                    rawsent = []
                    for token in sentence:
                        rawsent.append(token.form)
                        if token.upos == 'PUNCT':
                            sent.append(token.form)
                        elif token.deprel != '_':
                            sent.append(token.deprel + str(token.feats))
                    doc.append(sent)
                    rawdoc.append(rawsent)
                doctoparse = ''
                listofdocs.append(doc)
                rawlist.append(rawdoc)
                doc = []
                rawdoc = []
            elif '<text>' not in line and '</text>' not in line:
                doctoparse += line
    return listofdocs, rawlist


def syntaxngrams(data, n):
    grams = set()
    for doc in data:
        for sent in doc:
            grams |= set(ngrams(sent, n))
    return {k: v for v, k in enumerate(grams)}


def create_vector(win, model):
    """ Creates a vector for a window span. """
    vec = [0] * len(model)
    wingrams = []
    for sent in win:
        wingrams.extend(sent)
    for ngram in wingrams:
        if ngram in model.keys():
            vec[model[ngram]] += 1
    return vec, norm(vec)  # so we have both a vector itself and its norm in a tuple


def movement(doc, winsize, model):
    """ Calculates vectors for sents winsize, sort of sent 1-2 * 3-4, then 2-3 * 4-5 and so on. """
    vecs = []
    start = 0
    end = 1
    for i in range(len(doc) - 1):
        # This way we have a list twice the size of intervals, so that calculating cosines we'd go with step = 2
        if doc[end:]:
            vecs.append(create_vector(doc[start:end], model))
            if len(doc) - 1 - end >= winsize:
                vecs.append(create_vector(doc[end:end + winsize], model))
            else:
                vecs.append(create_vector(doc[end:], model))
        if end - start >= winsize:
            start += 1
        end += 1
    return vecs


def cosine(vecs):
    """ Calculate cosine distances for vecs in a list """
    dists = []
    for i in range(1, len(vecs), 2):
        dists.append(dot(vecs[i][0], vecs[i - 1][0]) / (vecs[i][1] * vecs[i - 1][1] + 1))
    return dists


def splitter(cos, doc):
    """ Segment a doc by d-scores """
    segmented = []
    dscores = []
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
        dscores.append(0.5 * (hl + hr - 2 * cos[i]))
    mean = sum(dscores) / len(dscores)
    sigma = (sum((x - mean) ** 2 for x in dscores) / len(dscores)) ** 0.5
    # threshold = mean - sigma / 2
    threshold = mean + 1.5 * sigma
    for i in range(len(dscores)):
        if dscores[i] > threshold:
            breakerpoints.append(i + 1)  # list of cosines contains points between sents
    if not breakerpoints:  # no d-score appeared high enough to pass the threshold
        return
    # Segmenting work goes here
    starter = 0
    for index in breakerpoints:
        if index != starter:
            segmented.append(doc[starter:index])
        starter = index
    if doc[starter:]:  # considers tail
        segmented.append(doc[starter:])
    return segmented, breakerpoints


def topictiling():
    nograms = int(input('Number of gram: '))
    file = '/home/al/PythonFiles/files/disser/readydata/anastasyev/PALJ/ljpa_1.txt'
    dataset, rawtext = process_data(file)
    model = syntaxngrams(dataset, nograms)

    for i in range(len(dataset)):
        for j in range(len(dataset[i])):
            dataset[i][j] = list(ngrams(dataset[i][j], nograms))

    for windowsize in range(2, 6):
        no = f'{nograms}-gramSYNTAXMORPH_{windowsize}'
        breakpoints = {}
        pathtores = f'/home/al/PythonFiles/files/disser/experiments/experiment_{no}.txt'  # path to resulting text
        finale = open(pathtores, 'w', encoding='utf8')
        for k in range(500):
            vectors = movement(dataset[k], windowsize, model)
            if vectors:
                cosines = cosine(vectors)
                res = splitter(cosines, rawtext[k])
                if res:
                    for part in res[0]:
                        for sent in part:
                            print(*sent, file=finale)
                        print('@@@@@@@@SEG@@@@@@@@', file=finale)
                    breakpoints[k] = res[1]
                else:
                    print('ZERO BREAKPOINTS FOUND', file=finale)
                print('\n~~~ENDOFDOC~~~\n', file=finale)
        finale.close()
        pickle.dump(breakpoints, open(f'/home/al/PythonFiles/files/disser/experiments/breakpoints_{no}', 'wb'))


if __name__ == '__main__':
    topictiling()
