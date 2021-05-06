import pickle
import os
import pandas as pd
from collections import Counter
from morphoclass import MorphoToken


def lex_div(dataset, name):
    poscount = Counter()
    tokencount = 0
    counter = 0
    for doc in dataset:
        if counter >= 300:
            break
        if 15 <= len(doc) <= 30:
            counter += 1
            for sent in doc:
                for token in sent:
                    if token.category == 'word' and token.pos != 'PUNCT':
                        tokencount += 1
                        poscount[token.pos] += 1
    if counter < 300:
        print(f'{name} doesn\'t have enough docs. got only {counter}')
    res = {pos: round(poscount[pos] * 100 / tokencount, 2) for pos in poscount.keys()}
    return res


def main():
    p = '/home/al/PythonFiles/files/disser/readydata/morpho'
    files = os.listdir(p)
    results = {}
    for f in files:
        if os.path.splitext(f)[1] or os.path.isdir(f):
            continue
        fullp = os.path.join(p, f)
        data = pickle.load(open(fullp, 'rb'))
        results[f] = lex_div(data, f)
    df = pd.DataFrame.from_dict(results, orient='index')
    df.to_excel('/home/al/PythonFiles/files/disser/readydata/posdiv.xlsx')


if __name__ == '__main__':
    main()
