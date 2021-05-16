import pickle
import os
import pandas as pd
from morphoclass import MorphoToken


def lex_div(dataset):
    res = {}
    uniquenonw = set()
    nonw = 0
    words = 0
    uniquewords = set()
    counter = 0

    for doc in dataset:
        if counter >= 600000:
            break
        for sent in doc:
            for token in sent:
                if counter < 600000:
                    if token.category == 'word':
                        uniquewords.add(token.lemma)
                        words += 1

                    else:
                        uniquenonw.add(token.form)
                        nonw += 1
                counter += 1
    if words:
        res['lexical'] = round(len(uniquewords) * 100 / words, 2)
    else:
        res['lexical'] = 0
    if nonw:
        res['nonlexical'] = round(len(uniquenonw) * 100 / nonw, 4)
    else:
        res['nonlexical'] = 0
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
        results[f] = lex_div(data)
    df = pd.DataFrame.from_dict(results, orient='index')
    df.to_excel('lexdiv.xlsx')


if __name__ == '__main__':
    main()
