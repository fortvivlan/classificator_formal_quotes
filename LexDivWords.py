import pickle
import os
import pandas as pd
from morphoclass import MorphoToken


def lex_div(dataset, name):
    res = {}
    uniquenonw = set()
    nonw = 0
    words = 0
    uniquewords = set()
    uniquemoji = set()
    res['emoji total'] = 0
    counter = 0
    for doc in dataset:
        if counter >= 300:
            break
        if 15 <= len(doc) <= 30:
            counter += 1
            for sent in doc:
                for token in sent:
                    if token.category == 'word':
                        uniquewords.add(token.lemma)
                        words += 1
                    else:
                        if token.category == 'emoji':
                            uniquemoji.add(token.form)
                            res['emoji total'] += 1
                        uniquenonw.add(token.form)
                        nonw += 1
    if words:
        res['lexical'] = round(len(uniquewords) * 100 / words, 2)
    else:
        res['lexical'] = 0
    if nonw:
        res['nonlexical'] = round(len(uniquenonw) * 100 / nonw, 4)
    else:
        res['nonlexical'] = 0
    res['unique emoji'] = len(uniquemoji)
    if counter < 300:
        print(f'{name} doesn\'t have enough docs. got only {counter}')
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
    df.to_excel('/home/al/PythonFiles/files/disser/readydata/lexdiv.xlsx')


if __name__ == '__main__':
    main()
