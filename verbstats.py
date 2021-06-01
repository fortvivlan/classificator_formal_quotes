import pyconll
from collections import Counter, defaultdict
import os
import pandas as pd


def process_data(path):
    aspect = Counter()
    mood = Counter()
    tense = Counter()
    with open(path, 'r', encoding='utf8') as file:
        doc = ''
        for line in file:
            if '</text>' in line and doc:
                parse = pyconll.load_from_string(doc)
                for sentence in parse:
                    for token in sentence:
                        if token.upos == 'VERB':
                            if token.feats.get('Aspect'):
                                aspect['Aspect: ' + ''.join(token.feats['Aspect'])] += 1
                            if token.feats.get('Mood'):
                                mood['Mood: ' + ''.join(token.feats['Mood'])] += 1
                            if token.feats.get('Tense'):
                                tense['Tense: ' + ''.join(token.feats['Tense'])] += 1
                doc = ''
            elif '<text>' not in line and '</text>' not in line:
                doc += line
    return aspect, mood, tense


def summer(dct):
    sumz = sum(dct.values())
    for key in dct:
        dct[key] = round(dct[key] * 100 / sumz, 2)


def main():
    root = '/home/al/PythonFiles/files/disser/readydata/anastasyev'
    totals = defaultdict(dict)
    folders = os.listdir(root)
    for folder in folders:
        files = os.listdir(os.path.join(root, folder))
        totalaspect = defaultdict(dict)
        totalmood = defaultdict(dict)
        totaltense = defaultdict(dict)
        for file in files:
            asp, mood, tense = (process_data(os.path.join(root, folder, file)))
            totalaspect.update(asp)
            totalmood.update(mood)
            totaltense.update(tense)
        summer(totalaspect)
        summer(totalmood)
        summer(totaltense)
        totals[folder].update(totalaspect)
        totals[folder].update(totalmood)
        totals[folder].update(totaltense)

    df = pd.DataFrame.from_dict(totals, orient='index')
    df.to_excel('/home/al/PythonFiles/files/disser/readydata/anast_verbdiv.xlsx')


if __name__ == '__main__':
    main()
