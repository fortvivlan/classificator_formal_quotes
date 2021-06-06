import pyconll
from collections import Counter, defaultdict
import os
import pandas as pd


def process_data(path):
    syntax = Counter()
    with open(path, 'r', encoding='utf8') as file:
        doc = ''
        for line in file:
            if '</text>' in line and doc:
                parse = pyconll.load_from_string(doc)
                for sentence in parse:
                    for token in sentence:
                        if token.deprel != '_':
                            syntax[token.deprel] += 1
                doc = ''
            elif '<text>' not in line and '</text>' not in line:
                doc += line
    return syntax


def summer(dct):
    sumz = sum(dct.values())
    for key in dct:
        dct[key] = round(dct[key] * 100 / sumz, 2)
    return dct


def main():
    root = '/home/al/PythonFiles/files/disser/readydata/anastasyev'
    totals = defaultdict(dict)
    folders = os.listdir(root)
    for folder in folders:
        files = os.listdir(os.path.join(root, folder))
        for file in files:
            totals[folder].update(summer(process_data(os.path.join(root, folder, file))))

    df = pd.DataFrame.from_dict(totals)
    df.to_excel('/home/al/PythonFiles/files/disser/readydata/anast_syntax.xlsx')


if __name__ == '__main__':
    main()
