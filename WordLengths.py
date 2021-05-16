import pickle
import os
import pandas as pd
from morphoclass import MorphoToken


def quartilecounter(dataset):
    segment = []
    counter = 0
    for doc in dataset:
        if counter >= 60000:
            break
        for sent in doc:
            for token in sent:
                if counter < 600000 and token.category == 'word':
                    segment.append(len(token.form))
                    counter += 1
    segment.sort()
    if segment:
        mean = round(sum(segment) / len(segment), 2)
        lower = len(segment) * 0.25
        upper = len(segment) * 0.75
        middle = len(segment) // 2
        if len(segment) % 2 == 0 and len(segment) > 2:
            median = (segment[middle] + segment[middle + 1]) / 2
        else:
            median = segment[middle]
        lowerint = int(lower)
        upperint = int(upper)
        if lowerint != lower and lowerint != len(segment) - 1:
            lower_q = (segment[lowerint] + segment[lowerint + 1]) / 2
        else:
            lower_q = segment[lowerint]
        if upperint != upper and upperint != len(segment) - 1:
            upper_q = (segment[upperint] + segment[upperint + 1]) / 2
        else:
            upper_q = segment[upperint]
        quartile = upper_q - lower_q
        return {'Межквартильный размах': quartile, 'Среднее арифметическое': mean, 'Медиана': median,
                'Верхняя квартиль': segment[upperint], 'Нижняя квартиль': segment[lowerint]}


def main():
    p = '/home/al/PythonFiles/files/disser/readydata/morpho'
    files = os.listdir(p)
    results = {}
    for f in files:
        if os.path.splitext(f)[1] or os.path.isdir(f):
            continue
        fullp = os.path.join(p, f)
        data = pickle.load(open(fullp, 'rb'))
        results[f] = quartilecounter(data)
    df = pd.DataFrame.from_dict(results, orient='index')
    df.to_excel('wordlengths.xlsx')


if __name__ == '__main__':
    main()
