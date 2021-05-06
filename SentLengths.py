import pickle
import os
import pandas as pd
from morphoclass import MorphoToken


def quartilecounter(dataset):
    docs = []
    for doc in dataset:
        sample = []
        for sent in doc:
            sample.append(len([token.form for token in sent if token.category == 'word']))
        sample.sort()
        if sample:
            mean = sum(sample) / len(sample)
            samplingrange = sample[-1] - sample[0]
            lower = len(sample) * 0.25
            upper = len(sample) * 0.75
            middle = len(sample) // 2
            if len(sample) % 2 == 0 and len(sample) > 2:
                median = (sample[middle] + sample[middle + 1]) / 2
            else:
                median = sample[middle]
            lowerint = int(lower)
            upperint = int(upper)
            if lowerint != lower and lowerint != len(sample) - 1:
                lower_q = (sample[lowerint] + sample[lowerint + 1]) / 2
            else:
                lower_q = sample[lowerint]
            if upperint != upper and upperint != len(sample) - 1:
                upper_q = (sample[upperint] + sample[upperint + 1]) / 2
            else:
                upper_q = sample[upperint]
            quartile = upper_q - lower_q
            docs.append({'srange': samplingrange, 'quart': quartile, 'mean': mean,
                         'median': median, 'upper': upperint, 'lower': lowerint})
        else:
            docs.append({'srange': 0, 'quart': 0, 'mean': 0, 'median': 0, 'upper': 0, 'lower': 0})
    return docs


def main():
    p = '/home/al/PythonFiles/files/disser/readydata/morpho'
    files = os.listdir(p)
    results = {}
    for f in files:
        if os.path.splitext(f)[1] or os.path.isdir(f):
            continue
        fullp = os.path.join(p, f)
        data = pickle.load(open(fullp, 'rb'))
        rangequart = quartilecounter(data)
        rangemean = 0
        quartmean = 0
        totalmean = 0
        medmean = 0
        upperq = 0
        lowerq = 0
        length = len(rangequart)
        for elem in rangequart:
            rangemean += elem['srange']
            quartmean += elem['quart']
            totalmean += elem['mean']
            medmean += elem['median']
            upperq += elem['upper']
            lowerq += elem['lower']
        results[f] = {'range': round(rangemean / length, 2), 'quartile diff': round(quartmean / length, 2),
                      'mean': round(totalmean / length, 2), 'median': round(medmean / length, 2),
                      'upper quartile': round(upperq / length, 2), 'lower quartile': round(lowerq / length, 2)}
    print(results)
    df = pd.DataFrame.from_dict(results, orient='index')
    df.to_excel('/home/al/PythonFiles/files/disser/readydata/sentlengths.xlsx')


if __name__ == '__main__':
    main()
