from razdel import sentenize
from tokenice4 import Tokenice
from rnnmorph.predictor import RNNMorphPredictor
import json
from json import JSONEncoder
import datetime
import os
import pickle
from morphoclass import MorphoToken


class TokenEncoder(JSONEncoder):
    def default(self, o):
        return o.__dict__


def parsing(path, name):
    starttime = datetime.datetime.now().replace(microsecond=0)
    print(f'We start parsing {name}. Current time is {starttime}.')
    with open(path, 'r', encoding='utf8') \
            as jison:
        dataset = json.load(jison)

    readydata = []
    tokenice = Tokenice()
    predictor = RNNMorphPredictor(language='ru')
    n = 1
    onepercent = len(dataset) // 100
    print(f'Count to {len(dataset)}!11')
    for elem in dataset:
        sents = [s.text for s in sentenize(elem['text']) if s.text]
        for i in range(len(sents)):
            tokenice.tokenicer(sents[i])
            tokeniced = [(token.form, token.category) for token in tokenice]
            rnnmorphed = [(token.normal_form, token.pos) for token in predictor.predict([tok[0] for tok in tokeniced])]
            newsent = []
            for j in range(len(tokeniced)):
                newsent.append(MorphoToken(form=tokeniced[j][0], lemma=rnnmorphed[j][0],
                                           category=tokeniced[j][1], pos=rnnmorphed[j][1]))
            sents[i] = newsent
        readydata.append(sents)
        if n % onepercent == 0:
            print(n // onepercent, '%', sep='')
        n += 1

    endtime = datetime.datetime.now().replace(microsecond=0)
    print(f'We\'ve done it! {name} parsed, hallelujah! Time passed:', endtime - starttime)
    with open(f'/home/al/PythonFiles/files/disser/readydata/morpho/{name}.txt', 'w', encoding='utf8') as file:
        for elem in readydata[:20]:
            print(*elem, sep='\n', file=file)
            print('~' * 10, file=file)

    pickle.dump(readydata, open(f'/home/al/PythonFiles/files/disser/readydata/morpho/{name}', 'wb'))
    with open(f'/home/al/PythonFiles/files/disser/readydata/morpho/{name}.json', 'w', encoding='utf8') as jsonfile:
        json.dump(readydata, jsonfile, ensure_ascii=False, cls=TokenEncoder)
    print('Recording finished, next...')


def work():
    p = '/home/al/PythonFiles/files/disser/readydata/20k/test'
    files = os.listdir(p)
    for f in files:
        if not f.endswith('.json'):
            continue
        parsing(os.path.join(p, f), f[:-5])
    print("It's all done, boss!")


if __name__ == "__main__":
    work()
