import json
import pickle
import os
import sys
import re
from collections import Counter
from nltk.corpus import stopwords
from time import time

import gensim
import gensim.corpora as corpora
import pandas as pd


def dataset_process(data):
    # pattern = r'(?:\w)+(?:-?\w+)*'
    onepercent = len(data) // 100
    for j in range(len(data)):
        i = 0
        c = 0
        length = len(data[j])
        while i + c < length:
            data[j][i] = data[j][i + c]
            if (re.fullmatch(r'[-\W]+|\[\d+]', data[j][i]) or data[j][i] in stop_words) and not data[j][i] in ',.':
                c += 1
            else:
                i += 1
        del data[j][-c:]
        if j % onepercent == 0:
            print(j // onepercent, '%', sep='')
    return data


def load_fulldata():
    path = '/home/al/Documents/PythonFiles/files/disser/readydata/20k/lemmatized/'
    files = os.listdir(path)
    data = []
    for file in files:
        pathfile = os.path.join(path, file)
        if pathfile.endswith('json'):
            with open(pathfile, 'r', encoding='utf8') as f:
                ds = json.load(f)
                for element in ds:
                    doc = []
                    for sent in element:
                        doc.extend(sent)
                    data.append(doc)
    return data


def load_smalldata(name):
    path = f'/home/al/Documents/PythonFiles/files/disser/readydata/20k/lemmatized/{name}.json'
    with open(path, 'r',
              encoding='utf8') as file:
        ds = json.load(file)
    data = []
    for element in ds:
        doc = []
        for sent in element:
            doc.extend(sent)
        data.append(doc)
    return data


# Mallet
mallet_path = '/home/al/Documents/PythonFiles/MyCode/Disser/mallet-2.0.8/bin/mallet'
stop_words = stopwords.words('russian')

''' Choose your fighter '''
print('Scriptwork started!')
t = time()
print('Full(1) or small(2)?')
choicefs = int(input())
if choicefs == 1:
    dataset = load_fulldata()
elif choicefs == 2:
    p = '/home/al/Documents/PythonFiles/files/disser/readydata/20k/lemmatized/'
    filenames = os.listdir(p)
    print('Your fighters are as follows:')
    print(*filenames)
    fighter = input('Choose your fighter: ')
    dataset = load_smalldata(fighter)
else:
    sys.exit('U dumbass')

print('"""datasets loaded"""')

''' Process dataset'''
print('Started processing by stopwords and punctuation')
data_processed = dataset_process(dataset)
del dataset
print(f'Length of processed dataset: {sum(map(len, data_processed))}')
print('"""datasets processed"""')

''' Create model '''
id2word = corpora.Dictionary(data_processed)
print('"""corpus dictionary created"""')
corpus = [id2word.doc2bow(text) for text in data_processed]
print('"""corpus gathered"""')
tops = int(input('How many topics? Zero for default (20)\n'))
if not tops:
    tops = 20
print('We start creating model')
ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=tops, id2word=id2word)
print('"""model created"""')

''' Creating dataframe [and writing it to csv] '''
topns = int(input('How many terms? Zero for default (20)\n'))
if not topns:
    topns = 20
topics = [[(term, round(wt, 3)) for term, wt in ldamallet.show_topic(n, topn=topns)]
          for n in range(0, ldamallet.num_topics)]
topics_df = pd.DataFrame([[term for term, wt in topic] for topic in topics],
                         columns=['Term' + str(i) for i in range(1, 21)],
                         index=['Topic ' + str(t) for t in range(1, ldamallet.num_topics+1)]).T
print('"""dataframe gathered"""')

''' Annotation '''
LDAmodel = {}
for term in topics_df.itertuples():
    for i in range(1, len(term)):
        LDAmodel[term[i]] = i
print('"""terms annotated. saving and printing..."""')

''' Saving annotation '''
pickle.dump(LDAmodel, open('/home/al/Documents/PythonFiles/files/disser/readydata/20k/lemmatized/LDAver2', 'wb'))
with open('/home/al/Documents/PythonFiles/files/disser/readydata/20k/lemmatized/LDAtestv2.txt', 'w', encoding='utf8') \
        as testtext:
    s = 1
    for key, value in sorted(LDAmodel.items(), key=lambda x: (x[1], x[0])):
        if s != value:
            print(file=testtext)
            s = value
        print(f'Term: {key:<10}TopicNo: {value:<30}', file=testtext)

print(f'Time of work: {round(time() - t, 2)}. Sloooowwwww')
