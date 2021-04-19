import json
import pickle
import os
import sys
from collections import Counter

import gensim.corpora as corpora
from gensim.models.wrappers import LdaMallet
import pandas as pd


def dataset_process(data, size=3000):
    """ Function to choose which tokens are used for the model: this version is for frequent terms """
    words = []
    for sent in data:  # we don't need sents to count freqs
        words.extend(sent)
    freq = {tok[0] for tok in Counter(words).most_common(size)}
    del words  # to save RAM
    for docn in range(len(data)):  # docn is an index of a doc in dataset
        # deleting tokens not in freq from the docs
        ind = 0  # index of a token
        count = 0  # shifting
        length = len(data)
        while ind + count < length:
            data[docn][ind] = data[docn][ind + count]  # shifting the list
            if data[docn][ind] in freq:  # if token in freq, we leave it be and increase index
                ind += 1
            else:  # if not, we erase it by shifting right side elements
                count += 1
        del data[docn][-count:]  # deleting tail


def load_fulldata():
    """ Loading all datasets - beware, they're around 0.5 Gb """
    path = r'F:\CODE\files\disser\readydata\20k\lemmatized'
    files = os.listdir(path)
    data = []
    for file in files:
        pathfile = os.path.join(path, file)
        if pathfile.endswith('json'):
            with open(pathfile, 'r', encoding='utf8') as f:
                ds = json.load(f)
                # The data is in docs split into sents. We don't need sents now
                for element in ds:
                    doc = []
                    for sent in element:
                        doc.extend(sent)
                    data.append(doc)
    return data


def load_smalldata(name):
    """ Loading one dataset to test model """
    path = f'F:\\CODE\\files\\disser\\readydata\\20k\\lemmatized\\{name}.json'
    with open(path, 'r',
              encoding='utf8') as file:
        ds = json.load(file)
    data = []
    # The data is in docs split into sents. We don't need sents now
    for element in ds:
        doc = []
        for sent in element:
            doc.extend(sent)
        data.append(doc)
    return data


# Mallet - quite a question if Mallet will work on windows without JDK installed (I'll install it anyway tho)
mallet_path = r'F:\CODE\MyCode\Disser\mallet-2.0.8\bin\mallet'

''' Choosing dataset '''
print('Scriptwork started!')
print('Full(1) or small(2)?')
choicefs = int(input())
if choicefs == 1:
    dataset = load_fulldata()
elif choicefs == 2:
    p = r'F:\CODE\files\disser\readydata\20k\lemmatized'
    filenames = os.listdir(p)
    print('Your fighters are as follows:')
    print(*filenames)
    fighter = input('Choose your fighter: ')
    dataset = load_smalldata(fighter)
else:
    sys.exit('U dumbass')  # why did I do that?..

print('"""datasets loaded"""')

''' Process dataset '''
sizefreqs = int(input('How many common tokens? Zero for default (3000)\n'))
if sizefreqs:
    dataset_process(dataset, sizefreqs)
else:
    dataset_process(dataset)
print('"""datasets processed by frequency"""')

''' Create model '''
id2word = corpora.Dictionary(dataset)
print('"""corpus dictionary created"""')
corpus = [id2word.doc2bow(text) for text in dataset]
print('"""corpus gathered"""')
tops = int(input('How many topics? Zero for default (20)\n'))
if not tops:
    tops = 20
print('We start creating model')
ldamallet = LdaMallet(mallet_path, corpus=corpus, num_topics=tops, id2word=id2word)
print('"""model created"""')

''' Creating dataframe '''
topics = [[(term, round(wt, 3)) for term, wt in ldamallet.show_topic(n, topn=20)]
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
pickle.dump(LDAmodel, open(r'F:\CODE\files\disser\LDAs\LDAmodel', 'wb'))
with open(r'F:\CODE\files\disser\LDAs\LDAtest.txt', 'w', encoding='utf8') \
        as testtext:
    s = 1
    for key, value in sorted(LDAmodel.items(), key=lambda x: (x[1], x[0])):
        if s != value:
            print(file=testtext)
            s = value
        print(f'Term: {key:<10}TopicNo: {value:<30}', file=testtext)
