"""First version of annotating which uses frequency"""

import pickle
import os
import sys

from inspect import getmembers, isfunction
import process
import gensim.corpora as corpora
from gensim.models.wrappers import LdaMallet
import pandas as pd


def load_fulldata(func):
    """ Loading all datasets - beware, they're around several Gb """
    bigresult = []
    path = '/home/al/PythonFiles/files/disser/readydata/morpho/'
    files = os.listdir(path)
    for file in files:
        pathfile = os.path.join(path, file)
        if os.path.splitext(file)[1] or os.path.isdir(file) or file == 'PARTIAL_LJ':
            continue
        data = pickle.load(open(pathfile, 'rb'))
        bigresult.extend(func(data))
        del data
        print(f'\t{file} processed')
    return bigresult


def load_smalldata(name, func):
    """ Loading one dataset to test model """
    path = f'/home/al/PythonFiles/files/disser/readydata/morpho/{name}'
    data = pickle.load(open(path, 'rb'))
    return func(data)


def main():
    # Mallet
    mallet_path = '/home/al/PythonFiles/MyCode/Disser/mallet-2.0.8/bin/mallet'

    ''' Choosing dataset '''
    print('Badum-tss!')
    funcs = dict(getmembers(process, isfunction))
    print('Available processes as far:', *list(funcs.keys()))
    choiceprocess = input('Which one we use?\n')
    processfunc = funcs.get(choiceprocess)
    if not processfunc:
        sys.exit('Something went awry! Look what u\' re typing')
    print('Full(1) or small(2)? Beware! full dataset is damn huge!')
    choicefs = int(input())
    if choicefs == 1:
        dataset = load_fulldata(processfunc)
    elif choicefs == 2:
        p = '/home/al/PythonFiles/files/disser/readydata/morpho/'
        filenames = [name for name in os.listdir(p) if not os.path.splitext(name)[1] or os.path.isdir(name)]
        print('Your fighters are as follows:')
        print(*filenames)
        fighter = input('Choose your fighter: ')
        dataset = load_smalldata(fighter, processfunc)
    else:
        sys.exit('U dumbass')  # why did I do that?..

    print('"""datasets loaded and processed"""')

    ''' Create model '''
    id2word = corpora.Dictionary(dataset)
    print('"""corpus dictionary created"""')
    corpus = [id2word.doc2bow(text) for text in dataset]
    print('"""corpus gathered"""')
    print('"""computing topics"""')
    while True:
        tops = int(input('How many topics? print 0 to kill process\n'))
        if not tops:
            break
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
        lda_model = {}
        for term in topics_df.itertuples():
            for i in range(1, len(term)):
                lda_model[term[i]] = i
        print('"""terms annotated. saving and printing..."""')

        ''' Saving annotation '''
        namemodel = f'{choiceprocess}_{tops}'
        print(f'Saving as {namemodel}. There were {tops} topics.')
        pickle.dump(lda_model, open(f'/home/al/PythonFiles/files/disser/LDAs/{namemodel}', 'wb'))
    print('"""finished, exit"""')


if __name__ == '__main__':
    main()
