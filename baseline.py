import pickle
import re
from morphoclass import MorphoToken


def segmenter(dataset):
    cpattern = r'\(c\)|\U000000A9|\(C\)|\(с\)|\(С\)|цитата|цитировать'
    quotepattern = '["«“‘”’„][^"«»“‘”’„]{15,}["“‘»”’„]'
    lonequote = r'["«»“‘”’„]'
    resultingindexes = []
    for doc in dataset:
        inds = []
        for i in range(len(doc)):
            sent = ''.join([token.lemma for token in doc[i]])
            if re.search(cpattern, sent) or re.search(quotepattern, sent) or len(re.findall(lonequote, sent)) == 1:
                if i < len(doc) - 1:
                    inds.append(i + 1)
        resultingindexes.append(inds)
    return resultingindexes


def splitter(doc, indexes):
    segmented = []
    starter = 0
    for index in indexes:
        if index != starter:
            segmented.append(doc[starter:index])
        starter = index
    if doc[starter:]:  # considers tail
        segmented.append(doc[starter:])
    return segmented


def main():
    p = '/home/al/PythonFiles/files/disser/readydata/morpho/PARTIAL_LJ'
    data = pickle.load(open(p, 'rb'))[:500]
    breakpoints = segmenter(data)
    bdict = {}
    for i in range(len(breakpoints)):
        bdict[i] = breakpoints[i]
    pickle.dump(bdict, open('/home/al/PythonFiles/files/disser/experiments/breakpoints_BASE', 'wb'))
    result = []
    for i in range(len(data)):
        document = []
        for sent in data[i]:
            document.append([token.form for token in sent])
        result.append(splitter(document, breakpoints[i]))
    with open('/home/al/PythonFiles/files/disser/experiments/BASE.txt', 'w', encoding='utf8') as txt:
        for i in range(len(result)):
            for sent in data[i]:
                print(*[token.form for token in sent], file=txt)
            print('\n\n~~~~~~\n\n', file=txt)
            print(breakpoints, file=txt)
            if len(result[i]) > 1:
                for part in result[i]:
                    for sent in part:
                        print(*sent, file=txt)
                    print('@@@@@@SEG@@@@@@', file=txt)
            else:
                print('NO BREAKPOINTS FOUND', file=txt)
            print('\n\n~~~~~~\n\n', file=txt)


if __name__ == '__main__':
    main()
