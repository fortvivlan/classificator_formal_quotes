import re
import json
import pandas
# import pymystem3
#
# mystem = pymystem3.Mystem(entire_input=False)

quotepattern = '["«“‘”’„][^"«»“‘”’„]{15,}["“‘»”’„]'
cpattern = r'\(c\)|\U000000A9|\(C\)|\(с\)|\(С\)'
dialoguepattern = r'(?m)^(?: ?[-—–]|\w+?:)[\s\w]{2}(?!=\/\/)'
saidpattern = r'(?:сказал|говорил|говорит|спросил|спрашивает|пишет|писал).{5,}:'

saidlist = {'сказать', 'говорить', 'спрашивать', 'писать'}
saidlist2 = {'сказал', 'говорил', 'говорит', 'спросил', 'спрашивает', 'пишет', 'писал'}
prediction = 0
verdict = ''
dialog = False
i = 0

# with open(r'F:/Python/files/disser/partial_author.json', 'r', encoding='utf8') as jsonfile:
#     massive = json.load(jsonfile)

with open(r'F:/Python/files/disser/AC30_1000.tsv', newline='', encoding='utf8') as tsvfile:
    dataset = pandas.read_csv(tsvfile, sep='\t')
    massive = dataset['text'].to_dict()

res = []

for key in massive:
    if re.search(quotepattern, massive[key]):
        prediction += 1
        verdict += 'quote '
    if re.search(cpattern, massive[key]):
        prediction += 1
        verdict += 'copyright '
    dialogue = re.finditer(dialoguepattern, massive[key])
    if dialogue:
        for match in dialogue:
            prediction += 0.2
            dialog = True
    if dialog:
        verdict += 'dialogue '
    if re.search(saidpattern, massive[key].lower()):
        prediction += 0.4
        verdict += 'saidquote '
    if re.search('repost|regram', massive[key].lower()):
        prediction += 2
        verdict += 'repost '
    # lemmas = set(mystem.lemmatize(' '.join(elem['text'].split('\n'))))
    # for word in saidlist:
    #     if word in lemmas and ':' in elem['text']:
    #         prediction += 0.3
    # for word in saidlist2:
    #     if word in elem['text'] and ':' in elem['text']:
    #         prediction += 0.3

    if prediction == 0:
        verdict = 'NFM'
    res.append({})
    res[i]['text'] = massive[key]
    res[i]['prediction'] = prediction
    res[i]['verdict'] = verdict.strip()
    # print(elem['text'], prediction, verdict, sep='\n', file=dump)
    prediction = 0
    verdict = ''
    dialog = False
    i += 1

with open(r'F:/Python/files/disser/result30.json', 'w', encoding='utf8') as dumpfile:
    json.dump(res, dumpfile, ensure_ascii=False, indent=4)
