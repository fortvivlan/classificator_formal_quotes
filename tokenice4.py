#################################
##
#  Токенизатор версия 0.4.Unstable
##
#################################
import re
import pickle
from time import time


class Token:
    """Form: a representation of the token.
    Category: awailable categories must be as follows:

    word: a 'word'
    punct: punctuation mark
    emoji: emoji or smiley"""
    def __init__(self, form: str, *, category: str):
        """Form: a representation of the token.
        Category: 'word', 'punct' or 'emoji' allowed"""
        self.form = form
        self.category = category

    def __str__(self):
        return f'{self.form} : {self.category}'

    def __repr__(self):
        return f"Token('{self.form}', category='{self.category}')"


class Tokenice:
    """Main wrapper"""
    def __init__(self):
        self.abbrev = pickle.load(open('Data/abbrev', 'rb'))
        self.defis = pickle.load(open('Data/defis', 'rb'))
        self.doubledef = {'-ан-дер-', '-апон-', '-д’', '-да-', '-де-', '-дель-', '-ди-', '-ду-', '-дус-', '-душ-',
                          '-дю-', '-иль-', '-ин-', '-ла-', '-мин-', '-на-', '-над-', '-о-', '-он-', '-оф-', '-сюр-',
                          '-уш-', '-эз-', '-эль-', '-эс-', '-эш-', '-на-'}
        self.startdef = pickle.load(open('Data/startswith', 'rb'))
        self.enddef = pickle.load(open('Data/endswith', 'rb'))
        self.result = []

    def __iter__(self):
        for elem in self.result:
            yield elem
        self.result.clear()  # Kill me please

    def __str__(self):
        if len(self.result) <= 10:
            return self.result
        else:
            return f'[{self.result[0]}, {self.result[1]}, ... {self.result[-1]}]'

    def _abbr(self, word: str) -> bool:
        """Abbreviation check. Abbr list set in the function for the sake of performance"""
        if word.lower() in self.abbrev:
            return True
        return False

    def _defile(self, word: str) -> list:
        """Hyphenation check function. Receives a string (left token) and checks if it should be split"""
        tokenl = word.lower()
        # List check: the heaviest work of all, methinks
        if tokenl in self.defis:
            return [Token(word, category='word')]
        if tokenl[0].isalpha():
            for elem in self.startdef:
                if tokenl.startswith(elem):
                    return [Token(word, category='word')]
        if tokenl[1].isalpha():
            for elem in self.enddef:
                if tokenl.endswith(elem):
                    return [Token(word, category='word')]
        for elem in self.doubledef:
            if elem in tokenl:
                return [Token(word, category='word')]
        # Token not found in any list, good: we split it (or not)
        res = []
        matches = re.finditer('-', word)  # The algorythm is the same
        start = 0
        for match in matches:
            chunk = word[start:match.start()]
            if chunk:
                res.append(Token(chunk, category='word'))
            res.append(Token(match.group(0), category='punct'))
            start = match.end()
            if not re.search('-', word[start:]):
                if word[start:]:
                    res.append(Token(word[start:], category='word'))
        # Even then we might not want to split the token: if it's a number or an abbreviation with a digit
        if res:
            if len([char for char in res if (char.form.isdigit() or char == '-')]) == len(res) \
                    and res[0] != '-' and res[-1] != '-':  # We don't split numbers 11-22-33
                return [Token(word, category='word')]
            if res[0].form.isdigit() and len(res) > 2:  # Cases like 1-ый
                if res[2].form.isalpha():
                    return [Token(word, category='word')]
        return res

    def _liner(self, string: str):
        """Internal tokeniser which returns a list of tokens in a string"""
        if string.isspace():  # if a string is empty or consists of spaces, returns empty list
            return []
        punct = r'[.?!]+'
        twos = r'[:;]'
        smiles = r'[=:;]-?[)(]{1,}|\){2,}|\({2,}|\({1,3}:|[=:;]\*+'  # smileys like :-), :), (:, :*, :-*
        other = r'[©$@%№\^&*+=\\/<>~]+|[\U000000A1-\U000000BF]+'
        singles = r'[,…—–\s\'"\{\[\(\}\]\)\|«“‘»”’„]'
        wikibrackets = r'\[[0-9]+\]'
        delimiters = f'{smiles}|{twos}|{wikibrackets}|{singles}|{punct}|{other}'
        pattern = re.compile(delimiters)
        if not re.search(pattern, string):  # if no delimiters, we return a string as a whole 'word'
            # This is a check if there's one string consisting of smth like 123abc. We split it
            check = re.findall(r'^(\d+)([^\d]+)$', string)
            if check:
                return [Token(check[0][0], category='word'), Token(check[0][1], category='word')]
            return [Token(string, category='word')]
        result = []
        matches = re.finditer(pattern, string)  # an iterator with delimiters as matches is created
        start = 0  # starting point created
        joined = False  # must be True if elements are already added to result
        addedlt = False  # True if left token is added but delim is not
        bulletdistance = 0  # for bullets
        joinstring = []  # for possibly merged tokens
        closebracket = 0

        # main parsing cycle
        for match in matches:
            left_token = string[start:match.start()]  # left token is a 'word' left side of the delimiter
            delimit = match.group(0)
            if delimit.strip():  # check if delim is not space
                delimiter = delimit.strip()  # we cut off spaces
            else:
                delimiter = delimit  # if delim is a space we leave it as is. Thus, a space will be a ' '
            start = match.end()  # move start to next point

            if delimiter == '(':
                closebracket += 1

            """A bullet is considered everything \\d[).] in the beginning af the string"""
            if not result and left_token.isdigit() and not joinstring:
                if delimiter in ').' and string[start:start + 2]:
                    if string[start:].isspace():
                        # bulleting is considered punctuation
                        result.append(Token(left_token + delimiter, category='punct'))
                        joined = True
                        bulletdistance = 1

            """If there is a bullet like 1), we check whether the current delim is a . mark"""
            if bulletdistance == 1 and delimiter == '.':  # left token, btw, in this case is empty by default
                if result[-1].form[-1] != '.':
                    result[-1].form += delimiter
                    bulletdistance += 1
                    joined = True
            elif bulletdistance != 0:
                bulletdistance += 1

            """if joinstring not empty and left token is not a digit, then we append it to the list"""
            if joinstring:
                if not left_token.isdigit():
                    if len(joinstring) <= 2:
                        result.extend(joinstring)
                    joinstring = []

            """A general check if left token is not empty, but result is"""
            if left_token and not joined:
                # A check for emoji with Dd3Зз*
                if result:
                    if result[-1].form in ':;':
                        if re.search(r'^[Dd3Зз*]+$', left_token):
                            eyes = result[-1].form
                            result[-1] = Token(eyes + left_token, category='emoji')
                            addedlt = True

                check = re.findall(r'^(\d+)([^\d]+)$', left_token)  # to split tokens of a kind 123руб
                if check:
                    if check[0][1].isalpha() or check[0][1] in '€$₽':  # if second part is a letter or a cash symbol
                        result.append(Token(check[0][0], category='word'))  # adding each to result
                        result.append(Token(check[0][1], category='word'))
                        addedlt = True
                    if delimiter == '.' and result:  # also if delim is a point we merge it
                        if result[-1].form.isalpha():
                            result[-1].form += delimiter
                            joined = True

                """For merging dates and like: XX:XX:XX, considering .: and slashes"""
                if left_token.isdigit():  # left token is a digit, delim is in our p.marks, we merge it into joinstring
                    if delimiter and delimiter in '.:\\/':
                        joinstring.append(Token(left_token, category='word'))
                        joinstring.append(Token(delimiter, category='punct'))
                        joined = True
                    elif joinstring:  # if delim is not in our marks we extend result with it and empty joinstring
                        result.append(Token(''.join([x.form for x in joinstring]) + left_token, category='word'))
                        joinstring = []
                        addedlt = True

                """If delim is punct we check abbreviations"""
                if delimiter == '.' and not left_token.isdigit():
                    if len(left_token) == 1:  # A lone letter with a point is merged by default.
                        if result:
                            # т.д. and т.п. is merged, too. Well, there may be discrepancies o_O
                            if result[-1].form == 'т.':
                                result[-1].form += (left_token + delimiter)
                                joined = True
                        if not joined:
                            result.append(Token(left_token + delimiter, category='word'))
                            joined = True
                    elif self._abbr(left_token):  # check in set with abbr
                        result.append(Token(left_token + delimiter, category='word'))
                        joined = True

                """Hyphen check"""
                if '-' in left_token and len(left_token) > 1:
                    # defile returns a list of class Token instances so extend should be fine
                    result.extend(self._defile(left_token))
                    addedlt = True

                """If previous checks returned none we simply add left token as a 'word'"""
                if not joined and not addedlt:
                    if left_token != '-':
                        result.append(Token(left_token, category='word'))
                    else:
                        result.append(Token(left_token, category='punct'))

            """Is delim is no space and has not been added, we add it depending on its type"""
            if not delimiter.isspace():
                if not joined:
                    if re.search(smiles, delimiter):
                        if re.search(r'^\)+$', delimiter) and len(delimiter) == closebracket:
                            for elem in delimiter:
                                result.append(Token(elem, category='punct'))
                        else:
                            result.append(Token(delimiter, category='emoji'))
                    else:
                        result.append(Token(delimiter, category='punct'))

            # set bools to default
            joined = False
            addedlt = False
            if bulletdistance != 1:
                bulletdistance = 0

            """Tail check"""
        else:
            if joinstring:  # joinstring not empty
                if len(joinstring) <= 2 and not string[start:].isdigit():
                    result.extend(joinstring)
                else:
                    if string[start:].isdigit():
                        result.append(Token(''.join([x.form for x in joinstring]) + string[start:], category='word'))
                        joined = True
                    else:
                        result.append(Token(''.join([x.form for x in joinstring]), category='word'))
            if string[start:] and not joined:
                # check for 123abc
                check = re.findall(r'^([\d]+)([^\d]+)$', string[start:])
                if check:
                    result.append(Token(check[0][0], category='word'))
                    result.append(Token(check[0][1], category='word'))
                else:
                    result.append(Token(string[start:], category='word'))
        return result

    def tokenicer(self, obj):
        """Main function which is called for the segmenter to work"""
        if isinstance(obj, str):
            chunk = self._liner(obj)
            if chunk:
                self.result.extend(chunk)
        else:
            for line in obj:
                if line.strip():
                    chunk = self._liner(line.strip())
                    if chunk:
                        self.result.extend(chunk)


# tokenize = Tokenice()
# fle = open('/home/al/Documents/PythonFiles/files/disser/sample_10000.txt', 'r', encoding='utf8')
# t = time()
# tokenize.tokenicer(fle)
# print(f'Time passed: {time() - t}, OMFG you are slow!')
# fle.close()
# with open('/home/al/Documents/PythonFiles/files/disser/sampletoken.txt', 'w', encoding='utf8') as fil:
#     print(*list(tokenize), sep='\n', file=fil)
# tokenize.tokenicer(input())
# print(*list(tokenize))
