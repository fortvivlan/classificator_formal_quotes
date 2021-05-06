class MorphoToken:
    """form: original form of the token
       lemma: normal form
       pos: part of speech
       category: word, emoji, punct
    """
    def __init__(self, *, form, lemma, category, pos=None):
        self.form = form
        self.lemma = lemma
        self.category = category
        self.pos = pos

    def __str__(self):
        return f'{self.form} :: lemma {self.lemma} :: POS {self.pos} :: category {self.category}'

    def __repr__(self):
        if self.pos:
            return f"MorphoToken(form='{self.form}', " \
                   f"lemma='{self.lemma}', category='{self.category}', pos='{self.pos}')"
        else:
            return f"MorphoToken(form='{self.form}', lemma='{self.lemma}', category='{self.category}')"
