# -*- coding: utf-8 -*-
# Import Keras libraries and packages
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.model_selection import GridSearchCV

def CLF(text_clean):
    parameters = {
        'bow__ngram_range': [(1, 1), (1, 2)],
        'tfidf__use_idf': (True, False),
        'classifier__alpha': (1e-2, 1e-3),
    }
    pipeline_clf=Pipeline([
        ('bow', CountVectorizer(analyzer=text_clean)),
        ('tfidf', TfidfTransformer()),
        ('classifier', MultinomialNB())
        ])
    gs_clf = GridSearchCV(pipeline_clf, parameters, cv=7)
    return gs_clf