#Logistic Using Sklearn

import pandas as pd
import numpy as np
from ggplot import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import auc_score

train = pd.read_csv('/Users/jessicagarson/Downloads/Movie Reviews/train.csv')
test = pd.read_csv('/Users/jessicagarson/Downloads/Movie Reviews/test.csv')
train.head()
test.head()

p = ggplot(aes(x='Sentiment'), data = train) 
p + geom_histogram()

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train.Phrase)
X_test = vectorizer.transform(test.Phrase)

cross_val_score(LogisticRegression(), X_train, train.Sentiment)
model = LogisticRegression().fit(X_train, list(train.Sentiment))

model.predict_proba(X_test)

a = pd.DataFrame({'stuff':predict_out})

plotout = ggplot(aes(x = 'stuff'), data=a)
plotout + geom_histogram()

solution = pd.DataFrame({'PhraseId': test.PhraseId, 'Sentiment': predict_out})
solution.to_csv('submission.csv', index=False)
