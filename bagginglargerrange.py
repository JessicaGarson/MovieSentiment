import pandas as pd
import numpy as np
from ggplot import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import auc_score

train = pd.read_csv('/Users/jessicagarson/Downloads/Movie Reviews/train.csv')
test = pd.read_csv('/Users/jessicagarson/Downloads/Movie Reviews/test.csv')

def bagmodel(s):
    vectorizer = CountVectorizer()
    X_dict = vectorizer.fit_transform(s.Phrase)
    choices = np.random.choice(range(len(s)), len(s), replace = True)
    s = s.ix[choices,:]
    X_train = vectorizer.transform(s.Phrase)
    model = LogisticRegression().fit(X_train, list(s.Sentiment))
    return model

models = []
for i in range(100):
    print i
    models.append(bagmodel(train))

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train.Phrase)
X_test = vectorizer.transform(test.Phrase)


# results = [x.predict(X_test) for x in models
result = [x.predict(X_test) for x in models]


from collections import Counter

def combination(s):
    thing = Counter(s)
    return thing.most_common(1)[0]

combination([3,3,2,3,3,])

result_final = []

for i in range(len(test)):
    a, b = combination([x[i] for x in result])
    result_final.append(a)

result_final[0]

solution = pd.DataFrame({'PhraseId': test.PhraseId, 'Sentiment': result_final})
solution.to_csv('submissionbaggedagain.csv', index=False)

plotout = ggplot(aes(x = 'Sentiment'), data=solution)
plotout + geom_histogram()