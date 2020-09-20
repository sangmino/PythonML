import os
import sys
import tarfile
import time
import urllib.request
import pyprind
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import re
from nltk.stem.porter import PorterStemmer
import nltk
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import gzip
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from distutils.version import LooseVersion as Version
from sklearn import __version__ as sklearn_version
from sklearn.decomposition import LatentDirichletAllocation




#--------------------------------------------------------------------#
# region Preprocessing the movie dataset into more convenient format

basepath = '../Data/aclImdb'

labels = {'pos': 1, 'neg': 0}
pbar = pyprind.ProgBar(50000)
df = pd.DataFrame()
for s in ('test', 'train'):
    for l in ('pos', 'neg'):
        path = os.path.join(basepath, s, l)
        for file in sorted(os.listdir(path)):
            with open(os.path.join(path, file), 
                      'r', encoding='utf-8') as infile:
                txt = infile.read()
            df = df.append([[txt, labels[l]]], 
                           ignore_index=True)
            pbar.update()
df.columns = ['review', 'sentiment']


## Shuffling the DataFrame:
np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))

## Saving the assembled data as CSV file:
df.to_csv('movie_data.csv', index=False, encoding = 'utf-8')
df = pd.read_csv('movie_data.csv', encoding = 'utf-8')
print(df.head(3))
print(df.shape)

# endregion

df = pd.read_csv('movie_data.csv', encoding = 'utf-8')

#--------------------------------------------------------------------#
# region Introducing the bag-of-words model

## Transforming documents into feature vectors
count = CountVectorizer()
docs = np.array([
        'The sun is shining',
        'The weather is sweet',
        'The sun is shining, the weather is sweet, and one and one is two'])
bag = count.fit_transform(docs)


# Now let us print the contents of the vocabulary to get a better understanding of the underlying concepts:
print(count.vocabulary_)
print(bag.toarray())

## Assessing word relevancy via term frequency-inverse document frequency
np.set_printoptions(precision = 2)

## TF-IDF Transformer
tfidf = TfidfTransformer(use_idf = True, norm = 'l2', smooth_idf = True)
print(tfidf.fit_transform(count.fit_transform(docs)).toarray())

tf_is = 3
n_docs = 3
idf_is = np.log((n_docs+1) / (3+1))
tfidf_is = tf_is * (idf_is + 1)
print('tf-idf of term "is" = %.2f' % tfidf_is)

# endregion

#--------------------------------------------------------------------#
# region Cleaning text data

def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = (re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', ''))
    return text

preprocessor(df.loc[0, 'review'][-50:])
preprocessor("</a>This :) is :( a test :-)!")

# Apply the preprocessor function
df['review'] = df['review'].apply(preprocessor)

# endregion

#--------------------------------------------------------------------#
# region Processing documents into tokens

## Helper Functions
porter = PorterStemmer()
def tokenizer(text):
    return text.split()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

tokenizer('runners like running and thus they run')
tokenizer_porter('runners like running and thus they run')


## Removing Stop Words
nltk.download('stopwords')
stop = stopwords.words('english')
[w for w in tokenizer_porter('a runner likes running and runs a lot')[-10:] if w not in stop]

# endregion

#--------------------------------------------------------------------#
# region Training a logistic regression model for document classification

## Strip HTML and punctuation to speed up the GridSearch later:
X_train = df.loc[:25000, 'review'].values
y_train = df.loc[:25000, 'sentiment'].values
X_test = df.loc[25000:, 'review'].values
y_test = df.loc[25000:, 'sentiment'].values

## Fit the logistic regression model using 5-fold stratified cross-validation
tfidf = TfidfVectorizer(strip_accents = None, lowercase = False, preprocessor = None)
param_grid = [{'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]},
              {'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'vect__use_idf':[False],
               'vect__norm':[None],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]},
              ]

lr_tfidf = Pipeline([('vect', tfidf),
                     ('clf', LogisticRegression(random_state = 0, solver = 'liblinear'))])

gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid,
                           scoring='accuracy',
                           cv=5,
                           verbose=2,
                           n_jobs=-1)

## Fit the model
gs_lr_tfidf.fit(X_train, y_train)
print('Best parameter set: %s ' % gs_lr_tfidf.best_params_)
print('CV Accuracy: %.3f' % gs_lr_tfidf.best_score_)

## Using the best model, print the accuracy scores on training and test
clf = gs_lr_tfidf.best_estimator_
print('Test Accuracy: %.3f' % clf.score(X_test, y_test))

# endregion

#--------------------------------------------------------------------#
# region Working with bigger data - online algorithms and out-of-core learning

## Helper Functions

# stream_docs: reads in and returns one document at a time
def stream_docs(path):
    with open(path, 'r', encoding='utf-8') as csv:
        next(csv)  # skip header
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label

next(stream_docs(path='movie_data.csv'))

# Takes a document stream and returns a particular number of documents specified by the size parameters
def get_minibatch(doc_stream, size):
    docs, y = [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None
    return docs, y


# HashingVectorizer: data-independent
# We can't use CountVectorizer and TfidVectorizer since they have to hold the vocabulary in memory
vect = HashingVectorizer(decode_error='ignore', 
                         n_features=2**21, # <-- large # reduces the change of causing hash collisions
                         preprocessor=None, 
                         tokenizer=tokenizer)

## Out-of-core learning
clf = SGDClassifier(loss='log', random_state=1)
doc_stream = stream_docs(path='movie_data.csv')

pbar = pyprind.ProgBar(45)
classes = np.array([0, 1])
for _ in range(45):
    X_train, y_train = get_minibatch(doc_stream, size=1000)
    if not X_train:
        break
    X_train = vect.transform(X_train)
    clf.partial_fit(X_train, y_train, classes=classes)
    pbar.update()

X_test, y_test = get_minibatch(doc_stream, size=5000)
X_test = vect.transform(X_test)
print('Accuracy: %.3f' % clf.score(X_test, y_test))

# endregion

#--------------------------------------------------------------------#
# region Latent Dirichlet Allocation with scikit-learn

df = pd.read_csv('movie_data.csv', encoding='utf-8')
df.head(3)

## Create the bag-of-words matrix as input to the LDA
count = CountVectorizer(stop_words='english',
                        max_df=.1, # maximum document frequency of words (exclude too frequent words)
                        max_features=5000)
X = count.fit_transform(df['review'].values)

## LDA via Scikit-Learn
lda = LatentDirichletAllocation(n_components=10,
                                random_state=123,
                                learning_method='batch') # estimation done in one iteration
X_topics = lda.fit_transform(X)

## Print the five most important words for each of the 10 topics
n_top_words = 5
feature_names = count.get_feature_names()
for topic_idx, topic in enumerate(lda.components_):
    print("Topic %d:" % (topic_idx + 1))
    print(" ".join([feature_names[i]
                    for i in topic.argsort()\
                        [:-n_top_words - 1:-1]]))

## Confirm that the result makes sense
horror = X_topics[:, 5].argsort()[::-1]
for iter_idx, movie_idx in enumerate(horror[:3]):
    print('\nHorror movie #%d:' % (iter_idx + 1))
    print(df['review'][movie_idx][:300], '...')

# endregion