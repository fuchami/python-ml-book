# coding:utf-8

#%%
import os
import sys
import tarfile
import time
import numpy as np


source = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
target = 'aclImdb_v1.tar.gz'


def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = progress_size / (1024.**2 * duration)
    percent = count * block_size * 100. / total_size
    sys.stdout.write("\r%d%% | %d MB | %.2f MB/s | %d sec elapsed" %
                    (percent, progress_size / (1024.**2), speed, duration))
    sys.stdout.flush()


if not os.path.isdir('aclImdb') and not os.path.isfile('aclImdb_v1.tar.gz'):
    
    if (sys.version_info < (3, 0)):
        import urllib
        urllib.urlretrieve(source, target, reporthook)
    
    else:
        import urllib.request
        urllib.request.urlretrieve(source, target, reporthook)

# %%
import pandas as pd
import os

# 'basepath'の値を展開した映画レビューデータセットのディレクトリに置き換える
basepath = 'aclImdb'
labels = {'pos': 1, 'neg':0 }

df = pd.DataFrame()
for s in ('test', 'train'):
    for l in ('pos', 'neg'):
        path = os.path.join(basepath, s, l)
        for file in sorted(os.listdir(path)):
            with open(os.path.join(path, file), 'r', encoding='utf-8') as infile:
                txt = infile.read()
            df = df.append([[txt, labels[l]]], ignore_index=True)

df.columns = ['review', 'sentiment']


# %%
import numpy as np
np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))

df.to_csv('movie_data.csv', index=False, encoding='utf-8')

#%%
import pandas as pd
df = pd.read_csv('movie_data.csv', encoding='utf-8')
df.columns = ['review', 'sentiment']
df.head(3)

# %%
from sklearn.feature_extraction.text import CountVectorizer
count = CountVectorizer()
docs = np.array([
    'The sun is shining',
    'The weather is sweet',
    'The sun is shining, the weather is sweet, and one and one is two'])

bag = count.fit_transform(docs)
print(count.vocabulary_)

# %%
print(bag.toarray())
print(count.get_feature_names())


# %%
from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)
np.set_printoptions(precision=2)
print(tfidf.fit_transform(count.fit_transform(docs)).toarray())


#%% 手動でtf-idf
tf_is = 3
n_docs = 3
idf_is = np.log((n_docs+1) / (3+1))
tfidf_is = tf_is * (idf_is + 1)
print(f'tf-idf of term "is" = {tfidf_is}')

#%%
tfidf = TfidfTransformer(use_idf=True, norm=None, smooth_idf=True)
raw_tfidf = tfidf.fit_transform(count.fit_transform(docs)).toarray()[-1]
raw_tfidf

#%%
l2_tfidf = raw_tfidf / np.sqrt(np.sum(raw_tfidf**2))
l2_tfidf

# %% Clean Text data
# 生データは汚い
df.loc[0, 'review'][-50:]

# %%
import re
def preprocessor(text):
    text = re.sub('<[^>]*', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)

    text = (re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', ''))
    return text


# %%
preprocessor(df.loc[0, 'review'][-50:])

# %%
preprocessor("</a>This :) is :( a test :-)!")

# %%
df['review'] = df['review'].apply(preprocessor)


# %%
def tokenizer(text):
    return text.split()

tokenizer('runners like running and thus they run')

# %%
from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()

def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]


tokenizer_porter('runners like running and thus they run')

# %%
import nltk

nltk.download('stopwords')

# %%
from nltk.corpus import stopwords

stop = stopwords.words('english')
[w for w in tokenizer_porter('runners like running and thus they run')[-10:]
    if w not in stop]

# %%
X_train = df.loc[:25000, 'review'].values
y_train = df.loc[:25000, 'sentiment'].values
X_test = df.loc[25000:, 'review'].values
y_test = df.loc[25000:, 'sentiment'].values

#%%
print(len(X_train), len(y_train), len(X_test), len(y_test))
# %%
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(strip_accents=None,
                        lowercase=False,
                        preprocessor=None)

param_grid = [{'vect__ngram_range': [(1, 1)],
                'vect__stop_words': [stop, None],
                'vect__tokenizer': [tokenizer, tokenizer_porter],
                'clf__penalty': ['l1', 'l2'],
                'clf__C': [1.0, 10.0, 100.0]},
                {'vect__ngram_range': [(1, 1)],
                'vect__stop_words': [stop, None],
                'vect__tokenizer': [tokenizer, tokenizer_porter],
                'vect__use_idf': [False],
                'vect__norm': [None],
                'clf__penalty': ['l1', 'l2'],
                'clf__C': [1.0, 10.0, 100.0]}
            ]

lr_tfidf = Pipeline([('vect', tfidf),
                    ('clf', LogisticRegression(random_state=0))])

gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid=param_grid,
                            scoring='accuracy',
                            cv=5,
                            verbose=1,
                            n_jobs=1)

#%%
gs_lr_tfidf.fit(X_train, y_train)
# %%
print('Best Parameter set: %s' % gs_lr_tfidf.best_params_)

# %%
print('CV Accuracy: %.3f' % gs_lr_tfidf.best_score_)

clf = gs_lr_tfidf.best_estimator_
print('Test Accuracy: %.3f' % clf.score(X_test, y_test))

#%% オンラインアルゴリズムとアウトオブコア学習
import numpy as np
import re
from nltk.corpus import stopwords

stop = stopwords.words('english')

def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')

    tokenized = [w for w in text.split() if w not in stop]
    return tokenized

def strem_docs(path):
    with open(path, 'r', encoding='utf-8') as csv:
        next(csv)
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label

next(strem_docs(path='movie_data.csv'))

#%% 
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


#%%
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

vect = HashingVectorizer(decode_error='ignore',
                        n_features=2**21,
                        preprocessor=None,
                        tokenizer=tokenizer)

clf = SGDClassifier(loss='log', random_state=1, max_iter=1)
doc_stream = strem_docs(path='movie_data.csv')

# %%
import pyprind
pbar = pyprind.ProgBar(45)
classes = np.array([0, 1])

for _ in range(45):
    X_train, y_train = get_minibatch(doc_stream, size=1000)
    if not X_train:
        break
    X_train = vect.transform(X_train)
    clf.partial_fit(X_train, y_train, classes=classes)
    pbar.update()


# %%
X_test, y_test = get_minibatch(doc_stream, size=5000)
X_test = vect.transform(X_test)
print('Accuracy: %.3f' % clf.score(X_test, y_test))

# %%
clf = clf.partial_fit(X_test, y_test)

# %% 8.4 潜在ディリクレ配分によるトピックモデルの構築
import pandas as pd

df = pd.read_csv('movie_data.csv', encoding='utf-8')
df.head(3)


# %%
from sklearn.feature_extraction.text import CountVectorizer
count = CountVectorizer(stop_words='english', 
                        max_df=.1,
                        max_features=5000)
X = count.fit_transform(df['review'].values)

# %%
from sklearn.decomposition import LatentDirichletAllocation
lda = LatentDirichletAllocation(n_components=10,
                                random_state=123,
                                learning_method='batch')
X_topics = lda.fit_transform(X)

# %%
lda.components_.shape

#%%
n_top_words = 5
feature_name = count.get_feature_names()
for topic_idx, topic in enumerate(lda.components_):
    print("Topic %d" % (topic_idx +1))
    print(" ".join([feature_name[i]
                    for i in topic.argsort() [:-n_top_words -1 : -1]]))


# %%
horror = X_topics[:, 5].argsort()[::-1]
for iter_idx, movie_idx in enumerate(horror[:3]):
    print('\nHorror movie #%d: '% (iter_idx +1))
    print(df['review'][movie_idx][:300], '...')

# %%
