# coding:utf-8

#%% サンプルデータの作成
import pandas as pd
from io import StringIO

csv_data = \
'''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,'''

df = pd.read_csv(StringIO(csv_data))
df

# %%
# 欠損値をカウント
df.isnull().sum()

# %% 欠損値を含む行を削除
df.dropna()


# %% 欠損値を含む列を削除
df.dropna(axis=1)


# %% すべての列がNanである行だけ削除
df.dropna(how='all')


# %% 非Nan値が4つ未満の行を削除
df.dropna(thresh=4)


# %% 特定の列にNanが含まれてリウ行だけを削除
df.dropna(subset=['C'])


# %% 欠損値を平均補完する
from sklearn.preprocessing import Imputer

imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
imr = imr.fit(df.values)
# 補完を実行
imputed_data = imr.transform(df.values)
imputed_data

# %% Categoricalデータの処理

import pandas as pd
# サンプルデータの生成
df = pd.DataFrame([
    ['green', 'M', 10.1, 'class1'],
    ['red', 'L', 13.5, 'class2'],
    ['blue', 'XL', 15.3, 'class1']])

df.columns = ['color', 'size', 'price', 'classlabel']
df

# %%
# Tシャツのサイズと整数を対応させるDictを作成
size_mapping = { 'XL': 3, 'L': 2, 'M': 1}
# Tシャツのサイズを整数に変換
df['size'] = df['size'].map(size_mapping)
df

# %%
inv_size_mapping = {v: k for k, v in size_mapping.items()}
print('inv_size_mapping:', inv_size_mapping)
df['size'].map(inv_size_mapping)

# %% クラスラベルのエンコーディング
import numpy as np
# クラスラベルと整数を対応させるディクショナリを生成
class_mapping = {label:idx for idx, label in enumerate(np.unique(df['classlabel']))}
class_mapping

# %% クラスラベルを整数に変換
df['classlabel'] = df['classlabel'].map(class_mapping)
df

# %% 元の文字表現に戻す
inv_class_mapping = {v: k for k, v in class_mapping.items()}
# 整数からクラスラベルに変換
df['classlabel'] = df['classlabel'].map(inv_class_mapping)
df

# %% sklearnのLabelEncoderでも同様の処理が可能
from sklearn.preprocessing import LabelEncoder
class_le = LabelEncoder()
y = class_le.fit_transform(df['classlabel'].values)
y

# %% クラスラベルを文字列に戻す
class_le.inverse_transform(y)

# %% Tシャツの色、サイズ、価格を抽出
X = df[['color','size','price']].values
print(X)
color_le = LabelEncoder()
X[:, 0] = color_le.fit_transform(X[:, 0])
X

# %% 上記のやり方だと順序のないcolorに順序の概念を加えてしまう
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(categorical_features=[0])
ohe.fit_transform(X).toarray()

# %%
pd.get_dummies(df[['price','color','size']])


# %%
# 多重共線性の問題から、特徴量の列を1つ削除
pd.get_dummies(df[['price','color','size']], drop_first=True)

# %%
ohe = OneHotEncoder(categorical_features=[0])
ohe.fit_transform(X).toarray()[:, 1:]

# %% datasetをtrainとtestに分割する
# wineデータセットを読み込む
df_wine = pd.read_csv('https://archive.ics.uci.edu/'
                    'ml/machine-learning-databases/wine/wine.data',
                    header=None)
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 
                    'Alcalinity of ash', 'Magnesium', 'Total phenols',
                    'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 
                    'Color intensity', 'Hue', 
                    'OD280/OD315 of diluted wines', 'Proline']
print('Class labels', np.unique(df_wine['Class label']))
df_wine.head()

# %%
from sklearn.model_selection import train_test_split
# 特徴量とラベルを別々に抽出
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
# トレーニングデータとテストデータに分割
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)


# %% スケーリングのお話
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.transform(X_test)

# %%
ex = np.array([0,1,2,3,4,5])
print('standardized:', (ex-ex.mean()) /ex.std() )
print('normalized:', (ex -ex.min())/(ex.max() - ex.min()))


# %%
from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)


# %%
from sklearn.linear_model import LogisticRegression
LogisticRegression(penalty='l1')

# %%
lr = LogisticRegression(penalty='l1', C=1.0)
lr.fit(X_train_std, y_train)

print("Training Accuracy:", lr.score(X_train_std, y_train))
print("Test Accuracy:", lr.score(X_test_std, y_test))

# %% 切片の表示
lr.intercept_


# %% 重み係数の表示
lr.coef_

# %%
import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.subplot(111)

colors = ['blue', 'green', 'red', 'cyan', 
        'magenta', 'yellow', 'black', 
        'pink', 'lightgreen', 'lightblue', 
        'gray', 'indigo', 'orange']

weights, params = [], []
for c in np.arange(-4., 6.):
    lr = LogisticRegression(penalty='l1', C=10.**c, random_state=0)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10**c)

weights = np.array(weights)
# 各重み係数をプロット
for column, color in zip(range(weights.shape[1]), colors):
    # 横軸を逆正則化パラメータ、縦軸を重み係数とした折れ線グラフ
    plt.plot(params, weights[:, column], label=df_wine.columns[column+1], color=color)

plt.axhline(0, color='black', linestyle='--', linewidth=3)
plt.xlim([10**(-5), 10**5])
plt.ylabel('weight coefficient')
plt.xlabel('C')

plt.xscale('log')
plt.legend(loc='upper left')
ax.legend(loc='upper center', bbox_to_anchor=(1.38, 1.03), ncol=1, fancybox=True)
plt.show()

# %% Sequential Backward Selection:SBS
from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score

class SBS():
    """
    逐次後退選択(Sequential backward selection)を実行するクラス
    """

    def __init__(self, estimator, k_features, scoring=accuracy_score,
                    test_size=0.25, random_state=1):
        self.scoring = scoring # 特徴量の評価指標
        self.estimator = estimator # 推定器
        self.k_features = k_features 
        self.test_size = test_size
        self.random_state = random_state
    
    def fit(self, X, y):
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=self.test_size,
                                random_state=self.random_state)

        # すべての特徴量の個数、列インデックス
        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        # すべての特徴量を用いてスコア算出
        score = self._calc_score(X_train, y_train, X_test, y_test, self.indices_)

        self.scores_ = [score]
        # 指定した特徴量の個数になるまで反復
        while dim > self.k_features:
            scores = [] # スコア
            subsets = [] # 列インデックス

            # 特徴量の部分集合を表す列インデックスの組み合わせごとに反復
            for p in combinations(self.indices_, r=dim -1):
                score = self._calc_score(X_train, y_train, X_test, y_test, p)

                scores.append(score)
                # 特徴量の部分集合を表す列インデックスのリストを格納
                subsets.append(p)
            
            # 最良のスコアのインデックスを抽出
            best = np.argmax(scores)
            # 最良となる列インデックスを抽出して格納
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            # 次元数を減らして次のステップへ
            dim -= 1

            # スコアを格納
            self.scores_.append(scores[best])
        
        # 最後に格納したスコア
        self.k_score_ = self.scores_[-1]

        return self
    
    def transform(self, X):
        # 抽出した特徴量を返す
        return X[:, self.indices_]

    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        # 指定された列番号indicesの特徴量を抽出してモデルを適合
        self.estimator.fit(X_train[:, indices], y_train)
        # テストデータを用いてクラスラベルを予測
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score


# %%
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

knn = KNeighborsClassifier(n_neighbors=5)
# 特徴量の個数が1になるまで特徴量を選択
sbs = SBS(knn, k_features=1)

sbs.fit(X_train_std, y_train)

# 特徴量の個数のリスト
k_feat = [len(k) for k in sbs.subsets_]

# 横軸を特徴量の個数、縦軸をスコアとしたグラフ
plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.7, 1.02])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.tight_layout()
plt.show()


# %% 最小限の特徴部分集合(k=3)を見てみる
k3 = list(sbs.subsets_[10])
print(df_wine.columns[1:][k3])


# %%
# 13すべての特徴量を用いてモデルを適合
knn.fit(X_train_std, y_train)

print('Trainig accuracy:', knn.score(X_train_std, y_train))
print('Test accuracy:', knn.score(X_test_std, y_test))


# %%
# 3つの特徴量でモデルを適合
knn.fit(X_train_std[:, k3], y_train)
print('Trainig accuracy:', knn.score(X_train_std[:, k3], y_train))
print('Test accuracy:', knn.score(X_test_std[:, k3], y_test))


# %% feature importance with Random Forest
from sklearn.ensemble import RandomForestClassifier
# Wineデータセットの特徴量の名称
feat_labels = df_wine.columns[1:]

forest = RandomForestClassifier(n_estimators=500, random_state=1)
forest.fit(X_train, y_train)

importances = forest.feature_importances_
# 重要度の降順で特徴量のインデックスを取得
indices = np.argsort(importances)[::-1]

for f in range(X_train.shape[1]):
    print("%2d %-*s %f" % 
                (f+1, 30, feat_labels[indices[f]], importances[indices[f]]))

plt.title("featue importances")
plt.bar(range(X_train.shape[1]), importances[indices], align='center')
plt.xticks(range(X_train.shape[1]), feat_labels[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()


# %%
from sklearn.feature_selection import SelectFromModel
# 特徴量選択オブジェクトを生成(重要度の閾値を0.1に)
sfm = SelectFromModel(forest, threshold=0.1, prefit=True)

X_selected = sfm.transform(X_train)
print('Number of features that meet this threshold criterion:', X_selected.shape[1])

for f in range(X_selected.shape[1]):
    print("%2d %-*s %f" % 
                (f+1, 30, feat_labels[indices[f]], importances[indices[f]]))

# %%
