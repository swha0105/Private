#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'week 3'))
	print(os.getcwd())
except:
	pass
#%% [markdown]
# ## Prerequisites

#%%
random_state = 42
random_state

#%% [markdown]
# ## Otto Group Product Classification Challenge
# 
# 이번 과제는 세계 최대의 전자상거래 회사 중 하나인 [Otto Group](https://www.ottogroup.com/)에서 주최하는 [Otto Group Product Classification Challenge](https://www.kaggle.com/c/otto-group-product-classification-challenge/) 경진대회에 참석해보겠습니다.
# 
# Otto Group은 익명화(anonymization)된 상품 정보에 대한 데이터를 제공하는데, 경진대회 참석자는 이 데이터를 활용하여 주어진 상품 카테고리(target)를 예측해야 합니다. 상품 카테고리는 Class_1부터 Class_9까지 총 9개가 있습니다. 주어진 데이터를 Decision Tree, Random Forest, 그리고 Gradient Boosting Machine를 활용하여 예측해보도록 하겠습니다.
# 
# 
# 

#%%
import pandas as pd

data = pd.read_csv("~/data/Dsschool/train.csv", index_col="id")

print(data.shape)
data.head()

#%% [markdown]
# ## Preprocessing

#%%
label_name = "target"
label_name


#%%
feature_names = data.columns.difference([label_name])

print(len(feature_names))
feature_names


#%%
X = data[feature_names]

print(X.shape)
X.head()


#%%
y = data[label_name]

print(y.unique())

print(y.shape)
y.head()

#%% [markdown]
# ### Benchmark
#%% [markdown]
# ### Hold-Out Validation

#%%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


#%%
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)

y_train_predict = model.predict(X_train)
y_test_predict = model.predict(X_test)

train_score = (y_train_predict == y_train).mean()
test_score = (y_test_predict == y_test).mean()

print(f"Accuracy(train) = {train_score:.6f}")
print(f"Accuracy(test) = {test_score:.6f}")

#%% [markdown]
# ### Hyperparameter
#%% [markdown]
# **n_estimators**

#%%
from sklearn.ensemble import RandomForestClassifier

n_estimators_list = list(range(30, 301, 30))

history = []

for n_estimators in n_estimators_list:
    model = RandomForestClassifier(n_estimators = n_estimators,
                                   random_state=42)

    model.fit(X_train, y_train)

    y_train_predict = model.predict(X_train)
    y_test_predict = model.predict(X_test)

    train_score = (y_train_predict == y_train).mean()
    test_score = (y_test_predict == y_test).mean()
    
    print(f"n_estimators = {n_estimators}, train = {train_score:.6f} test = {test_score:.6f}")
    
    history.append({
        'n_estimators': n_estimators,
        'accuracy(train)': train_score,
        'accuracy(test)': test_score,
    })

history = pd.DataFrame(history)
history.head()


#%%
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

plt.plot(history['n_estimators'], history['accuracy(train)'], label='accuracy(train)')
plt.plot(history['n_estimators'], history['accuracy(test)'], label='accuracy(test)')

# plt.ylim(0.0, 1.0)

plt.legend()

#%% [markdown]
# **max_features**

#%%
from sklearn.ensemble import RandomForestClassifier

max_features_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

history = []

for max_features in max_features_list:
    model = RandomForestClassifier(n_estimators = 100,
                                   max_features = max_features,
                                   random_state=42)

    model.fit(X_train, y_train)

    y_train_predict = model.predict(X_train)
    y_test_predict = model.predict(X_test)

    train_score = (y_train_predict == y_train).mean()
    test_score = (y_test_predict == y_test).mean()
    
    print(f"max_features = {max_features}, train = {train_score:.6f} test = {test_score:.6f}")
    
    history.append({
        'max_features': max_features,
        'accuracy(train)': train_score,
        'accuracy(test)': test_score,
    })

history = pd.DataFrame(history)
history.head()


#%%
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

plt.plot(history['max_features'], history['accuracy(train)'], label='accuracy(train)')
plt.plot(history['max_features'], history['accuracy(test)'], label='accuracy(test)')

# plt.ylim(0.0, 1.0)

plt.legend()

#%% [markdown]
# **class_weight**
### imblance dataset
#%%
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss

model = RandomForestClassifier(n_estimators = 100,
                               class_weight = None,
                               random_state = 42)
model.fit(X_train, y_train)

y_train_predict = model.predict_proba(X_train)
y_test_predict = model.predict_proba(X_test)

train_score = log_loss(y_train, y_train_predict)
test_score = log_loss(y_test, y_test_predict)

print(f"class weight(None): train = {train_score:.6f} test = {test_score:.6f}")

model = RandomForestClassifier(n_estimators = 100,
                               class_weight = 'balanced',
                               random_state = 42)
model.fit(X_train, y_train)

y_train_predict = model.predict_proba(X_train)
y_test_predict = model.predict_proba(X_test)

train_score = log_loss(y_train, y_train_predict)
test_score = log_loss(y_test, y_test_predict)

print(f"class weight(balanced): train = {train_score:.6f} test = {test_score:.6f}")

#%% [markdown]
# ### Cross Validation

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict

RandomForestClassifier(n_estimators=10 ,random_state=42)

y_predict = cross_val_predict(model, X, y, cv= 5)

accuracy = (y == y_predict).mean()

print(f"accuracy = {accuracy:.6f}")

#%%

from sklearn.model_selection import cross_val_score

cross_val_score(model, X, y, cv = 5, scoring = "accuracy")

print(f"accuracy = {accuracy:.6f}")
#%%
# Write your code here!

#%% [markdown]
# ### Log Loss

import numpy as np
import matplotlib.pyplot as plt

xx = np.linspace(0, 1)
yy = -np.log(xx)

plt.plot(xx,yy)

#%%
# Write your code here!
## 의류 가전 의약
y = np.array([0, 0, 1])
p = np.array([0.2, 0.1, 0.7])

-1.0 * 

#%%


