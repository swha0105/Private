#%%
import pandas as pd

data = pd.read_csv('/home/ha/codes/DSschool/week 2/data/train.csv', index_col = "id")
print(data.shape)
data.head()

#%%
label_name = "target"
label_name

#%%
feature_names = data.columns.difference([label_name]) #difference 는 안에꺼 빼주는거..
feature_names

#%%

X = data[feature_names] #이거 생각 잘해보자

print(X.shape)
X.head()


#%%
Y = data[label_name]

print(Y.shape)
Y.head()


#%%

### Use Decision Tree
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(random_state = 42) #random값이 존재하는데 고정..
model

#%%
model.fit(X,Y)


#%%
Y_predict = model.predict(X)
Y_predict

#%%
accuracy = (Y == Y_predict).mean()

f"Accuracy == {accuracy:.6f}"   #f뒤에는 포맷팅 전용!! , :.Nf 은 소숫점 N까지 추령
#%%
### 위에선 통쨰로 학습 시키고 통쨰로 테스트..
####### 이제 데이터 쪼개서 ..  (Hold-out Validation)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = \
    train_test_split(X, Y, test_size = 0.3, random_state = 42) #검증용 테스트 30%  당연히 랜덤이기때문에 random_state 고정

print(X_train.shape, Y_train.shape)
print(X_test.shape,Y_test.shape)

#%%
model.fit(X_train,Y_train)

Y_train_predict = model.predict(X_train) #model.predict는 tree 그 자체를 리턴
Y_test_predict = model.predict(X_test)

train_accuracy = (Y_train == Y_train_predict).mean()
test_accuracy = (Y_test == Y_test_predict).mean()

print(f"Accuracy(Train) = {train_accuracy:.6f}")
print(f"Accuracy(Test) = {test_accuracy:.6f}")
#%%
#from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydot
import io

dot_data = io.StringIO()
#dot_data = 
export_graphviz(model,out_file='dot_data.dot',filled=True,rounded=True)
#,feature_names=feature_names)
#special_characters=True)

#from subprocess import call
#call(['dot','Tpng','dot_data.dot','-o','dot_data.png','Gdpi=600'])
#!dot -Tpng dot_data.dot -o dot_data.png -Gdpi=600 -Tsvg

!dot -Tpng dot_data.dot -o dot_data.png -Tsvg

import matplotlib.pyplot as plt
plt.figure(figsize = (5, 5))
plt.imshow(plt.imread('dot_data.png'))
plt.show()

#graph = pydot.graph_from_dot_data(dot_data.getvalue())[0]
#Image(graph.create_png())
#Image(filename = 'dot_data.png')


#%%
### Hyperparameter
model = DecisionTreeClassifier(max_depth = 20, random_state = 42) #max_depth를 제한해서 overfitting 방지

model.fit(X_train,Y_train)

Y_train_predict = model.predict(X_train)
Y_test_predict = model.predict(X_test)

train_accuracy = (Y_train == Y_train_predict).mean()
test_accuracy = (Y_test == Y_test_predict).mean()

print(f"Accuracy(Train) = {train_accuracy:.6f}")
print(f"Accuracy(Test) = {test_accuracy:.6f}")

### export graphviz 찾아보기!! 
#%%

### 다 돌려보기
max_depth_list = range(2, 51)

history = []

for max_depth in max_depth_list:
    model = DecisionTreeClassifier(max_depth = max_depth, random_state = 42) #max_depth를 제한해서 overfitting 방지

    model.fit(X_train,Y_train)

    Y_train_predict = model.predict(X_train)
    Y_test_predict = model.predict(X_test)

    train_accuracy = (Y_train == Y_train_predict).mean()
    test_accuracy = (Y_test == Y_test_predict).mean()

    print(f"Accuracy(Train) = {train_accuracy:.6f}")
    print(f"Accuracy(Test) = {test_accuracy:.6f}")

    history.append({'max_depth': max_depth, 'Accuracy(Train)': train_accuracy, 
    'Accuracy(Test)': test_accuracy})

history = pd.DataFrame(history)
history


#%%
%matplotlib inline 
import matplotlib.pyplot as plt

plt.plot(history["max_depth"], history["Accuracy(Train)"], label="Accuracy(Train)")
plt.plot(history["max_depth"], history["Accuracy(Test)"], label="Accuracy(Test)")

plt.legend()
#%%  

##min_samples_split 가지치기 전에 샘플이 몇개 확보되어있어야 하는가

min_samples_split_list = range(10, 1001, 10)

history = []

for min_samples_split in min_samples_split_list:
    model = DecisionTreeClassifier(min_samples_split \
     = min_samples_split, random_state = 42) #max_depth를 제한해서 overfitting 방지

    model.fit(X_train,Y_train)

    Y_train_predict = model.predict(X_train)
    Y_test_predict = model.predict(X_test)

    train_accuracy = (Y_train == Y_train_predict).mean()
    test_accuracy = (Y_test == Y_test_predict).mean()

    print(f"Accuracy(Train) = {train_accuracy:.6f}")
    print(f"Accuracy(Test) = {test_accuracy:.6f}")

    history.append({'min_samples_split': min_samples_split, 'Accuracy(Train)': train_accuracy, 
    'Accuracy(Test)': test_accuracy})

history = pd.DataFrame(history)
historyn

#%%

%matplotlib inline 
import matplotlib.pyplot as plt

plt.plot(history["min_samples_split"], history["Accuracy(Train)"], label="Accuracy(Train)")
plt.plot(history["min_samples_split"], history["Accuracy(Test)"], label="Accuracy(Test)")

plt.legend()

#%%

from sklearn.ensemble import RandomFrestClassifier
