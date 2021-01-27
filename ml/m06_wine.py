
# 분류 머신 러닝 다양한 모델 

import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


dataset = load_wine()
x = dataset.data
y = dataset.target


x_train, x_test, y_train, y_test = train_test_split(x, y,  train_size=0.7, random_state = 66 ) 

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# 2. 모델 

'''
#model = LinearSVC()

#model = SVC()

#model = KNeighborsClassifier()

#model = DecisionTreeClassifier()

#model = RandomForestClassifier()

model = LogisticRegression()



# 3. 컴파일, 훈련


# model.fit(x_train, y_train, epochs=100, batch_size=8, validation_data=(x_val,y_val), callbacks=[eraly_stopping])
model.fit(x,y)

y_pred = model.predict(x_test)
print(x_test, "의 예측결과 : ", y_pred)

result = model.score(x_test, y_test)
print("model.score : ", result)

# 회기에서는 r2_score으로 계산한다 
acc = accuracy_score(y_test, y_pred)
print('accuracy_score : ', acc)
'''


models = [LinearSVC(), SVC(), KNeighborsClassifier(), DecisionTreeClassifier(), RandomForestClassifier(), LogisticRegression()]
for i in models:
    model = i

    #3. compile fit
    model.fit(x_train,y_train)

    #4. evaluation, prediction
    print(f'\n{i}')
    result = model.score(x_test,y_test)
    print('model_score : ', result)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test,y_pred)
    print('accuracy_score : ', accuracy)
    

