
# 머신러닝 방식의 딥러닝  


from sklearn.svm import LinearSVC, SVC
import numpy as np
from  sklearn.metrics import accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# 1. 데이터 

x_data = [[0,0],[1,0],[0,1],[1,1]]
y_data = [0,1,1,0]

# 2. 모델

# model = LinearSVC()
# model = SVC()

model = Sequential()

# 히든 레이어 쌓으면 된다
model.add(Dense(1, input_dim=2, activation='sigmoid'))

# 3. 컴파일, 훈련

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_data, y_data, batch_size=1, epochs=100)

# 4. 평가, 예측

y_pred = model.predict(x_data)
print(x_data, "의 예측결과 : ", y_pred)

#result = model.score(x_data, y_data)

result = model.evaluate(x_data, y_data)
print("model.score : ", result[1])

# acc = accuracy_score(y_data, y_pred)
# print('accuracy_score : ', acc)
