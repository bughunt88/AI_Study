from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 1. 데이터 

x = np.array(range(1,101))
y = np.array(range(1,101))


#x_train = x[:60] # : 앞에 아무것도 없으면 처음부터라고 나타낸다  1 ~ 60
#x_val = x[60:80] # 61 ~ 80
#x_test = x[80:]  # 81 ~ 100
# 리스트의 슬라이싱

#y_train = y[:60] # : 앞에 아무것도 없으면 처음부터라고 나타낸다  1 ~ 60
#y_val = y[60:80] # 61 ~ 80
#y_test = y[80:]  # 81 ~ 100
# 리스트의 슬라이싱


from sklearn.model_selection import train_test_split
# 싸이킷 런에서 스플릿 해주는 기능이 있다 

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.6, shuffle=True) # shuffle=False 은 순서 섞지 말고 나오도록 하는 함수 , 섞고 싶으면 True, 기본 값은 트루이다

print(x_train)

print(x_train.shape)
print(x_test.shape)

print(y_train.shape)
print(y_test.shape)


# 2. 모델 구성

model = Sequential()
model.add(Dense(5, input_dim = 1))

model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(10))

model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100 )

# 4. 평가, 예측

loss, mae = model.evaluate(x_test, y_test)

print('loss : ', loss)
print('mae : ', mae )

y_predict = model.predict(x_test)

print(y_predict)

# shuffle = False
# loss :  0.021176215261220932
# mae :  0.14000539481639862

# shuffle = True
# loss :  0.0024943221360445023
# mae :  0.04311959445476532

# 성능이 좋아진다