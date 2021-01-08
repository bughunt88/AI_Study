'''

conv2D( 10,    (3,3), input_shape=(              5,     5,      1) )
      Filter   kernel               batch_size, row, column, channels

'''


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D

model = Sequential()

# summary에서 파라미터 구하는 식 
# (input_dim x kernel_size + bias)x filter
# (1 x 2x2 + 1) x 10 = 50

model.add(Conv2D(filters=10,  padding='same',  kernel_size=(2,2), strides=1, input_shape=(10,10,1)) )
#                 노드의 수, 원하는 사이즈로 자른다 ,  받아오는 이미지의 사이즈 (가로,세로,색) #색 1은 흑백, 3은 컬러

# padding = 'same' 은 출력 이미지 사이즈가 입력 이미지 사이즈와 동일합니다.

# strides는 보폭 써져있는 숫자만큼 멀어진다


model.add(MaxPooling2D(pool_size=(2,3)))
# 풀링에서 지정한 범위만큼에서 가장 특색있는 부분만 뽑아서 정리한다
# 디폴트는 2 -> (2,2)


model.add(Conv2D(9,  kernel_size=(2,2), padding='valid') )
# model.add(Conv2D(9,  kernel_size=(2,3) ) ) # kernel_size는 달라도 상관 없다
# model.add(Conv2D(8,2)) # 그냥 2라고 쳐도 (2,2)로 인식한다

# padding의 디폴트는 valid, 유효한 영역만 출력이 됩니다. 따라서 출력 이미지 사이즈는 입력 사이즈보다 작습니다.


model.add(Flatten())
# Conv2D를 Dense랑 엮으려면 Flatten을 써야 한다 
# Conv2D는 4차원으로 되어있는데 이것을 2차원으로 변경해서 보내준다 

model.add(Dense(1))
model.summary()


