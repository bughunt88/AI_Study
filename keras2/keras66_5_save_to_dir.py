import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True, # 수평 뒤집기 
    vertical_flip=True, # 수직 뒤집기 
    width_shift_range=0.1, # 수평 이동
    height_shift_range=0.1, # 수직 이동
    rotation_range=5, # 회전 
    zoom_range=1.2, # 확대
    shear_range=0.7, # 층 밀리기 강도?
    fill_mode='nearest' # 빈자리는 근처에 있는 것으로(padding='same'과 비슷)
)


'''
- rotation_range: 이미지 회전 범위 (degrees)
- width_shift, height_shift: 그림을 수평 또는 수직으로 랜덤하게 평행 이동시키는 범위 
                                (원본 가로, 세로 길이에 대한 비율 값)
- rescale: 원본 영상은 0-255의 RGB 계수로 구성되는데, 이 같은 입력값은 
            모델을 효과적으로 학습시키기에 너무 높습니다 (통상적인 learning rate를 사용할 경우). 
            그래서 이를 1/255로 스케일링하여 0-1 범위로 변환시켜줍니다. 
            이는 다른 전처리 과정에 앞서 가장 먼저 적용됩니다.
- shear_range: 임의 전단 변환 (shearing transformation) 범위
- zoom_range: 임의 확대/축소 범위
- horizontal_flip`: True로 설정할 경우, 50% 확률로 이미지를 수평으로 뒤집습니다. 
    원본 이미지에 수평 비대칭성이 없을 때 효과적입니다. 즉, 뒤집어도 자연스러울 때 사용하면 좋습니다.
- fill_mode 이미지를 회전, 이동하거나 축소할 때 생기는 공간을 채우는 방식
'''

test_datagen = ImageDataGenerator(rescale=1./255)


# flow 또는 flow_from_directory

# train_generator
xy_train = train_datagen.flow_from_directory(
    '../data/image/brain/train',
    target_size=(150,150),
    batch_size=5,
    class_mode='binary',
    save_to_dir='../data/image/brain_generator/train/'
)
# Found 160 images belonging to 2 classes.




# test_generator
xy_test = test_datagen.flow_from_directory(
    '../data/image/brain/test',
    target_size=(150,150), # 리사이징 가능
    batch_size=5,
    class_mode='binary'
)
# Found 120 images belonging to 2 classes.


print(xy_train[1][0])

