# 실습
# 남자 여자 구별
# ImageDataGenerator의 fit 사용해서 완성

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import load_img, img_to_array


img_female = load_img('../data/image/sex/female/final_1000.jpg', target_size=(224,224))
img_male = load_img('../data/image/sex/male/final_1.jpg', target_size=(224,224))

# plt.imshow(img_dog)
# plt.show()

arr_male = img_to_array(img_male)
arr_female = img_to_array(img_female)

# RGB -> BGR
from tensorflow.keras.applications.vgg16 import preprocess_input

#arr_male = preprocess_input(arr_male)
#arr_female = preprocess_input(arr_female)

arr_input = np.stack([arr_male, arr_female])

from tensorflow.keras.applications import VGG16

model = VGG16()
results = model.predict(arr_input)

# 이미지 결과 확인
from tensorflow.keras.applications.vgg16 import decode_predictions

decode_result = decode_predictions(results)

print("resluts[0] : ",decode_result[0])
print("resluts[1] : ",decode_result[1])

