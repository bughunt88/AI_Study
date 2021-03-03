# 이미지는 
# data/image/vgg 에 4개를 넣으시오.
# 개, 고양이, 라이언, 슈트
# 파일명 : 
# dog1.jpg, cat1.jpg, lion1.jpg, suit1.jpg

from tensorflow.keras.applications import VGG16
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

img_dog = load_img('../data/image/vgg/dog1.jpg', target_size=(224,224))
img_cat = load_img('../data/image/vgg/cat1.JFIF', target_size=(224,224))
img_lion = load_img('../data/image/vgg/lion1.PNG', target_size=(224,224))
img_suit = load_img('../data/image/vgg/suit1.jpg', target_size=(224,224))

# plt.imshow(img_dog)
# plt.show()

arr_dog = img_to_array(img_dog)
arr_cat = img_to_array(img_cat)
arr_lion = img_to_array(img_lion)
arr_suit = img_to_array(img_suit)

# RGB -> BGR

from tensorflow.keras.applications.vgg16 import preprocess_input

arr_dog = preprocess_input(arr_dog)
arr_cat = preprocess_input(arr_cat)
arr_lion = preprocess_input(arr_lion)
arr_suit = preprocess_input(arr_suit)

print(type(arr_dog))
print(arr_dog.shape)

arr_input = np.stack([arr_dog, arr_cat, arr_lion, arr_suit])

#2. 모델 구성

print(arr_input.shape)


model = VGG16()
results = model.predict(arr_input)

print(results)
print("results.shepe : ", results.shape)


# 이미지 결과 확인
from tensorflow.keras.applications.vgg16 import decode_predictions

decode_result = decode_predictions(results)

print("resluts[0] : ",decode_result[0])
print("resluts[1] : ",decode_result[1])
print("resluts[2] : ",decode_result[2])
print("resluts[3] : ",decode_result[3])

