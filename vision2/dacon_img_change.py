import numpy as np
import PIL
from numpy import asarray
from PIL import Image


# dirty 데이터는 train 데이터 훈련시키자!
# 50000개 
# dirty_mnist_2nd_answer.csv 는 dirty의 y값 


# test_dirty 데이터는 test 데이터!
# 5000개 
# y값을 찾는것이 목표


img=[]
for i in range(50000):
    filepath='/Users/bughunt/Downloads/2차 배포/dirty_mnist_2nd/%05d.png'%i
    image=Image.open(filepath)
    image_data=asarray(image)
    img.append(image_data)


np.save('/Users/bughunt/Downloads/2차 배포/test.npy', arr=img)

img_ch_np=np.load('/Users/bughunt/Downloads/2차 배포/test.npy')


print("끝")

print(img_ch_np.shape)