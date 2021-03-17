import numpy as np
import PIL
from numpy import asarray
from PIL import Image
import cv2


# 오픈 cv를 통해 전처리 후 128, 128로 리사이징 npy 저장!

img=[]
for i in range(50000):
    filepath='../data/vision2/dirty_mnist_2nd/%05d.png'%i
    #image=Image.open(filepath)
    #image_data = image.resize((128,128))
    image = cv2.imread(filepath) # cv2.IMREAD_GRAYSCALE
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image2 = np.where((image <= 254) & (image != 0), 0, image)
    image3 = cv2.dilate(image2, kernel=np.ones((2, 2), np.uint8), iterations=1)
    image_data = cv2.medianBlur(src=image3, ksize= 5)  #점처럼 놓여있는  noise들을 제거할수있음
    image_data = cv2.resize(image_data, (128, 128))
    image_data = np.array(image_data)
    img.append(image_data)
#np.save('../data/vision2/train_data.npy', arr=img)



print("train 끝")

img=[]
for i in range(50000,55000):
    filepath='../data/vision2/test_dirty_mnist_2nd/%05d.png'%i
    #image=Image.open(filepath)
    #image_data = image.resize((128,128))
    image = cv2.imread(filepath) # cv2.IMREAD_GRAYSCALE
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image2 = np.where((image <= 254) & (image != 0), 0, image)
    image3 = cv2.dilate(image2, kernel=np.ones((2, 2), np.uint8), iterations=1)
    image_data = cv2.medianBlur(src=image3, ksize= 5)  #점처럼 놓여있는  noise들을 제거할수있음
    image_data = cv2.resize(image_data, (128, 128))

    image_data = np.array(image_data)
    
    
    img.append(image_data)
#np.save('../data/vision2/predict_data.npy', arr=img)


print("predict 끝")
