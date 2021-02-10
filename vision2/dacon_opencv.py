import cv2
import numpy as np
from matplotlib import pyplot as plt

image_path = '../data/vision2/test_dirty_mnist_2nd/50001.png'

plt.figure(figsize=(12,6))
image = cv2.imread(image_path) # cv2.IMREAD_GRAYSCALE
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print(image.shape)
#img = cv2.imshow('original', image)
#cv2.waitKey(0)

#254보다 작고 0이아니면 0으로 만들어주기
image2 = np.where((image <= 254) & (image != 0), 0, image)
#cv2.imshow('filterd', image2)

image3 = cv2.dilate(image2, kernel=np.ones((2, 2), np.uint8), iterations=1)
#cv2.imshow('dilate', image3)
#dilate -> 이미지 팽창
image4 = cv2.medianBlur(src=image3, ksize= 5)  #점처럼 놓여있는  noise들을 제거할수있음

image = cv2.resize(image4, (128, 128))

print(image.shape)

cv2.imshow('median', image)
#medianBlur->커널 내의 필터중 밝기를 줄세워서 중간에 있는 값으로 현재 픽셀 값을 대체

cv2.waitKey(0) #cv2 실행