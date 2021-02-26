import cv2
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as pilimg
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. 노이즈부터 제거하자 (한수오빠 dacon_img_change.py 참고)
# rect = (1,1,img.shape[0]-1,img.shape[1]-1)
file_path = '../data/vision2/test_dirty_mnist_2nd/50003.png'


image = cv2.imread(file_path) # cv2.IMREAD_GRAYSCALE
image = cv2.cvtColor(image, cv2.IMREAD_GRAYSCALE)
image2 = np.where((image <= 254) , 0, image)
image3 = cv2.dilate(image2, kernel=np.ones((2, 2), np.uint8), iterations=1)
# image5 = cv2.erode(image3, kernel=np.ones((2, 2), np.uint8), iterations =1)
# image_data = cv2.GaussianBlur(image5, (5,5), 0)
image_data = cv2.medianBlur(src=image3, ksize=5)  #점처럼 놓여있는  noise들을 제거할수있음
# image_data = cv2.resize(image_data, (128, 128))
image_data = np.array(image_data)
image_data = image_data.astype(np.uint8)

cv2.imshow('image_data',image)

# 외곽검출
edged = cv2.Canny(image_data, 10, 250)
# cv2.imshow('Edged', edged)
# 엣지 이미지로 closed를 찾기(끊어지지 않는 선)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
# cv2.imshow('closed', closed)
# cv2.waitKey(0)
# close이미지와 findContours()를 이용하여 컨투어 경계를 찾기
contours, _ = cv2.findContours(closed.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
total = 0
# 외곽선 그리는 용도. 이미지에 그리기 때문에 이 코드 적용하면 원래 이미지에
# 초록색 선 생김
contours_image = cv2.drawContours(image, contours, -1, (0,255,0), 5)
# cv2.imshow('contours_image', contours_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows() 
contours_xy = np.array(contours)
print(contours_xy.shape)
print(contours_xy[0])

# 각 모서리에 대한 좌표
# x의 min과 max 찾기
x_min, x_max = 0,0
y_min, y_max = 0,0

for i in range(len(contours_xy)):
    value_x = list()
    value_y = list()
    for j in range(len(contours_xy[i])):
        value_x.append(contours_xy[i][j][0][0]) #네번째 괄호가 0일때 x의 값
        x_min = min(value_x)
        x_max = max(value_x)
    for j in range(len(contours_xy[i])):
        value_y.append(contours_xy[i][j][0][1]) #네번째 괄호가 0일때 x의 값
        y_min = min(value_y)
        y_max = max(value_y)
    # image trim 하기
    # 이미지를 자르기 위해 높이와 넓이
    x = x_min 
    y = y_min 
    w = x_max-x_min
    h = y_max-y_min

import cv2

contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#                     cv2.findContours(이진화 이미지, 검색 방법, 근사화 방법)
for i in range(len(contours)):
    cv2.drawContours(image.copy(), [contours[i]], 0, (255, 0, 0), 2)
   #cv2.drawContours(이미지, [윤곽선], 윤곽선 인덱스, (B, G, R), 두께, 선형 타입)
    #cv2.putText(image11, str(i), tuple(contours[i][0][0]), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 1)
    # x, y, w, h = cv2.boundingRect(i)
    print(i, hierarchy[0][i])
    cv2.imshow("src", edged)
    cv2.waitKey(0)

cv2.destroyAllWindows()