import os
import cv2
import numpy as np


image = cv2.imread('../data/vision2/test_dirty_mnist_2nd/50001.png', cv2.IMREAD_GRAYSCALE)
pix = np.array(image)
# 이미지 전처리 -------------------------------------

#254보다 작고 0이아니면 0으로 만들어주기
x_df2 = np.where((pix <= 254) & (pix != 0), 0, pix)

# 이미지 팽창
x_df3 = cv2.dilate(x_df2, kernel=np.ones((2, 2), np.uint8), iterations=1)

# 블러 적용, 노이즈 제거
x_df4 = cv2.medianBlur(src=x_df3, ksize= 5)

# 이전 파일 것
canny = cv2.Canny(x_df4, 30, 70)

contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
i = 0
for c in contours:
    # get the bounding rect
    x, y, w, h = cv2.boundingRect(c)
    # to save the images

    cv2.imshow("final", image[y:y+h,x:x+w])

