import numpy as np
import cv2
from matplotlib import pyplot as plt

img2 = cv2.imread("../data/vision2/1/19403.png",0)
image_data = cv2.imread("../data/vision2/1/00000.png", cv2.IMREAD_GRAYSCALE)

'''
# 전처리 
image2 = np.where((image <= 254) & (image != 0), 0, image)
image3 = cv2.dilate(image2, kernel=np.ones((2, 2), np.uint8), iterations=1)
image_data = cv2.medianBlur(src=image3, ksize= 5)  #점처럼 놓여있는  noise들을 제거할수있음
'''

sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(image_data,None)
kp2, des2 = sift.detectAndCompute(img2,None)
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)
good = []
for m,n in matches:
    if m.distance < 0.6*n.distance:
        good.append([m])
  
print(good)

img3 = cv2.drawMatchesKnn(image_data,kp1,img2,kp2,good,None,flags=2)
plt.imshow(img3),plt.show()