import cv2
import numpy as np



original = cv2.imread('../../data/lpd_competition/train/95/31.jpg', cv2.IMREAD_COLOR)
img = cv2.imread('../../data/lpd_competition/train/95/31.jpg', cv2.IMREAD_COLOR)
img = cv2.resize(img,(300,400))

# img = cv2.resize(img,(256,256))
img2 = img[0:250, 0:256].copy()
img2 = cv2.resize(img2,(256,256))
original = cv2.resize(original,(256,256))

# kernel = np.array([[0, -1, 0],
#                 [-1, 5, -1],
#                 [0, -1, 0]])

# # 커널 적용 
# img2 = cv2.filter2D(img2, -1, kernel)

# cv2.imshow("origin", img)
cv2.imshow('origin', original)
cv2.imshow('cut', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()     
