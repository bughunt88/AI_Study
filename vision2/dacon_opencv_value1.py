

import cv2
import numpy as np

qimg = cv2.imread("../data/vision2/1/00000.png",0)
timg = cv2.imread("../data/vision2/1/19403.png",0)

# timg = cv2.imread('./images/rotated_irene.jpg',0) # trainImage

# SIFT 
orb = cv2.ORB_create()

# SIFT로 키포인트와 디스크립터 찾기
kp1,des1 = orb.detectAndCompute(qimg,None)
kp2,des2 = orb.detectAndCompute(timg,None)


# BFMatcher 객체 생성
bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)

# 디스크립터들 매칭시키기
matches = bf.match(des1,des2)

# 거리에 기반하여 순서 정렬하기
matches = sorted(matches, key = lambda x:x.distance)

# 첫 10개 매칭만 그리기
# flags=2는 일치되는 특성 포인트만 화면에 표시!
res = cv2.drawMatches(qimg,kp1,timg,kp2,matches[:10],res,flags=2)

cv2.imshow("Feature Matching",res)
cv2.waitKey(0)
cv2.destroyAllWindows()