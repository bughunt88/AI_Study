import numpy as np
import cv2
import matplotlib.pyplot as plt

img1 =cv2.imread("../data/vision2/1/19403.png",cv2.IMREAD_GRAYSCALE)
img2 =cv2.imread("../data/vision2/1/00000.png",cv2.IMREAD_GRAYSCALE)


plt.imshow(img1)
plt.show()


img1 = cv2.resize(img1, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_CUBIC)
img2 = cv2.resize(img2, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_CUBIC)
res = None



orb=cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)


bf= cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches=bf.match(des1,des2)



matches = sorted(matches, key=lambda x:x.distance)
res=cv2.drawMatches(img1,kp1,img2,kp2,matches[:2],res,flags=0)



cv2.imshow("Feature Matching",res)
cv2.waitKey(0)
cv2.destroyAllWindows()



