
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import PIL.Image as pilimg



image = cv2.imread('../data/vision2/test_dirty_mnist_2nd/50000.png', cv2.IMREAD_GRAYSCALE)
pix = np.array(image)


# 이미지 전처리 -------------------------------------

#254보다 작고 0이아니면 0으로 만들어주기
x_df2 = np.where((pix <= 254) & (pix != 0), 0, pix)

# 이미지 팽창
x_df3 = cv2.dilate(x_df2, kernel=np.ones((2, 2), np.uint8), iterations=1)

# 블러 적용, 노이즈 제거
x_df4 = cv2.medianBlur(src=x_df3, ksize= 5)


# 이미지 리쉐잎 -------------------------------------
# 리쉐잎


# 이전 파일 것
canny = cv2.Canny(x_df4, 30, 70)





sobelx = cv2.Sobel(x_df4, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(x_df4, cv2.CV_64F, 0, 1, ksize=3)
laplacian = cv2.Laplacian(x_df4, cv2.CV_8U)

images = [canny, sobelx, sobely, laplacian]
titles = ['canny', 'sobelx', 'sobely', 'laplacian']

for i in range(4):
    plt.subplot(2, 2, i+1), plt.imshow(images[i]), plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()