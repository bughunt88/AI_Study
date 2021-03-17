
# 불가 



import numpy as np
import PIL
from numpy import asarray
from PIL import Image
import cv2
import matplotlib.pyplot as plt




import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'


filepath='../data/LPD_competition/train/189/16.jpg'

#filepath='../data/vision2/test_dirty_mnist_2nd/50000.png'
    #image=Image.open(filepath)
    #image_data = image.resize((128,128))
image = cv2.imread(filepath) # cv2.IMREAD_GRAYSCALE
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
'''
image2 = np.where((image <= 254) & (image != 0), 0, image)
image3 = cv2.dilate(image2, kernel=np.ones((2, 2), np.uint8), iterations=1)
image_data = cv2.medianBlur(src=image3, ksize= 5)  #점처럼 놓여있는  noise들을 제거할수있음
'''
plt.figure(figsize=(20, 5))
ax = plt.subplot(2, 10, 1)
plt.imshow(image)
plt.show()


print(pytesseract.image_to_string(image))
