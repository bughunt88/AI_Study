import numpy as np
import PIL
from numpy import asarray
from PIL import Image
import cv2


# 오픈 cv를 통해 전처리 후 128, 128로 리사이징 npy 저장!

img=[]
img_y=[]
for i in range(1000):
    for de in range(48):
        filepath='../data/LPD_competition/train/'+str(i)+'/'+str(de)+'.jpg'
        #image=Image.open(filepath)
        #image_data = image.resize((128,128))
        image = cv2.imread(filepath) # cv2.IMREAD_GRAYSCALE
    # 커널 생성(대상이 있는 픽셀을 강조)

        image_sharp = cv2.resize(image, (128, 128))

        image_data = np.array(image_sharp)
        img.append(image_data)
        img_y.append(i)

np.save('../data/LPD_competition/npy/train_data_x.npy', arr=img)
np.save('../data/LPD_competition/npy/train_data_y.npy', arr=img_y)


print("train 끝")

img=[]
for i in range(72000):
    filepath='../data/LPD_competition/test/'+str(i)+'.jpg'
    #image=Image.open(filepath)
    #image_data = image.resize((128,128))
    image = cv2.imread(filepath) # cv2.IMREAD_GRAYSCALE

    image_sharp = cv2.resize(image, (128, 128))

    image_data = np.array(image_sharp)
    img.append(image_data)

np.save('../data/LPD_competition/npy/predict_data.npy', arr=img)


print("predict 끝")
