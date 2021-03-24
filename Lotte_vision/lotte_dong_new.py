import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image

rotXdeg = 90
rotYdeg = 90
rotZdeg = 90
f = 500
dist = 500

def onRotXChange(val):
    global rotXdeg
    rotXdeg = val
def onRotYChange(val):
    global rotYdeg
    rotYdeg = val
def onRotZChange(val):
    global rotZdeg
    rotZdeg = val
def onFchange(val):
    global f
    f=val
def onDistChange(val):
    global dist
    dist=val

for i in range(1000):
    for img in range(24,48):

        original = cv2.imread('../data/lpd/train_new/{0:04}/{1:02}.jpg'.format(i, img))
        modified = np.ndarray(shape=original.shape,dtype=original.dtype)

        h , w = original.shape[:2]

        # while True:

        rotX = (30)*np.pi/180
        rotY = 0
        rotZ = 0

        A1= np.matrix([[1, 0, -w/2],
                    [0, 1, -h/2],
                    [0, 0, 0   ],
                    [0, 0, 1   ]])

        RX = np.matrix([[1,           0,            0, 0],
                        [0,np.cos(rotX),-np.sin(rotX), 0],
                        [0,np.sin(rotX),np.cos(rotX) , 0],
                        [0,           0,            0, 1]])

        RY = np.matrix([[ np.cos(rotY), 0, np.sin(rotY), 0],
                        [            0, 1,            0, 0],
                        [ -np.sin(rotY), 0, np.cos(rotY), 0],
                        [            0, 0,            0, 1]])

        RZ = np.matrix([[ np.cos(rotZ), -np.sin(rotZ), 0, 0],
                        [ np.sin(rotZ), np.cos(rotZ), 0, 0],
                        [            0,            0, 1, 0],
                        [            0,            0, 0, 1]])

        R = RX * RY * RZ

        T = np.matrix([[1,0,0,0],
                    [0,1,0,0],
                    [0,0,1,dist],
                    [0,0,0,1]])

        A2= np.matrix([[f, 0, w/2,0],
                    [0, f, h/2,0],
                    [0, 0,   1,0]])

        H = A2 * (T * (R * A1))
        
        cv2.warpPerspective(original, H, (w, h), modified, cv2.INTER_CUBIC)

        cv2.imwrite('../data/lpd/train_new2/{0:04}/{1:02}.jpg'.format(i, img+24), modified)
            