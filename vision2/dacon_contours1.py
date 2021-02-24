import os
import cv2


src = cv2.imread('../data/vision2/test_dirty_mnist_2nd/50001.png')
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
print(hierarchy)

idx = 0
while True:
    sel = input(f'items 0 ~ {len(hierarchy[0])-1} >> (quit: q) ')
    if sel == 'q':
        break
    if sel.isdigit() and int(sel) < len(hierarchy[0]):
        print(hierarchy[0][int(sel)])

        idx = hierarchy[0, int(sel), 0]
        # idx = hierarchy[0][sel][0]
        dst = src.copy()
        cv2.drawContours(dst, contours, idx, (255, 0, 0), 2, cv2.LINE_8, hierarchy)
        cv2.imshow('src', src)
        cv2.imshow('dst', dst)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()