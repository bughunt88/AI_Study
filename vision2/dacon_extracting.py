import numpy as np
import cv2
 
def x_cord_contour(contour):
    # This function take a contour from findContours
    # it then outputs the x centroid coordinates
    
    
    M = cv2.moments(contour)
    return (int(M['m10']/M['m00']))
 
 
def makeSquare(not_square):
    # This function takes an image and makes the dimenions square
    # It adds black pixels as the padding where needed
    
    BLACK = [0,0,0]
    img_dim = not_square.shape
    height = img_dim[0]
    width = img_dim[1]
    #print("Height = ", height, "Width = ", width)
    if (height == width):
        square = not_square
        return square
    else:
        doublesize = cv2.resize(not_square,(2*width, 2*height), interpolation = cv2.INTER_CUBIC)
        height = height * 2
        width = width * 2
        #print("New Height = ", height, "New Width = ", width)
        if (height > width):
            pad = int((height - width)/2)
            #print("Padding = ", pad)
            doublesize_square = cv2.copyMakeBorder(doublesize,0,0,pad,pad,cv2.BORDER_CONSTANT,value=BLACK)
        else:
            pad = int((width - height)/2)
            #print("Padding = ", pad)
            doublesize_square = cv2.copyMakeBorder(doublesize,pad,pad,0,0,cv2.BORDER_CONSTANT,value=BLACK)
    doublesize_square_dim = doublesize_square.shape
    #print("Sq Height = ", doublesize_square_dim[0], "Sq Width = ", doublesize_square_dim[1])
    return doublesize_square
 
 
def resize_to_pixel(dimensions, image):
    # This function then re-sizes an image to the specificied dimenions
    
    buffer_pix = 4
    dimensions  = dimensions - buffer_pix
    squared = image
    r = float(dimensions) / squared.shape[1]
    dim = (dimensions, int(squared.shape[0] * r))
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    img_dim2 = resized.shape
    height_r = img_dim2[0]
    width_r = img_dim2[1]
    BLACK = [0,0,0]
    if (height_r > width_r):
        resized = cv2.copyMakeBorder(resized,0,0,0,1,cv2.BORDER_CONSTANT,value=BLACK)
    if (height_r < width_r):
        resized = cv2.copyMakeBorder(resized,1,0,0,0,cv2.BORDER_CONSTANT,value=BLACK)
    p = 2
    ReSizedImg = cv2.copyMakeBorder(resized,p,p,p,p,cv2.BORDER_CONSTANT,value=BLACK)
    img_dim = ReSizedImg.shape
    height = img_dim[0]
    width = img_dim[1]
    #print("Padded Height = ", height, "Width = ", width)
    return ReSizedImg
 



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

# Fint Contours
contours, _ = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
#Sort out contours left to right by using their x cordinates
filtered_contours = [c for c in contours if cv2.contourArea(c) > 10]
contours = sorted(filtered_contours, key = x_cord_contour, reverse = False)
# Create empty array to store entire number
full_number = []
 
# loop over the contours
for c in contours:
    # compute the bounding box for the rectangle
    (x, y, w, h) = cv2.boundingRect(c)    
    
    #cv2.drawContours(image, contours, -1, (0,255,0), 3)
    #cv2.imshow("Contours", image)
 
    if w >= 5 and h >= 25:
        roi = canny[y:y + h, x:x + w]
        ret, roi = cv2.threshold(roi, 127, 255,cv2.THRESH_BINARY_INV)
        squared = makeSquare(roi)
        final = resize_to_pixel(20, squared)
        cv2.imshow("final", final)

        #final_array = final.reshape((1,400))
        #final_array = final_array.astype(np.float32)
        #ret, result, neighbours, dist = knn.findNearest(final_array, k=1)
        #number = str(int(float(result[0])))
        #full_number.append(number)
        # draw a rectangle around the digit, the show what the
        # digit was classified as
        #cv2.rectangle(canny, (x, y), (x + w, y + h), (0, 0, 255), 2)
        #cv2.putText(image, number, (x , y + 155),
        #    cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 2)
        #cv2.imshow("image", image)
        cv2.waitKey(0) 
        
cv2.destroyAllWindows()

