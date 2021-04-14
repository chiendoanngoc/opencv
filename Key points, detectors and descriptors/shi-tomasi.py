# Shi-Tomasi Corner Detector & Good Features to Track
# In opencv it is called goodfeaturestotrack


import cv2
import numpy as np

img = cv2.imread('images/triangle.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#input image, #points, quality level (0-1), min euclidean dist. between detected points
corners = cv2.goodFeaturesToTrack(gray,50,0.05,10)
corners = np.int0(corners)   #np.int0 is int64

for i in corners:
    x,y = i.ravel()   # Ravel Returns a contiguous flattened array.
#    print(x,y)
    cv2.circle(img,(x,y),3,255,-1)  #Draws circle (Img, center, radius, color, etc.)

cv2.imshow('Corners',img)
cv2.waitKey(0)