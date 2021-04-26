import cv2
import numpy as np

def resizeImage(img, scale=0.75):
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    dimensions = (width, height)

    return cv2.resize(img, dimensions, interpolation=cv2.INTER_AREA)

img = cv2.imread("images/kinect.jpg", 0)
img1 = cv2.imread("images/kinect1.jpg", 0)
img2 = resizeImage(cv2.imread("images/kinect2.jpg", 0), 0.3)

orb = cv2.ORB_create()

kp, des = orb.detectAndCompute(img, None)
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# imgKp = cv2.drawKeypoints(img, kp, None)
# imgKp1 = cv2.drawKeypoints(img1, kp1, None)
# imgKp2 = cv2.drawKeypoints(img2, kp2, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des, des2, k=2)

good = []
for m, n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

img_match = cv2.drawMatchesKnn(img, kp, img2, kp2, good, None, flags=2)

# print(des[0])
# cv2.imshow("Kinect", img)
# cv2.imshow("Kinect1", img1)
# cv2.imshow("Kinect2", img2)
# cv2.imshow("Kinect KP", imgKp)
# cv2.imshow("Kinect1 KP", imgKp1)
# cv2.imshow("Kinect2 KP", imgKp2)
cv2.imshow("Matches", img_match)
cv2.waitKey(0)
