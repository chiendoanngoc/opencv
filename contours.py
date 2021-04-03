import cv2

img = cv2.imread("opencv-logo.png")
imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Original", img)
# cv2.imshow("Gray", imggray)

ret, thresh = cv2.threshold(imggray, 150, 255, 0)
# cv2.imshow("Threshold", thresh)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

# print("Number of contours = " + str(len(contours)))
cv2.drawContours(img, contours, -1, (100, 100, 100), 3)
cv2.imshow("Contours", img)

cv2.waitKey(0)
cv2.destroyAllWindows()