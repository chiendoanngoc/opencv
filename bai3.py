import cv2

img = cv2.imread("cat3.jpg")
cv2.imshow("Image", img)

# BLUR
blur = cv2.GaussianBlur(img, (3, 3), cv2.BORDER_DEFAULT)
cv2.imshow("Blur", blur)

more_blur = cv2.GaussianBlur(img, (7, 7), cv2.BORDER_DEFAULT)
cv2.imshow("More Blur", more_blur)

# EDGE
canny = cv2.Canny(img, 125, 175)
cv2.imshow("Edge", canny)

cv2.waitKey(0)