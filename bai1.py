import cv2

img = cv2.imread("cat3.jpg")
crop_img = img[200:600, 100:500]

cv2.imshow("anh meo", img)
cv2.imshow("anh meo crop", crop_img)

# hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# cv2.imshow("anh meo hsv", hsv_img)

# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow("anh meo gray", gray_img)

cv2.waitKey()