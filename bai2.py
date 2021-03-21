import cv2

img = cv2.imread("cat3.jpg")
cv2.imshow("Image", img)

def resizeImage(img, scale=0.75):
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    dimensions = (width, height)

    return cv2.resize(img, dimensions, interpolation=cv2.INTER_AREA)

img075 = resizeImage(img)
cv2.imshow("0.75", img075)

img05 = resizeImage(img, 0.5)
cv2.imshow("0.5", img05)

cv2.waitKey(0)