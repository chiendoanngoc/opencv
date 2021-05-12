import cv2
import numpy as np
from matplotlib import pyplot as plt

# Reading Image
image = cv2.imread('images/gao4.png')
height = image.shape[0]
width = image.shape[1]

# Delete picture edge noise
# image = image[10:height-10,10:width-10,:] 
cv2.imshow('original Image', image)
cv2.waitKey(0)

# Median blur
image_blur = cv2.medianBlur(image,3)
cv2.imshow('blur',image_blur)
cv2.waitKey(0)

# unsharp
gaussian = cv2.GaussianBlur(image_blur, (5, 5), 2.0)
unsharp_img = cv2.addWeighted(image_blur, 1.5, gaussian, -0.5, 0, image)
cv2.imshow('unsharp', unsharp_img)
cv2.waitKey(0)

# Gauss
gaussian = cv2.GaussianBlur(unsharp_img, (1, 1), 2.0)
cv2.imshow('gaussian', gaussian)
cv2.waitKey(0)

# Grayscale 
gray = cv2.cvtColor(gaussian, cv2.COLOR_BGR2GRAY) 
cv2.imshow('gray',gray)
cv2.waitKey(0)
# print(gray, gray.shape)

# FFT
dft = cv2.dft(np.float32(gray),flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
dft_shift[227:233, 219:225] = 255
dft_shift[227:233, 236:242] = 255

f_ishift = np.fft.ifftshift(dft_shift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

min, max = np.amin(img_back, (0,1)), np.amax(img_back, (0,1))
img_back = cv2.normalize(img_back, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
cv2.imshow('Sinus noise removed', img_back)
cv2.waitKey(0)
# plt.imshow(img_back, cmap = 'gray')
# plt.show()
# print(img_back, img_back.shape)

# Find Canny edges
# tim bien ngoai cung
# phan vung (lay nguong)
edged = cv2.Canny(img_back, 90, 200) 
cv2.imshow('edged Image',edged)
cv2.waitKey(0)

# Closing
# tham so khac
# kernel = np.ones((13,13),np.uint8)

kernel = np.ones((1,1),np.uint8)

closing = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
cv2.imshow('closing',closing)
cv2.waitKey(0) 

# Finding Contours 
contours, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  #Contour Retrieval Mode,   stores absolutely all the contour points.
#https://docs.opencv.org/3.4/d9/d8b/tutorial_py_contours_hierarchy.html

number=str(len(contours))

# Draw all contours 
# -1 signifies drawing all contours 
cv2.drawContours(image, contours, -1, (0, 255, 0), 1) 
cv2.putText(image, "Number of contours = " + number, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255	), 2)

cv2.imshow('Contours', image) 
cv2.waitKey(0) 
cv2.destroyAllWindows() 