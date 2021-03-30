import cv2
import numpy as np
from math import sqrt,exp

def distance(point1,point2):
    return sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

def idealFilterLP(D0,imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            if distance((y,x),center) < D0:
                base[y,x] = 1
    return base

def idealFilterHP(D0,imgShape):
    base = np.ones(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            if distance((y,x),center) < D0:
                base[y,x] = 0
    return base
# print(idealFilterLP(2,(5,5)))
# print(idealFilterHP(2,(10,10)))

def gaussianLP(D0,imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x] = exp(((-distance((y,x),center)**2)/(2*(D0**2))))
    return base

def gaussianHP(D0,imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x] = 1 - exp(((-distance((y,x),center)**2)/(2*(D0**2))))
    return base
# print("Gaussian LP")
# print(gaussianLP(3,(5,5)))
# print()
# print("Gaussian HP")
# print(gaussianHP(3,(5,5)))

img = cv2.imread('noise.png', 0)
print(img)
# cv2.imshow('Original Image', img)

original_spectrum = np.fft.fft2(img)
print(original_spectrum)
print(np.log(1+np.abs(original_spectrum)))
# cv2.imshow('Original Spectrum', np.log(1+np.abs(original_spectrum)))

cv2.waitKey(0)