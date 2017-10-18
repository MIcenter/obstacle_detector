import cv2
import numpy as np

def gabor_filter(img):
    kernel = cv2.getGaborKernel((31, 31), 4, np.pi / 16, 12, 1, 0, ktype=cv2.CV_32F)
    kernel /= 1.5 * kernel.sum()
    fimg = cv2.filter2D(img, cv2.CV_8UC3, kernel)
    return fimg
