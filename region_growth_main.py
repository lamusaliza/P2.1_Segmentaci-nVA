import numpy as np
import cv2
from region_growth import segmentation

threshold1 = 107
seed1 = (80, 127)
image1 = cv2.imread('mri.jpg', cv2.IMREAD_GRAYSCALE)

seg_image1 = segmentation(image1, threshold1, seed1)
cv2_imshow(image1)
cv2_imshow(seg_image1)

threshold2 = 150
seed2 = (130, 183)
image2 = cv2.imread('apple.jpg', cv2.IMREAD_GRAYSCALE)

seg_image2 = segmentation(image2, threshold2, seed2)
cv2_imshow(image2)
cv2_imshow(seg_image2)