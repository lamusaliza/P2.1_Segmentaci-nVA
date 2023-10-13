import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import random
import cv2
from sklearn.cluster import KMeans
from Metodos.clasificacion.kmeans import kmeans


# Cargar la imagen ***********************************
imagen = cv2.imread('img.png', cv2.IMREAD_UNCHANGED)
gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
#se segmenta y se erosiona
_ , mask = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
mask = cv2.erode(mask, np.ones((7, 7), np.uint8))