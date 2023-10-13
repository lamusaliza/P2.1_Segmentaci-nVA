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

cv2.imwrite('mascara.png', cv2.hconcat([imagen, np.stack((mask, mask, mask), axis=2)]))

contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contornosChidos = mask.copy()
contornosChidos = cv2.cvtColor(contornosChidos, cv2.COLOR_GRAY2BGR)
cv2.drawContours(contornosChidos, contornos, -1, (0, 255, 0), 3)
cv2.imwrite('contornos.png',contornosChidos)
med = []

for idx, cont in enumerate(contornos):
    area = int(cv2.contourArea(cont))

    if area > 300:
        mascara = np.zeros_like(imagen[:, :, 0], dtype=np.uint8) 
        cv2.drawContours(mascara, [cont], 0, 255, -1)
        mediaB, mediaG, mediaR, _ = cv2.mean(imagen, mask=mascara)
        
        med.append([mediaB,mediaG, mediaR])

mediaColores = np.array(med)