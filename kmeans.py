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

def objetos(imagen, contorno, label_indices, burCant):
    mask = np.zeros_like(imagen[:, :, 0])
    cv2.drawContours(mask, [contorno[i] for i in label_indices], -1, (255), -1)
    masked_image = cv2.bitwise_and(imagen, imagen, mask=mask)
    masked_image = cv2.putText(masked_image, f'{burCant} burbujas', (200, 1200), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=3, color=(255, 255, 255), thickness=10, lineType=cv2.LINE_AA)
    return masked_image

# Llama a la función K-Means

k = 6  # Número de clústeres
max_iterations = 10
clusters, centroids, labels = kmeans(mediaColores, k, max_iterations)
#********************************************
# Dibuja y guarda las imágenes segmentadas
img = imagen.copy()
for label in range(k):
    label_indices = np.where(labels == label)[0]
    # print(label)
    print(label_indices)
    burCant = len(label_indices)
    masked_image = objetos(imagen, contornos, label_indices, burCant)
    img = cv2.hconcat([img, masked_image])

cv2.imwrite('colores.png', img)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) )
plt.axis('off')
plt.show()

# Llama a la función K-Means para el metodo del codo
k_values = range(1, 10)
inertia_values = []

for k in k_values:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(mediaColores)
    inertia_values.append(kmeans.inertia_)

# Traza el gráfico de la puntuación de varianza en función de k
plt.figure(figsize=(8, 6))
plt.plot(k_values, inertia_values, marker='o')
plt.xlabel('Número de Clústeres (k)')
plt.ylabel('Puntuación de Varianza (Inertia)')
plt.title('Método del Codo para Determinar k')
plt.grid(True)
plt.savefig('codoPlot.png')