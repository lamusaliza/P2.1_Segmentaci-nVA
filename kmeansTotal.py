import cv2
import numpy as np
import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from Metodos.clasificacion.kmeans import kmeans

# Cargar la imagen
imagen = cv2.imread("img.png")  # Reemplaza "tu_imagen.jpg" por la ruta de tu imagen

# Aplicar kmeans a la imagen
data = imagen.reshape(-1, 3)  # Reorganiza la imagen para que sea una matriz de píxeles
k = 6  # Número de clústeres
max_iterations = 10
clusters, centroids, labels = kmeans(data, k, max_iterations)