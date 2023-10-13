import cv2
import numpy as np
import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from Metodos.clasificacion.kmeans import kmeans


imagen = cv2.imread("img.png")  

# Aplicar kmeans a la imagen
data = imagen.reshape(-1, 3)  
k = 6  # Número de clústeres
max_iterations = 10
clusters, centroids, labels = kmeans(data, k, max_iterations)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Colores para los clústeres
colors = ['r', 'g', 'b', 'c', 'm', 'y']


for i in range(k):
    cluster_data = np.array([data[j] for j in range(len(data)) if labels[j] == i])
    ax.scatter(cluster_data[:, 0], cluster_data[:, 1], cluster_data[:, 2], c=colors[i], label=f'Cluster {i+1}')

ax.set_xlabel('Canal Rojo')
ax.set_ylabel('Canal Verde')
ax.set_zlabel('Canal Azul')

plt.legend()
plt.show()