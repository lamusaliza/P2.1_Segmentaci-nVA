import numpy as np
import random


def euclideanDistance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))

def kmeans(data, k, max_iterations):
    centroids = random.sample(list(data), k)

    for _ in range(max_iterations):
        clusters = [[] for _ in range(k)]
        labels = np.zeros(len(data), dtype=int)  # Etiquetas para los puntos

        for i, point in enumerate(data):
            distances = [euclideanDistance(point, centroid) for centroid in centroids]
            nearest = np.argmin(distances)
            clusters[nearest].append(point)
            labels[i] = nearest  # Asigna la etiqueta al punto

        centroid_update = []
        for cluster in clusters:
            mean = np.mean(cluster, axis=0)
            centroid_update.append(mean)

        if np.all(np.array(centroid_update) == np.array(centroids)):
            break

        centroids = centroid_update

    return clusters, centroids, labels