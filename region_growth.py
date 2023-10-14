import cv2
import numpy as np

def neighbor(x, y):
    neighbors = []

    # arriba
    neighbors.append((x - 1, y))
    # abajo
    neighbors.append((x + 1, y))
    # izquierda
    neighbors.append((x, y - 1))
    # derecha
    neighbors.append((x, y + 1))
      # diagonal superior izquierda
    neighbors.append((x - 1, y - 1))
    # diagonal superior derecha
    neighbors.append((x - 1, y + 1))
    # diagonal inferior izquierda
    neighbors.append((x + 1, y - 1))
    # diagonal inferior derecha
    neighbors.append((x + 1, y + 1))
    return neighbors

def growth(pixel, seed, threshold):
    if abs(seed - pixel) < threshold:
        return True

    return False

def segmentation(image, threshold, seed):
  visited = np.zeros(image.shape, dtype=bool)

  stack = [(seed[0], seed[1])]
  region = np.zeros(image.shape, dtype=bool)

  while stack:
      x, y = stack.pop()
      region[x, y] = True
      visited[x, y] = True

      neighbors = neighbor(x, y)
      for nx, ny in neighbors:
          if 0 <= nx < image.shape[0] and 0 <= ny < image.shape[1] and not visited[nx, ny]:
              if growth(image[nx, ny], image[seed[0], seed[1]], threshold):
                  stack.append((nx, ny))

  segmented_image = image.copy()
  segmented_image[region] = 255

  return segmented_image