import numpy as np
import cv2
import matplotlib.pyplot as plt


def Otsu(image, T1, T2):
  threshold = []

  for T in range (T1, T2):
    count = np.zeros(256, dtype=int)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            count[image[i, j]] += 1

    wb, wf = 0, 0
    ub, uf = 0, 0
    ob, of = 0, 0
    b, f = 0, 0

    for i in range(256):
        if i < T:
            wb += count[i]
            ub += count[i] * i
            b = wb
        else:
            wf += count[i]
            uf += count[i] * i
            f = wf

    wb = wb / (image.shape[0] * image.shape[1])
    wf = wf / (image.shape[0] * image.shape[1])

    ub = ub / b
    uf = uf / f

    for i in range(256):
        if i < T:
            ob += count[i] * ((i - ub) ** 2)
        else:
            of += count[i] * ((i - uf) ** 2)

    ob = ob / b
    of = of / f

    o = (wb * ob) + (wf * of)

    threshold.append(o)

  min_t = threshold.index(min(threshold))

  otsu = np.zeros_like(image)

  for i in range(image.shape[0]):
      for j in range(image.shape[1]):
          if image[i, j] > T1 + min_t:
              otsu[i, j] = 255
          else:
              otsu[i, j] = 0
  return otsu