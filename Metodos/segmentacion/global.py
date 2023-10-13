import cv2
import numpy as np
import matplotlib.pyplot as plt

def umbralizacion_global(image, umbral_inicial=170, delta=1):
    # Carga la imagen en escala de grises
    imagen = cv2.imread(image, 0)

    # Inicializa el umbral anterior con un valor arbitrario grande para garantizar que se ejecute al menos una vez
    umbral_anterior = 255

    # Itera hasta que la diferencia entre el umbral actual y el umbral anterior sea menor que delta
    while abs(umbral_anterior - umbral_inicial) >= delta:
        # Divide la imagen en dos grupos G1 y G2
        grupo1 = imagen[imagen > umbral_inicial]
        grupo2 = imagen[imagen <= umbral_inicial]

        # Calcula los promedios de intensidad para cada grupo
        if len(grupo1) > 0:
            m1 = sum(grupo1) / len(grupo1)
        else:
            m1 = 0
        if len(grupo2) > 0:
            m2 = sum(grupo2) / len(grupo2)
        else:
            m2 = 0

        # Actualiza el umbral anterior con el umbral actual
        umbral_anterior = umbral_inicial

        # Calcula el nuevo umbral como el promedio de los promedios de intensidad
        umbral_inicial = (m1 + m2) / 2

    # Umbralización final
    imagen_umbralizada = imagen.copy()
    imagen_umbralizada[imagen > umbral_inicial] = 255
    imagen_umbralizada[imagen <= umbral_inicial] = 0

    return imagen_umbralizada