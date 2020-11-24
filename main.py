# This is a sample Python script.

import cv2
import numpy as np
import os

from hough import hough

import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

if __name__ == '__main__':
    path = 'C:/Users/Gloria Dani Abe/Desktop/Parcial_final/banderas'
    #bandera_name = (input("ingrese el nombre de la imagen deseada: (debe incluir .png eje= flag1.png ) \n"))
    bandera_name = "flag1.png"
    path_file = os.path.join(path, bandera_name)
    image = cv2.imread(path_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #cv2.imshow("Image", image)
    #cv2.waitKey(0)

    #1 número de colores en la bandera
    image = np.array(image, dtype=np.float64) / 255
    vecdistancia = []
    vecncolors= []
    graficar= []
    result  = np.ndarray(shape=(11,11), dtype=float)
    for i in range(0,len(result)):
        for j in range(0, len(result)):
             result[i][j]=0
    x = np.arange(1, 10, 1)
    acum=acumtotal=0
    for i in range(10000):
         vecdistancia.append(0)
         vecncolors.append(0)
    for i in range(9):
         graficar.append(0)

    # Load Image and transform to a 2D numpy array.
    rows, cols, ch = image.shape
    assert ch == 3
    image_array = np.reshape(image, (rows * cols, ch))
    for i in range(1,5,1):
         image_array_sample = shuffle(image_array, random_state=0)[:10000]
         print("Para n_clusters :", i)
         model = KMeans(n_clusters=i, random_state=0).fit(image_array_sample)
         modelo = KMeans(n_clusters=i, random_state=0).fit_transform(image_array_sample)
         centro = KMeans(n_clusters=i, random_state=0).fit_predict(image_array_sample)
         labels = model.predict(image_array)
         centers = model.cluster_centers_
         distance = model.inertia_
         for p in np.arange(0, i):
             for j in np.arange(0, len(centro)):
                 if p == centro[j]:
                     acum = acum + modelo[j][p]
             print("El cluster: ", p + 1,)
             result[p][i] = acum
             acum = 0

    #punto 3
    # high_thresh = 300
    # bw_edges = cv2.Canny(image, high_thresh * 0.3, high_thresh, L2gradient=True)
    #
    # hough = hough(bw_edges)
    # accumulator = hough.standard_HT()
    #
    # acc_thresh = 70
    # N_peaks = 11
    # nhood = [25, 9]
    # peaks = hough.find_peaks(accumulator, nhood, acc_thresh, N_peaks)
    #
    # [_, cols] = image.shape[:2]
    # #tipos de horientación
    # Horientacion_Ver = 0
    # Horientacion_Hor = 0
    # Horientacion_mix = 0
    # image_draw = np.copy(image)
    # for i in range(len(peaks)):
    #     rho = peaks[i][0]
    #     theta_ = hough.theta[peaks[i][1]]
    #
    #     theta_pi = np.pi * theta_ / 180
    #     theta_ = theta_ - 180
    #     a = np.cos(theta_pi)
    #     b = np.sin(theta_pi)
    #     x0 = a * rho + hough.center_x
    #     y0 = b * rho + hough.center_y
    #     c = -rho
    #     x1 = int(round(x0 + cols * (-b)))
    #     y1 = int(round(y0 + cols * a))
    #     x2 = int(round(x0 - cols * (-b)))
    #     y2 = int(round(y0 - cols * a))
    #     if np.abs(theta_) > 88 and  np.abs(theta_) < 92:
    #         image_draw = cv2.line(image_draw, (x1, y1), (x2, y2), [0, 255, 255], thickness=2)
    #         Horientacion_Hor = Horientacion_Hor + 1
    #
    #     elif np.abs(theta_) < 88 or  np.abs(theta_) > 92:
    #         image_draw = cv2.line(image_draw, (x1, y1), (x2, y2), [255, 0, 255], thickness=2)
    #         Horientacion_Ver = Horientacion_Ver + 1
    #     else:
    #         image_draw = cv2.line(image_draw, (x1, y1), (x2, y2), [0, 0, 255], thickness=2)
    #         Horientacion_mix=Horientacion_mix+1
    # val=len(peaks)
    # if Horientacion_Hor == val:
    #     print("la horientación de la bandera es Horizontal")
    #
    # if Horientacion_Ver == val:
    #     print("la horientación de la bandera es Vertical")
    # else:
    #     print("la horientación de la bandera es mixta")
    #
    # cv2.imshow("frame", bw_edges)
    # cv2.imshow("lines", image_draw)
    # cv2.waitKey(0)

