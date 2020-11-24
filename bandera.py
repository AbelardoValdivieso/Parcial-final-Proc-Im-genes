
import cv2
import numpy as np
import os

from hough import hough

import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

class bandera():
    def __init__(self):
        self.path = 'C:/Users/Gloria Dani Abe/Desktop/Parcial_final/banderas'
        #bandera_name = (input("ingrese el nombre de la imagen deseada: (debe incluir .png eje= flag1.png ) \n"))
        self.bandera_name = "flag3.png"
        self.path_file = os.path.join(self.path, self.bandera_name)
        self.image = cv2.imread(self.path_file)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

        cv2.imshow("Image", self.image)
        cv2.waitKey(0)
    def orientacion(self):

        high_thresh = 300
        bw_edges = cv2.Canny(self.image, high_thresh * 0.3, high_thresh, L2gradient=True)

        self.hough = hough(bw_edges)
        self.accumulator =self.hough.standard_HT()

        self.acc_thresh = 70
        self.N_peaks = 11
        self.nhood = [25, 9]
        peaks = hough.find_peaks(self.accumulator, self.nhood, self.acc_thresh, self.N_peaks)

        [_, cols] = self.image.shape[:2]
        # tipos de horientación
        Horientacion_Ver = 0
        Horientacion_Hor = 0
        Horientacion_mix = 0
        image_draw = np.copy(self.image)
        for i in range(len(peaks)):
            rho = peaks[i][0]
            theta_ = hough.theta[peaks[i][1]]

            theta_pi = np.pi * theta_ / 180
            theta_ = theta_ - 180
            a = np.cos(theta_pi)
            b = np.sin(theta_pi)
            x0 = a * rho + hough.center_x
            y0 = b * rho + hough.center_y
            c = -rho
            x1 = int(round(x0 + cols * (-b)))
            y1 = int(round(y0 + cols * a))
            x2 = int(round(x0 - cols * (-b)))
            y2 = int(round(y0 - cols * a))
            if np.abs(theta_) > 88 and np.abs(theta_) < 92:
                image_draw = cv2.line(image_draw, (x1, y1), (x2, y2), [0, 255, 255], thickness=2)
                Horientacion_Hor = Horientacion_Hor + 1

            elif np.abs(theta_) < 88 or np.abs(theta_) > 92:
                image_draw = cv2.line(image_draw, (x1, y1), (x2, y2), [255, 0, 255], thickness=2)
                Horientacion_Ver = Horientacion_Ver + 1
            else:
                image_draw = cv2.line(image_draw, (x1, y1), (x2, y2), [0, 0, 255], thickness=2)
                Horientacion_mix = Horientacion_mix + 1
        val = len(peaks)
        if Horientacion_Hor == val:
            print("la horientación de la bandera es Horizontal")

        if Horientacion_Ver == val:
            print("la horientación de la bandera es Vertical")
        else:
            print("la horientación de la bandera es mixta")

        cv2.imshow("frame", bw_edges)
        cv2.imshow("lines", image_draw)
        cv2.waitKey(0)

parcial=bandera()
parcial.orientacion()
