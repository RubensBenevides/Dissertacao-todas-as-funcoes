import math
import numpy as np
import open3d as o3d
import copy
import quaternion as quat
import matplotlib.pyplot as plt

I = np.identity(3)
pi = 3.14159265358979
grau = pi/180
# Cria 180 matrizes de rotacao girando 1 grau no eixo Z até 180:
lista_R = []
for i in range(1,181):
    R = np.array([[np.cos(grau*i), -np.sin(grau*i), 0],
                  [np.sin(grau*i),  np.cos(grau*i), 0],
                  [0, 0, 1]])
    lista_R.append(R)
# Calcula a distancia de cada uma dessas matrizes para a identidade
distancias = []
for i in range(180):
    dif = np.linalg.norm(lista_R[i]-I,"fro")
    distancias.append(dif)
# Plota a distancia em funcao do angulo
plt.plot(np.array(range(180)),distancias)
plt.show()
# Pegar so 100 primeiros elementos:
graus = np.array(range(100))
distancias = distancias[0:100]
plt.plot(graus,distancias)
plt.show() 
