import open3d as o3d
import numpy as np
np.set_printoptions(suppress=True)
import minhas_funcoes as myf
import time

# Importar nuvens:
n_nuvens = 13
Lista_nuvens = [o3d.io.read_point_cloud("nuvens_amostradas_50_cm/s%d.pcd" %i ) for i in range(n_nuvens)]
# Importar Poses:
Lista_poses = [np.loadtxt("finalposes/scan%d.dat" %i) for i in range(n_nuvens)]
# Definar quantidade de frames e animar:
n_frames = 100
myf.Reconstrucao_animada_uma_de_cada_vez(Lista_nuvens, Lista_poses, n_frames)