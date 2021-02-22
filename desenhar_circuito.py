import numpy as np 
import open3d as o3d 
import minhas_funcoes_Breman as myf

# Carregar translacoes:
matriz_translacoes = np.loadtxt("finalposes/translacoes_poses_groundthuth.txt")
matriz_translacoes_xy = np.loadtxt("finalposes/translacoes_poses_groundthuth.txt")
matriz_translacoes_xz = np.loadtxt("finalposes/translacoes_poses_groundthuth.txt")
# Carregar rotacoes:
lista_de_rotacoes = [np.identity(3)]
for i in range(1,13):
    R = np.loadtxt("finalposes/scan%d.dat" %i)[0:3,0:3]
    lista_de_rotacoes.append(R)
# Planificar translacoes nos planos xy e xz:
matriz_translacoes_xy[:,2] = 0
matriz_translacoes_xz[:,1] = 0
# Criar linhas e sistemas:
tamanho = 5.0
lista_de_sistemas, conjunto_de_linhas = myf.criar_linhas_e_frames_3D(matriz_translacoes,lista_de_rotacoes,tamanho)
lista_de_sistemas_xy, conjunto_de_linhas_xy = myf.criar_linhas_e_frames_3D(matriz_translacoes_xy,lista_de_rotacoes,tamanho)
lista_de_sistemas_xz, conjunto_de_linhas_xz = myf.criar_linhas_e_frames_3D(matriz_translacoes_xz,lista_de_rotacoes,tamanho)
# Desenhar linhas e sistemas:
campo_de_visao = -90.0 # graus
myf.desenhar_poses_com_linhas_e_frames(lista_de_sistemas,conjunto_de_linhas,campo_de_visao) # 3D
myf.desenhar_poses_com_linhas_e_frames(lista_de_sistemas_xy,conjunto_de_linhas_xy,campo_de_visao) # XY
myf.desenhar_poses_com_linhas_e_frames(lista_de_sistemas_xz,conjunto_de_linhas_xz,campo_de_visao) # XZ
# Calcular distancias planas:
distancias_planas = []
for i in range(len(matriz_translacoes_xy)):
    d = np.linalg.norm(matriz_translacoes_xy[i]-matriz_translacoes_xy[i-1])
    distancias_planas.append(d)
desniveis = []
for i in range(len(matriz_translacoes_xz)):
    d = np.linalg.norm(matriz_translacoes_xz[i]-matriz_translacoes_xz[i-1])
    desniveis.append(d)
print(distancias_planas)
print(desniveis)