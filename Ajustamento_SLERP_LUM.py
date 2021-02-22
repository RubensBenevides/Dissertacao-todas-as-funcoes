import numpy as np
np.set_printoptions(suppress=True)
import copy
import open3d as o3d
import quaternion as quat
import minhas_funcoes as myf
#import matplotlib
#import matplotlib.pyplot as plt

n_nuvens = 13 # quantidade de nuvens
# IMPORTAR NUVENS

voxel_size = 0.5  # metros
knn_filtro = 10   # knn 
desvio_p = 1.0    # limiar de filtragem (desvio padrao)
print("Importando nuvens e pre-processando...")
pcs_processadas = myf.carregar_nuvens_e_pre_processar(n_nuvens,voxel_size,knn_filtro,desvio_p)
print("Desenhando nuvens como entraram:")
o3d.visualization.draw_geometries(pcs_processadas,'NUVENS COMO ENTRARAM',1280,720,50,50,False)
'''
# REGISTRO AUTOMATICO FAST GLOBAL REGISTRATION (FGR) - nao funciona em Breman
knn_fpfh = 50
knn_no = 10
mu = 0.8
max_corr_dist = voxel_size
T_circuito, Fitness, RMSE = [],[],[]
for i in range(n_nuvens):
    if i < n_nuvens-1:
        print(f"Registrando nuvem {i+1} na {i}")
        T = myf.Fast_Global_Registration(pcs_processadas[i+1],pcs_processadas[i],knn_no,knn_fpfh,mu,max_corr_dist)
        #myf.desenhar_resultado_registro(pcs_processadas[i+1],pcs_processadas[i],T.transformation)
        T_circuito.append(T.transformation)
        Fitness.append(T.fitness)
        RMSE.append(T.inlier_rmse)
    if i == n_nuvens-1:
        print(f"Registrando nuvem 0 na {i}")
        T = myf.Fast_Global_Registration(pcs_processadas[0],pcs_processadas[i],knn_no,knn_fpfh,mu,max_corr_dist)
        #myf.desenhar_resultado_registro(pcs_processadas[0],pcs_processadas[i],T.transformation)
        T_circuito.append(T.transformation)
        Fitness.append(T.fitness)
        RMSE.append(T.inlier_rmse)
'''
# IMPORTAR CIRCUITO & POSES
T_circuito = [np.loadtxt("Ts_circuito_ICP_1000_iteracoes/T%d.txt" %i) for i in range(n_nuvens)]
Poses_Groundtruth = [np.loadtxt("finalposes/scan%d.dat" %i) for i in range(n_nuvens)]

# CALCULAR ERRO LOOPCLOSURE
myf.Calcular_Erro_LoopClosure(n_nuvens,T_circuito)

# CALCULAR POSES:
Pesos = list(np.ones(n_nuvens)) # Pesos pesos do ajustamento LU & Milios (1997)
Poses_Normais             = myf.reconstruir_Ts_para_origem(T_circuito)
Poses_Ajustadas_LUM       = myf.reconstruir_Ts_para_origem_LUM(T_circuito,Pesos)
Poses_Ajustadas_SLERP     = myf.reconstruir_Ts_para_origem_SLERP(T_circuito)
Poses_Ajustadas_SLERP_LUM = myf.reconstruir_Ts_para_origem_SLERP_LUM(T_circuito,Pesos)
# RECONSTRUIR POSES E RECONSTRUIR O DATASET:

myf.reconstruir_modelo_com_poses(Poses_Normais,pcs_processadas)
myf.reconstruir_modelo_com_poses(Poses_Ajustadas_LUM,pcs_processadas)
myf.reconstruir_modelo_com_poses(Poses_Ajustadas_SLERP,pcs_processadas)
myf.reconstruir_modelo_com_poses(Poses_Ajustadas_SLERP_LUM,pcs_processadas)
# Groundtruth:
myf.reconstruir_modelo_com_poses(Poses_Groundtruth[1:13],pcs_processadas)
'''
# Salvar lista de translacoes numa matriz:
matriz_translacoes_poses = np.zeros(3)
for i in range(n_nuvens-1):
    XYZ = Poses_Ajustadas_SLERP_LUM[i][0:3,3]
    matriz_translacoes_poses = np.hstack((matriz_translacoes_poses,XYZ))
matriz_translacoes_poses = np.reshape(matriz_translacoes_poses,[n_nuvens,3])
np.savetxt("poses_pelo_icp_com_SLERP_LUM\matriz_translacoes_poses.txt",matriz_translacoes_poses,fmt="%.10f")
'''
# COMPARAR POSES GROUNDTRUTH com ORIGINAIS | LUM | SLERP | SLERP+LUM
RMSE_antes, RMSE_depois_LUM, RMSE_depois_SLERP, RMSE_depois_SLERP_LUM = [],[],[],[]
RMSE_antes_R, RMSE_depois_SLERP_R = [],[]
for i in range(1,n_nuvens):
    Poses_Groundtruth = np.loadtxt("finalposes/scan%d.dat" %i)
    Diferenca_antes            = Poses_Groundtruth - Poses_Normais[i-1]
    Diferenca_depois_LUM       = Poses_Groundtruth - Poses_Ajustadas_LUM[i-1]
    Diferenca_depois_SLERP     = Poses_Groundtruth - Poses_Ajustadas_SLERP[i-1]
    Diferenca_depois_SLERP_LUM = Poses_Groundtruth - Poses_Ajustadas_SLERP_LUM[i-1]
    # Diferencas na Translacao:
    RMSE           = np.sqrt(sum(Diferenca_antes[0:3,3]**2))
    RMSE_LUM       = np.sqrt(sum(Diferenca_depois_LUM[0:3,3]**2))
    RMSE_SLERP     = np.sqrt(sum(Diferenca_depois_SLERP[0:3,3]**2))
    RMSE_SLERP_LUM = np.sqrt(sum(Diferenca_depois_SLERP_LUM[0:3,3]**2))
    # Diferencas na Rotacao:
    RMSE_R          = np.sqrt(sum(sum(Diferenca_antes[0:3,0:3]**2))) # Norma de Frobenius
    RMSE_SLERP_R    = np.sqrt(sum(sum(Diferenca_depois_SLERP[0:3,0:3]**2)))
    # Salvar RMSE das Translacoes:
    RMSE_antes.append(RMSE)
    RMSE_depois_LUM.append(RMSE_LUM)
    RMSE_depois_SLERP.append(RMSE_SLERP)
    RMSE_depois_SLERP_LUM.append(RMSE_SLERP_LUM)
    # Salvar RMSE das Rotacoes:
    RMSE_antes_R.append(RMSE_R)
    RMSE_depois_SLERP_R.append(RMSE_SLERP_R)
print(f"RMSE antes: {sum(RMSE_antes)}\nRMSE com LUM {sum(RMSE_depois_LUM)}\nRMSE com SLERP {sum(RMSE_depois_SLERP)}\nRMSE com SLERP+LUM {sum(RMSE_depois_SLERP_LUM)}")
print(f"RMSE ROTACAO antes: {sum(RMSE_antes_R)}\nRMSE ROTACAO depois SLERP {sum(RMSE_depois_SLERP_R)}")

# salvar RMSE 
A = np.vstack((np.array(RMSE_antes),np.array(RMSE_depois_LUM)))
B = np.vstack((np.array(RMSE_depois_SLERP),np.array(RMSE_depois_SLERP_LUM)))
C = np.vstack((A,B))
np.savetxt("RMSE_das_poses_abcd.txt",C,fmt="%.3f",delimiter=";")
# rotacoes
A = np.vstack((np.array(RMSE_antes_R),np.array(RMSE_depois_SLERP_R)))
np.savetxt("RMSE_das_rotacoes.txt",A,fmt="%.3f",delimiter=";")

