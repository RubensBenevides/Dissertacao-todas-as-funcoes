import numpy as np
import copy
import time
import open3d as o3d
import quaternion as quat

def carregar_nuvens_e_pre_processar(n_nuvens,voxel_size,knn,std):
    nuvens_processadas = []
    for i in range(n_nuvens):
        nuvem = o3d.io.read_point_cloud("nuvens/s%d.pcd" % i)
        nuvem_amostrada = nuvem.voxel_down_sample(voxel_size=voxel_size)
        nuvem_filtrada, ind = nuvem_amostrada.remove_statistical_outlier(nb_neighbors=knn,std_ratio=std)
        cor = np.random.rand(3)
        nuvem_filtrada.paint_uniform_color(cor**2)
        print(f"{nuvem} Amostrada e filtrada: {nuvem_filtrada}")
        nuvens_processadas.append(nuvem_filtrada)
    return nuvens_processadas

def reconstruir_modelo_com_poses(lista_poses,lista_nuvens):
    n_nuvens = len(lista_nuvens)
    copias = copy.deepcopy(lista_nuvens) # criar copias para manter originais no lugar
    for i in range(n_nuvens-1):
        # aplicar poses nas copias
        copias[i+1].transform(lista_poses[i])
    # desenhar nuvens transformadas
    o3d.visualization.draw_geometries(copias)

def desenhar_poses_com_linhas_e_frames(lista_geometria,line_set,campo_de_visao):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    # adiciona cada geometria na visualizacao:
    for i in range(len(lista_geometria)):
        vis.add_geometry(lista_geometria[i])
    # adiciona o conjunto de linhas na visualizacao:
    vis.add_geometry(line_set)
    ctr = vis.get_view_control()
    # define o campo de visao para 5 graus:
    ctr.change_field_of_view(step = campo_de_visao)
    vis.run()
    vis.destroy_window()

# Essa funcao recebe uma matriz com as translacoes das poses
# e uma lista com as matrizes de rotacoes destas poses
def criar_linhas_e_frames_3D(matriz_translacoes,lista_rotacoes,tamanho):
    lista_de_frames = []
    pontos = matriz_translacoes # pontos sas as tranlacoes das poses
    linhas = []
    for i in range(len(matriz_translacoes)):
        # Definir linhas:
        if i < len(matriz_translacoes):
            linha = [i,i+1]
            linhas.append(linha)
        elif i == len(matriz_translacoes)-1:
            linha = [i,0]
            linhas.append(linha)
        # Montar frames 3D coloridos (eixos):
        t = matriz_translacoes[i,:]
        R = lista_rotacoes[i]
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=tamanho, origin=t)
        mesh_frame.rotate(R)
        lista_de_frames.append(mesh_frame)
    # montar conjunto de linhas:
    conjunto_linhas = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(pontos),
                                            lines=o3d.utility.Vector2iVector(linhas))
    return lista_de_frames , conjunto_linhas

# Funcao que transforma quaternio(4x1) + t(3x1) em matriz homogenea T(4x4)
def transformar_quaternio_em_4x4(quaternio,translacao):
    aux = np.array([0,0,0,1])
    rotacao3x4 = np.hstack((quat.as_rotation_matrix(quaternio), np.transpose([translacao])))
    T = np.vstack((rotacao3x4,aux))
    return T

# Funcao que interpola entre duas transformacoes 0 e 1
def Interpolar_Poses(T1,T2,intervalo):
    # Media da translacao
    if (T2[0,3] > T1[0,3]):
        x_medio = np.interp(intervalo*(T2[0,3]-T1[0,3]), [T1[0,3], T2[0,3]], [T1[0,3], T2[0,3]] )
    else:
        x_medio = np.interp(-intervalo*(T1[0,3]-T2[0,3]), [T2[0,3], T1[0,3]], [T2[0,3], T1[0,3]] )
    if (T2[1,3] > T1[1,3]):
        y_medio = np.interp(intervalo*(T2[1,3]-T1[1,3]), [T1[1,3], T2[1,3]], [T1[1,3], T2[1,3]] )
    else:
        y_medio = np.interp(-intervalo*(T1[1,3]-T2[1,3]), [T2[1,3], T1[1,3]], [T2[1,3], T1[1,3]] )
    if (T2[2,3] > T1[2,3]):
        z_medio = np.interp(intervalo*(T2[2,3]-T1[2,3]), [T1[2,3], T2[2,3]], [T1[2,3], T2[2,3]] )
    else:
        z_medio = np.interp(-intervalo*(T1[2,3]-T2[2,3]), [T2[2,3], T1[2,3]], [T2[2,3], T1[2,3]] )
    t_media = np.reshape([x_medio,y_medio,z_medio],(3,1)) 
    # Transformar matrizes de rotacoes para quaternios 
    q1 = quat.from_rotation_matrix(T1[0:3,0:3])
    q2 = quat.from_rotation_matrix(T2[0:3,0:3])
    # Calcular rotacao media por SLERP em 0.5
    q_media = quat.quaternion_time_series.slerp(q1,q2,0,1,intervalo)
    # Montar T com R(3x3) + t(3x1)
    rotacao3x4 = np.hstack((quat.as_rotation_matrix(q_media), t_media))
    T_media = np.vstack((rotacao3x4,np.array([0,0,0,1])))
    return T_media

def planificar_nuvens_em_xy(lista_nuvens):
    lista_nuvens_planas = []
    for i in range(len(lista_nuvens)):
        xyz = np.asarray(lista_nuvens[i].points)
        xyz[:,2] = 0
        nuvem_plana = o3d.geometry.PointCloud()
        nuvem_plana.points = o3d.utility.Vector3dVector(xyz)
        lista_nuvens_planas.append(nuvem_plana)
    return lista_nuvens_planas

# Funcao para inverter uma pose:
def Transformar_de_volta(T_4x4):
    R_inv = np.transpose(T_4x4[0:3,0:3])
    t_inv = np.transpose([-R_inv@T_4x4[0:3,3]])
    T_inv = np.vstack((np.hstack((R_inv,t_inv)), np.array([0,0,0,1])))
    return T_inv

# Funcao para calcular matriz do LoopClosure
def Calcular_Erro_LoopClosure(array_circuito_Ts):
    n_nuvens = len(array_circuito_Ts)
    LoopClosure = np.identity(3)
    for i in range (n_nuvens-1,-1,-1):
        Ri = array_circuito_Ts[i][0:3,0:3] # pega apenas a rotacao da T
        LoopClosure = LoopClosure@Ri       # multiplica todas as n rotacoes
    # Calcular o erro no loop (distancia entre a matriz LoopClosure e a identidade)
    qloop = quat.from_rotation_matrix(LoopClosure)
    ErroNoLoop = np.linalg.norm(LoopClosure-np.identity(3),'fro')
    print(f"Distancia (Frobenious) da matriz LoopClosure para a identidade:\n{ErroNoLoop}")
    print(f"Matriz de rotacao LoopClosure:\n{LoopClosure}") 
    print(f"Quaternio LoopClosure:\n{qloop}")

# Funcao para desenhar o resultado do registro e preservar a nuvem transformada
def desenhar_resultado_registro(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color(np.random.rand(3))
    target_temp.paint_uniform_color(np.random.rand(3))
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def carregar_nuvens_originais(n_nuvens,a_cada_k_pontos):
    nuvens_originais = []
    for i in range(n_nuvens):
        nuvem = o3d.io.read_point_cloud("nuvens/s%d.pcd" % i)
        nuvem_a_cada_k_pontos = nuvem.uniform_down_sample(every_k_points=a_cada_k_pontos)
        print(f"{nuvem} Pontos selecionados: {nuvem_a_cada_k_pontos}")
        nuvens_originais.append(nuvem_a_cada_k_pontos)
    return nuvens_originais

def Fast_Global_Registration(source,target,knn_normal,knn_descritor,mu,distance):
    # Constroi kd-trees para calcular as normais das nuvens:
    #print(f"Calculando normais com {knn_normal} knn...")
    kd_tree_normais = o3d.geometry.KDTreeSearchParamKNN(knn_normal)
    source.estimate_normals(kd_tree_normais)
    target.estimate_normals(kd_tree_normais)
    # Constroi kd-trees para calculad descritores FPFH das nuvens:
    #print(f"Calculando descritores FPFH com {knn_descritor} knn")
    kd_tree_descritores = o3d.geometry.KDTreeSearchParamKNN(knn_descritor)
    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(source,kd_tree_descritores)
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(target,kd_tree_descritores)
    # Define os parâmetros do registro FGR:
    FGR_coarse=o3d.pipelines.registration.FastGlobalRegistrationOption(
                                            division_factor = mu,       # padrao: 1.4 
                                            use_absolute_scale = True,  # padrao: False
                                            decrease_mu = True,         # padrao: False
                                            maximum_correspondence_distance = distance,
                                            iteration_number = 100,     # padrao: 64
                                            tuple_scale = 0.95,         # padrao: 0.95
                                            maximum_tuple_count = 1000) # padrao: 640 
    # Aplica o registro FGR no par de nuvens source-target:
    #print("Aplicando Registro FGR\n")
    result_FGR = o3d.pipelines.registration.registration_fast_based_on_feature_matching(
                                                source,
                                                target,
                                                source_fpfh,
                                                target_fpfh,
                                                FGR_coarse)
    return result_FGR

def registrar_com_ICP(source,target,T_inicial,iteracoes,knn,distancia,tipo):
    if tipo == 'plano':
        # Constroi kd-trees e calcula normais das nuvens:
        kd_tree_normais = o3d.geometry.KDTreeSearchParamKNN(knn)
        source.estimate_normals(kd_tree_normais)
        target.estimate_normals(kd_tree_normais)
        print(f"Aplicando ICP robusto ({iteracoes} iteracoes)\n")
        loss = o3d.pipelines.registration.L1Loss()
        result_ICP = o3d.pipelines.registration.registration_icp(source,
                        target,
                        distancia, # distancia corr. que usamos pra avaliar o registro
                        T_inicial,
                        o3d.pipelines.registration.TransformationEstimationPointToPlane(loss),
                        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration = iteracoes))
    if tipo == 'ponto':
        print(f"Aplicando ICP ponto-a-ponto com {iteracoes} iteracoes...")
        result_ICP = o3d.pipelines.registration.registration_icp(source,
                        target,
                        distancia, # voxel que usamos pra avaliar o registro
                        T_inicial,
                        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), # false = T rigida (escala=1)
                        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration = iteracoes))
    return result_ICP

def cortar_nuvem_manual():
    print("Corte manual de geometrias")
    print("1) Pressione 'Y' duas vezes para alinhar na direção do eixo-y")
    print("2) Pressione 'K' para travar a tela e mudar para o modo de seleção")
    print("3) Arraste para seleção retangular,")
    print("   ou use ctrl + botão esquerdo para seleção poligonal")
    print("4) Pressione 'C' para salvar uma geometria selecionada")
    print("5) Pressione 'F' para o modo livre de visualização")
    pcd = o3d.io.read_point_cloud("pc0.pcd")
    o3d.visualization.draw_geometries_with_editing([pcd])

def escolher_pontos(pcd):
    print("")
    print("1) Selecione pelo menos 3 correspondências usando [shift + left click]")
    print("   Pressione [shift + right click] para desfazer a seleção")
    print("2) Depois de clicar nos pontos pressione 'Q' para fechar a janela")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # pontos selecionados pelo usuário
    vis.destroy_window()
    print("")
    return vis.get_picked_points()

def registro_manual(source,target):
    print("Registro manual")
    print("Visualizacao antes do registro")
    desenhar_resultado_registro(source, target, np.identity(4))
    # Escolha pontos de duas nuvens e crie correspondencias
    picked_id_source = escolher_pontos(source)
    picked_id_target = escolher_pontos(target)
    # Minimo de 3 pontos
    assert (len(picked_id_source) >= 3 and len(picked_id_target) >= 3)
    assert (len(picked_id_source) == len(picked_id_target))
    corr = np.zeros((len(picked_id_source), 2))
    corr[:, 0] = picked_id_source
    corr[:, 1] = picked_id_target
    # Estimar transformacao grosseira usando correspondencias
    print("Calculando transformação usando pontos escolhidos")
    p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    T_inicial = p2p.compute_transformation(source,target,o3d.utility.Vector2iVector(corr))
    desenhar_resultado_registro(source, target, T_inicial)
    print("")
    return T_inicial

def Montar_Matriz_Diagonal_Pesos(Pesos,n_nuvens):
    # Pesos deve ser do tipo lista 
    P = Pesos[0]*np.ones(3)
    for i in range(1,n_nuvens):
        P_aux = np.ones(3)*Pesos[i]
        P = np.hstack((P,P_aux))
    P = np.diagflat(P)
    return P

def Montar_Vetor_Lb_translacoes(lista_translacoes,lista_rotacoes_origem,n_nuvens):
    # Montar vetor Lb de 3*(n_nuvens) linhas por 1 coluna:
    Lb = lista_translacoes[0]
    LoopClosure_Translacao = lista_translacoes[0]
    for i in range(n_nuvens-1):
        aux_Lb = lista_rotacoes_origem[i]@lista_translacoes[i+1]
        LoopClosure_Translacao = LoopClosure_Translacao + aux_Lb
        Lb = np.hstack([Lb,aux_Lb]) # vetor de Lb com as translacoes rotacionadas
    Lb = np.reshape(Lb,(len(Lb),1)) # organiza o vetor em uma coluna
    return [Lb,LoopClosure_Translacao]

def Ajustamento_Quaternios_SLERP(lista_quat):
    n_nuvens = len(lista_quat)
    # Acumular rotacoes para a origem multiplicativamente (ordem direta):
    lista_quat_origem = []
    for i in range(n_nuvens):
        ri = quat.from_rotation_matrix(np.identity(3))
        for j in range (n_nuvens-i-1,-1,-1):  # conta de cima para baixo
            ri = ri*lista_quat[j]          # acumula j rotacoes multiplicativamente
        lista_quat_origem.append(ri)       # primeira rotacao da lista eh a LoopClosure
    lista_quat_origem = list(reversed(lista_quat_origem)) # inverte a ordem da lista
    # obter lista anterior pela ordem inversa de multuplicacao dos quaternios:
    quat_LoopClosure = lista_quat_origem[n_nuvens-1]
    lista_quat_origem_inv = []
    for i in range(n_nuvens-1):
        pi = lista_quat_origem[i]*quat_LoopClosure**(-1)
        lista_quat_origem_inv.append(pi)
    # AJUSTAMENTO - APLICAR SLERP ENTRE AS DUAS LISTAS DE QUATERNIOS
    lista_quat_ajustado = []
    for i in range(n_nuvens-1):
        qi_ajustado = quat.quaternion_time_series.slerp(lista_quat_origem[i],lista_quat_origem_inv[i],0,1,t_out=(i+1)/n_nuvens)
        lista_quat_ajustado.append(qi_ajustado)
    return [lista_quat_ajustado, quat_LoopClosure]

# FUNCAO QUE RETORNA POSES SEM AJUSTAMENTO:
def reconstruir_Ts_para_origem(T_circuito):
    n_nuvens = len(T_circuito)
    lista_translacoes,lista_rotacoes = [],[]
    # obter lista de rotacoes e translacoes do circuito:
    for i in range(n_nuvens):
        lista_translacoes.append(T_circuito[i][0:3,3])
        lista_rotacoes.append(T_circuito[i][0:3,0:3])
    # obter lista de rotacoes para a origem por composicao multiplicativa
    lista_rotacoes_origem = []
    for i in range(n_nuvens):
        LoopClosure = np.identity(3)
        for j in range (n_nuvens-i-1,-1,-1):        
            LoopClosure = LoopClosure@lista_rotacoes[j] # compoe j matrizes de rotacao
        lista_rotacoes_origem.append(LoopClosure) # lista de rotacoes compostas para origem
    lista_rotacoes_origem = list(reversed(lista_rotacoes_origem)) # inverte a lista, ultima eh a LoopClosure
    # obter as translacoes para a origem (t20, t30, t40, etc.)
    t_ini = lista_translacoes[0]
    lista_translacoes_origem = [t_ini]
    for i in range(n_nuvens-1):
        t_posterior = lista_rotacoes_origem[i]@lista_translacoes[i+1] + t_ini
        t_ini = t_posterior
        lista_translacoes_origem.append(t_posterior)
    # Juntar (rotacao + translacao) em T(4x4):
    Lista_de_Poses = []
    for i in range(n_nuvens):
        rotacao_translacao = np.hstack((lista_rotacoes_origem[i],np.transpose([lista_translacoes_origem[i]])))
        Pose4x4 = np.vstack((rotacao_translacao,np.array([0,0,0,1])))
        Lista_de_Poses.append(Pose4x4)
    return Lista_de_Poses

# FUNCAO QUE RETORNA POSES COM ROTACOES SLERPADAS:
def reconstruir_Ts_para_origem_SLERP(T_circuito):
    n_nuvens = len(T_circuito)
    lista_translacoes, lista_quat = [],[]
    # obter rotacoes e translacoes do circuito:
    for i in range(n_nuvens):
        lista_translacoes.append(T_circuito[i][0:3,3])
        lista_quat.append(quat.from_rotation_matrix(T_circuito[i][0:3,0:3]))
    # AJUSTAMENTO DOS QUATERNIOS POR SLERP
    lista_quat_ajustado, quat_LoopClosure = Ajustamento_Quaternios_SLERP(lista_quat)
    # Obter as translacoes de cada nuvem para a origem
    t_ini = lista_translacoes[0]
    lista_translacoes_origem = [lista_translacoes[0]]
    for i in range(n_nuvens-1):
        t_posterior = quat.as_rotation_matrix(lista_quat_ajustado[i])@lista_translacoes[i+1] + t_ini
        t_ini = t_posterior # t10 recebe t20 e continua
        lista_translacoes_origem.append(t_posterior)
    # Juntar (rotacoes slerpadas + translacoes) em pose T(4x4):
    Poses_SLERP = []
    aux_0001 = np.array([0,0,0,1])
    for i in range(n_nuvens-1):
        Matriz_R = quat.as_rotation_matrix(lista_quat_ajustado[i]) # Quaternios -> Matriz Rotacao
        rotacao_translacao = np.hstack((Matriz_R, np.transpose([lista_translacoes_origem[i]])))
        Pose = np.vstack((rotacao_translacao, aux_0001))
        Poses_SLERP.append(Pose)
    # Pose LoopClosure:
    Translacao_LoopClosure = np.transpose([lista_translacoes_origem[n_nuvens-1]])
    Matriz_R_LoopClosure = quat.as_rotation_matrix(quat_LoopClosure) # Quaternio Loopclosure -> Matriz Rotacao
    rotacao_translacao = np.hstack((Matriz_R_LoopClosure, Translacao_LoopClosure))
    Pose_LoopClosure = np.vstack((rotacao_translacao,aux_0001))
    Poses_SLERP.append(Pose_LoopClosure)
    return Poses_SLERP

# FUNCAO QUE RETORNA POSES COM AJUSTE NA TRANSLACAO:
def reconstruir_Ts_para_origem_LUM(T_circuito,Pesos):
    n_nuvens = len(T_circuito)
    lista_translacoes,lista_rotacoes = [],[]
    # obter lista de rotacoes e translacoes do circuito:
    for i in range(n_nuvens):
        lista_translacoes.append(T_circuito[i][0:3,3])
        lista_rotacoes.append(T_circuito[i][0:3,0:3])
    # obter lista de rotacoes para a origem por composicao multiplicativa
    lista_rotacoes_origem = []
    for i in range(n_nuvens):
        LoopClosure = np.identity(3)
        for j in range (n_nuvens-i-1,-1,-1):        
            LoopClosure = LoopClosure@lista_rotacoes[j] # acumula j matrizes rotacoes
        lista_rotacoes_origem.append(LoopClosure) # primeira rotacao da lista eh a LoopClosure
    lista_rotacoes_origem = list(reversed(lista_rotacoes_origem)) # inverte a lista
    # AJUSTAMENTO DAS TRANSLACOES ROTACIONADAS SEGUNDO (LU & MILIOS, 1998)
    # Montar vetor Lb de 3*(n_nuvens) linhas em 1 coluna:
    Lb,Translacao_LoopClosure = Montar_Vetor_Lb_translacoes(lista_translacoes,lista_rotacoes_origem,n_nuvens)
    # Matriz digonal dos Pesos P:
    P = Montar_Matriz_Diagonal_Pesos(Pesos,n_nuvens)
    # Montar matriz A com 3*(n_nuvens) linhas e 3*(n_nuvens-1) colunas
    A = np.diagflat([-np.ones((n_nuvens-1)*3)],-3)  # cria uma matriz com -1 na diagonal -3
    A = np.delete(A,[np.arange(len(A)-3,len(A))],1) # remove as 3 ultimas colunas de A
    np.fill_diagonal(A, 1.0)                        # preenche a diagonal principal com 1  
    AtPA = np.transpose(A)@P@A
    N = np.linalg.inv(AtPA)              # Matriz N = inv(A'PA) 
    U = np.transpose(A)@P@Lb             # Matriz U = A'Lb
    X = N@U                              # Matriz X de translacoes ajustadas para a origem
    V = -A@X+Lb                          # Residuo. O mesmo que sum(Lb) a soma das translacoes
    print(f"Sigma_Posteriori (Vt*P*V)/GL do ajustamento LUM: {(np.transpose(V)@P@V)/3} GL = 3")
    # Juntar (rotacao + translacao ajustada LUM) em T(4x4): 
    Poses_ajustadas_LUM = []
    aux_0001 = np.array([0,0,0,1])
    for i in range(n_nuvens-1):
        rotacao_translacao = np.hstack((lista_rotacoes_origem[i], X[3*i:3*(i+1)]))
        Pose = np.vstack((rotacao_translacao, aux_0001))
        Poses_ajustadas_LUM.append(Pose)
    # Pose LoopClosure:
    rotacao_translacao = np.hstack((lista_rotacoes_origem[n_nuvens-1], np.transpose([Translacao_LoopClosure])))
    Pose_LoopClosure = np.vstack((rotacao_translacao, aux_0001))
    Poses_ajustadas_LUM.append(Pose_LoopClosure)
    return Poses_ajustadas_LUM

# FUNCAO QUE RETORNA POSES AJUSTADAS POR SLERP(ROTACAO) + LUM(TRANSLACAO)
def reconstruir_Ts_para_origem_SLERP_LUM(T_circuito,Pesos):
    n_nuvens = len(T_circuito)
    lista_translacoes, lista_quat = [],[]
    # obter rotacoes e translacoes do circuito:
    for i in range(n_nuvens):
        lista_translacoes.append(T_circuito[i][0:3,3])
        lista_quat.append(quat.from_rotation_matrix(T_circuito[i][0:3,0:3]))
    # AJUSTAMENTO DOS QUATERNIOS POR SLERP
    lista_quat_ajustado, quat_LoopClosure = Ajustamento_Quaternios_SLERP(lista_quat)
    # AJUSTAMENTO DAS TRANSLACOES ROTACIONADAS SEGUNDO (LU & MILIOS, 1998)
    # Retornar de quaternios para matrizes de rotacao quat -> Matrix
    lista_rotacoes_origem = []
    for i in range(n_nuvens-1):
        rotacao_origem = quat.as_rotation_matrix(lista_quat_ajustado[i])
        lista_rotacoes_origem.append(rotacao_origem)
    # Montar vetor Lb: 3*(n_nuvens-1)+3 observacoes em uma coluna:
    Lb,Translacao_LoopClosure = Montar_Vetor_Lb_translacoes(lista_translacoes,lista_rotacoes_origem,n_nuvens)
    # Matriz diagonal dos Pesos P [(3*n_nuvens)x(3*n_nuvens)]:
    P = Montar_Matriz_Diagonal_Pesos(Pesos,n_nuvens)
    # Montar matriz A [(3*n_nuvens) x 3*(n_nuvens-1)]
    A = np.diagflat([-np.ones((n_nuvens-1)*3)],-3)  # cria uma matriz com -1 na diagonal -3
    A = np.delete(A,[np.arange(len(A)-3,len(A))],1) # remove as 3 ultimas colunas de A
    np.fill_diagonal(A, 1.0)                        # preenche a diagonal principal com 1's  
    AtPA = np.transpose(A)@P@A
    N = np.linalg.inv(AtPA)              # Matriz N = inv(A'PA) 
    U = np.transpose(A)@P@Lb             # Matriz U = A'Lb
    X = N@U                              # Translacoes ajustadas para a origem
    V = -A@X+Lb                          # Residuo. O mesmo que sum(Lb) a soma das translacoes
    print(f"Sigma_Posteriori (Vt*P*V)/GL do ajustamento LUM: {np.transpose(V)@P@V/3} GL = 3")
    # Juntar (rotacao slerpada + translacao ajustada LUM) em uma pose T(4x4): 
    Lista_Poses_Ajustadas_SLERP_LUM = []
    aux_0001 = np.array([0,0,0,1])
    for i in range(n_nuvens-1):
        rotacao_translacao = np.hstack((quat.as_rotation_matrix(lista_quat_ajustado[i]), X[i*3:3*(i+1)]))
        Pose = np.vstack((rotacao_translacao, aux_0001))
        Lista_Poses_Ajustadas_SLERP_LUM.append(Pose)
    # Pose LoopClosure:
    rotacao_translacao = np.hstack((quat.as_rotation_matrix(quat_LoopClosure), np.transpose([Translacao_LoopClosure])))
    Pose_LoopClosure = np.vstack((rotacao_translacao, aux_0001))
    Lista_Poses_Ajustadas_SLERP_LUM.append(Pose_LoopClosure)
    return Lista_Poses_Ajustadas_SLERP_LUM

# Esta funcao anima a reconstrucao de um dataset movendo as nuvens uma de cada vez
def Reconstrucao_animada_uma_de_cada_vez(Lista_nuvens,Lista_poses,n_frames):
    # Inicializar n de frames, se os quer salvar, visualizador e janela
    N = n_frames # N de frames
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    # Adicionar nuvens na visualizacao:
    for i in range(len(Lista_nuvens)):
        vis.add_geometry(Lista_nuvens[i])  # ! ADICIONAR A NUVEM 0 PRIMEIRO !
    # Interpolar entre poses e a identidade n vezes
    Todos_os_frames = []
    for i in range(1,len(Lista_poses)):
        N_frames_de_uma_pose = [Interpolar_Poses(Lista_poses[0],Lista_poses[i],(j+1)/N) for j in range(N)]
        # Tem-se Poses*N frames em Todos_os_frames, que eh uma lista de lista
        Todos_os_frames.append(N_frames_de_uma_pose)
    for i in range(len(Todos_os_frames)):
        for j in range(len(Todos_os_frames[i])):
            Lista_nuvens[i+1].transform(Todos_os_frames[i][j])
            vis.update_geometry(Lista_nuvens[i+1])
            vis.poll_events()
            vis.update_renderer()
            # Eh necessario transformar a nuvem i de volta:
            frame_inverso = Transformar_de_volta(Todos_os_frames[i][j])
            Lista_nuvens[i+1].transform(frame_inverso)
        time.sleep(1)

# Esta funcao anima a reconstrucao de um dataset movendo todas as nuvens de uma vez
def Reconstrucao_animada_todas_de_uma_vez(Lista_nuvens,Lista_poses,n_frames):
    n_nuvens = len(Lista_nuvens)
    n_colunas = 4
    n_linhas = 4*(n_nuvens-1)*n_frames
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    # Adicionar nuvens na visualizacao:
    for nuvem in range(len(Lista_nuvens)):
        vis.add_geometry(Lista_nuvens[nuvem])  # ! ADICIONAR A NUVEM 0 PRIMEIRO !
    # Inicializar matriz que recebe todos os n frames interpolados em cada pose 
    Todos_os_frames = np.zeros((n_linhas,n_colunas))
    # Loop dos frames:
    for i in range(n_frames):
        # Inicializar frame inicial de cada pose
        frames_poses = np.zeros((4*(n_nuvens-1),n_colunas))
        # Loop do frame para cada pose:
        for j in range(1,n_nuvens):
            # Interpolar entre cada pose e a identidade i vezes (n_frames)
            frame_j = Interpolar_Poses(Lista_poses[0], Lista_poses[j], (i+1)/n_frames)
            # Salvar j frames de cada pose
            frames_poses[4*j-4:4*j,:] = frame_j
        # Salvar frames de cada pose de j em j
        Todos_os_frames[4*(n_nuvens-1)*i:4*(n_nuvens-1)*(i+1),:] = frames_poses
        # Todos_os_frames eh uma matriz 1600x4 = (4x4 * n_frames * n_nuvens-1)
    for i in range(n_frames):
        for j in range(1,n_nuvens):
            Lista_nuvens[j].transform(Todos_os_frames[(4*(n_nuvens-1)*i)+(4*j-4):(4*(n_nuvens-1)*i)+(4*j),:])    
            vis.update_geometry(Lista_nuvens[j])
            vis.poll_events()
            vis.update_renderer()
            # Eh necessario transformar cada nuvem de volta:
            frame_inverso = Transformar_de_volta(Todos_os_frames[(4*(n_nuvens-1)*i)+(4*j-4):(4*(n_nuvens-1)*i)+(4*j),:])
            Lista_nuvens[j].transform(frame_inverso)
    time.sleep(5)