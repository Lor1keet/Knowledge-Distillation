import torch
import random
import numpy as np
import networkx as nx
from tqdm import tqdm
import os
import math
import matplotlib.pyplot as plt


def get_random_carp_problems(batch_size, vertex_size, edge_size, device, distribution, load_path=None, episode=None, max_dhcost=10, max_demand=10, min_degree=None, max_degree=None):
    """
    随机生成 CARP 问题实例，或者从文件中加载问题。
    
    参数:
    ----------
    batch_size : int
        批次大小
    vertex_size : int
        图中的节点数量
    edge_size : int
        图中的边数量
    distribution : dict
        用于生成图的分布参数
    load_path : str, optional
        如果不为 None，则从该路径加载问题
    episode : int, optional
        如果不为 None，指定从哪个 episode 加载问题

    返回:
    ----------
    depot_features : torch.Tensor
        仓库的特征 (batch, 1, num_features)
    customer_features : torch.Tensor
        客户的特征 (batch, edge_size, num_features)
    graph_dynamic : torch.Tensor
        图的动态信息 (batch, edge_size)
    graph_info : torch.Tensor
        图的信息，包含原始图中的节点和边信息 (batch, 4, edge_size + 1)
    D : torch.Tensor
        每张图的节点之间的最短路径邻接矩阵 (batch, vertex_size, vertex_size)
    A : torch.Tensor
        图的邻接矩阵 (batch, edge_size + 1, edge_size + 1)
    """
    if load_path is not None:
        # 从文件中加载问题实例
        import pickle
        filename = load_path
        assert os.path.splitext(filename)[1] == '.pkl'
        with open(filename, 'rb') as f:
            data = torch.load(f)

        if episode is not None:
            data = data[episode: episode + batch_size]

        depot_features, customer_features, graph_dynamic, graph_info, D_tensor, A_tensor = [], [], [], [], [], []
        
        for i in range(len(data)):
            depot_features.append(torch.FloatTensor(data[i][0]))
            customer_features.append(torch.FloatTensor(data[i][1]))
            graph_dynamic.append(torch.FloatTensor(data[i][2]))
            graph_info.append(torch.FloatTensor(data[i][3]))
            D_tensor.append(torch.FloatTensor(data[i][4]))
            A_tensor.append(torch.FloatTensor(data[i][5]))
        
        depot_features = torch.stack(depot_features, dim=0)
        customer_features = torch.stack(customer_features, dim=0)
        graph_dynamic = torch.stack(graph_dynamic, dim=0)
        graph_info = torch.stack(graph_info, dim=0)
        D = torch.stack(D_tensor, dim=0)
        A = torch.stack(A_tensor, dim=0)

    else:
        num_samples = batch_size
        min_degree = 1
        max_degree = 5
        
        if vertex_size == 10 and edge_size == 20:
            max_load = 30
        elif vertex_size == 30 and edge_size == 60:
            max_load = 60
        elif vertex_size == 50 and edge_size == 100:
            max_load = 100
        else:
            raise NotImplementedError
        
        print("车载容量: ",max_load)
        print("最大需求: ",max_demand)
        print("节点度范围: (%d,%d)"% (min_degree, max_degree))

    

        name = "node" + str(vertex_size) + "edge" + str(edge_size) + "_" + "features"
        file_path = os.path.join("mapinfo", name)
        file_path = os.path.join(file_path, str(num_samples))

        node_features = torch.zeros((num_samples, edge_size + 1, 6), dtype=torch.float32)
        dynamic = torch.zeros((num_samples, edge_size + 1), dtype=torch.float32)
        graph_info_ori = torch.zeros((num_samples, edge_size + 1, 5))
        D_tensor = torch.zeros((num_samples, vertex_size, vertex_size))
        A_tensor = torch.zeros((num_samples, edge_size + 1, edge_size + 1))
 
        if num_samples < 10000 or not os.path.exists(file_path):
           for sample in tqdm(range(num_samples), desc="Processing graphs"):
                 # 生成随机图
                G, depot, total_cost, total_demand = generate_graph_degree(vertex_size, edge_size, distribution,
                                                                            max_dhcost, max_demand, min_degree=min_degree, max_degree=max_degree)
               
                # 计算最短路径邻接矩阵
                D, _ = floyd(G)
                D_tensor[sample, :, :] = torch.tensor(D)
                # 使用 edge2vertex 将边转换为节点
                vertex_graph = edge2vertex(G, depot)

                i = 0
                for node, attributes in vertex_graph.nodes(data=True):
                        # 原始边的权重与需求值
                        edge_dhcost = attributes['dhcost']
                        edge_demand = attributes['demand']
                        # 原始图中边的两个端点
                        node_ori_1 = attributes['node_ori'][0]
                        node_ori_2 = attributes['node_ori'][1]
                        # 将每条边是否与仓库节点相连体现在特征 f_node_ori 上
                        f_node_ori_1 = 1 if node_ori_1 == depot else 0
                        f_node_ori_2 = 1 if node_ori_2 == depot else 0

                        node_feature = [f_node_ori_1,
                                        f_node_ori_2,
                                        # 从仓库节点到边的两个端点的最短距离
                                        D[depot][node_ori_1],
                                        D[depot][node_ori_2],
                                        # 归一化后的边权重与需求值
                                        edge_dhcost/total_cost,
                                        edge_demand/max_load
                                        ]
                        # 当前边需求值的归一化，用于后续训练中动态变化的状态
                        dynamic_np = [edge_demand/max_load]
                        # 图信息，保存节点和边的相关原始信息
                        graph_info = [node, node_ori_1, node_ori_2, edge_dhcost, edge_demand]
                        node_features[sample, i, :] = torch.tensor(node_feature)
                        dynamic[sample, i] = torch.tensor(dynamic_np)
                        graph_info_ori[sample, i, :] = torch.tensor(graph_info)
                        i += 1

                # 返回节点图的邻接矩阵
                adjacency_matrix = torch.tensor(nx.to_numpy_array(vertex_graph)).to(device)
                # 添加对角项全1（自环）
                E = torch.eye(adjacency_matrix.size(0))
                adjacency_matrix = adjacency_matrix + E
                # 对邻接矩阵的每一行进行求和，计算出每个节点的度数
                degree = torch.sum(adjacency_matrix, dim=1)
                # 将度数向量转换为对角矩阵，其中对角线上的元素为节点的度数，其他位置为0
                degree = torch.diag(degree).to(device)
                # 计算度矩阵的倒数平方根
                degree_inv_sqrt = torch.pow(degree, -0.5).to(device)
                degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0  # 处理度为0的情况，避免出现inf
                # 对称归一化，用于 GCN
                A = torch.matmul(torch.matmul(degree_inv_sqrt, adjacency_matrix), degree_inv_sqrt).float().to(device)
                A_tensor[sample, :, :] = A




        graph_info_ori = graph_info_ori.permute(0, 2, 1)  # [num_samples ,4, edge_size + 1]

        depot_features = node_features[:, 0, :] # [num_samples, _, num_features]
        customer_features = node_features[:, 1:, :] # [num_samples , edge_size, num_features]

        # self.depot_features = node_features[:, 0, 2:] # [num_samples, _, num_features]
        # self.customer_features = node_features[:, 1:, 2:] # [num_samples , edge_size, num_features]

        graph_dynamic = dynamic[:, 1:] # [num_samples, edge_size, 1]

        graph_info = graph_info_ori # [num_samples ,4, edge_size + 1]
        D = D_tensor  # 每张图的节点之间最短路径邻接矩阵  [num_samples ,vertex_size, vertex_size]
        A = A_tensor  # [num_samples ,edge_size + 1, edge_size + 1]

    return depot_features, customer_features, graph_dynamic, graph_info, D, A

def floyd(G):
    """
    Floyd-Warshall算法计算最短路径。

    参数:
    ----------
    G : networkx.Graph
        生成的无向图

    返回:
    ----------
    distance_matrix : numpy.ndarray
        最短路径距离矩阵
    path_matrix : numpy.ndarray
        最短路径中的中间节点
    """
    num_nodes = G.number_of_nodes()

    adj_matrix = np.full((num_nodes, num_nodes), np.inf)
    for node in G.nodes():
        adj_matrix[node, node] = 0
    for node1, node2, edge_data in G.edges(data=True):
        dhcost = edge_data.get('dhcost', 1) 
        adj_matrix[node1, node2] = dhcost
        adj_matrix[node2, node1] = dhcost

    num_nodes = len(adj_matrix)

    distance_matrix = np.copy(adj_matrix)
    path_matrix = np.ones((num_nodes, num_nodes), dtype=int) * -1

    for k in range(num_nodes):
        for i in range(num_nodes):
            for j in range(num_nodes):
                if distance_matrix[i, k] + distance_matrix[k, j] < distance_matrix[i, j]:
                    distance_matrix[i, j] = distance_matrix[i, k] + distance_matrix[k, j]
                    path_matrix[i, j] = k

    return distance_matrix, path_matrix

def edge2vertex(edge_graph, depot):
    """
    将边转换为节点，原始图中共享端点的边可以在新图中添加边表示它们的关联性。

    参数:
    ----------
    edge_graph : networkx.Graph
        原始的边图
    depot : int
        仓库节点的索引

    返回:
    ----------
    G : networkx.Graph
        转换后的点图
    """
    G = nx.Graph()
    edge_info_list = []
    edge_info_list.append((depot, depot, 0, 0))

    G.add_node(0, demand=0, dhcost=0, node_ori=(depot, depot))
    i = 1
    for node1, node2, edge_data in edge_graph.edges(data=True):
        demand = edge_data['demand']
        dhcost = edge_data['dhcost']
        G.add_node(i, demand=demand, dhcost=dhcost, node_ori=(node1, node2))
        i += 1
        edge_info_list.append((node1, node2, demand, dhcost))

    for index1, (index1_node1, index1_node2, index1_demand, index1_dhcost) in enumerate(edge_info_list):
        for index2 in range(index1 + 1, len(edge_info_list)):
            data_index2 = edge_info_list[index2]
            index2_node1 = data_index2[0]
            index2_node2 = data_index2[1]
            if (index1_node1 == index2_node1 or index1_node1 == index2_node2 or index1_node2 == index2_node1 or index1_node2 == index2_node2):
                G.add_edge(index1, index2)

    return G

def generate_graph_degree(vertex_size, edge_size, distribution, max_dhcost, max_demand, min_degree, max_degree):
    errorNum = 0
    
     # 如果 distribution 的类型为 small_world，则生成小世界网络
    if distribution['data_type'] == 'smallworld':
        k = distribution.get('k', 4)  # 每个节点连接的临近节点数
        p = distribution.get('p', 0.1)  # 重新连边的概率

        # 生成小世界网络
        G = nx.watts_strogatz_graph(n=vertex_size, k=k, p=p)
        
        # 为小世界网络生成节点坐标，限制在 [0, 1] 范围内
        pos = nx.spring_layout(G)  # 使用 spring 布局
        pos = {node: (x / max(1, max(x for x, y in pos.values())),
                      y / max(1, max(y for x, y in pos.values()))) for node, (x, y) in pos.items()}

        # 将坐标分配给节点
        nx.set_node_attributes(G, pos, 'pos')

        # 如果生成的边数不足或超出，调整边数
        G = adjust_edge_count(G, edge_size, max_dhcost, max_demand)

        # 初始化总成本和总需求
        total_cost, total_demand = 0, 0

        # 为每条边分配随机的路径成本和需求
        for (u, v) in G.edges():
            dhcost = random.randint(1, max_dhcost)
            demand = random.randint(1, max_demand)
            total_cost += dhcost
            total_demand += demand
            G[u][v]['dhcost'] = dhcost
            G[u][v]['demand'] = demand

        # 随机选择一个节点作为仓库
        depot = random.choice(list(G.nodes()))
        
        return G, depot, total_cost, total_demand
    
    
    else:
        # 根据分布类型生成节点坐标
        if distribution['data_type'] == 'uniform':
            depot_node_coords = np.random.rand(vertex_size, 2)
            use_random_edges = True
        elif distribution['data_type'] == 'cluster':
            depot_node_coords = generate_clustered_coords(vertex_size, distribution)
            use_random_edges = False
        else:
            raise ValueError("Unsupported distribution type!")

        while True:
            if errorNum == 100:
                raise ValueError("节点与度的数量设置不合理")

            total_cost, total_demand = 0, 0

            # 生成地图 根据点的度生成
            G = nx.Graph()
            G.add_nodes_from(range(vertex_size))

            # 为每个节点添加坐标
            for i in range(vertex_size):
                G.nodes[i]['pos'] = depot_node_coords[i]

            import heapq

            # 根据节点距离生成边的概率，只连接距离较近的节点
            def distance(node1, node2):
                pos1 = G.nodes[node1]['pos']
                pos2 = G.nodes[node2]['pos']
                pos1 = torch.tensor(pos1)
                pos2 = torch.tensor(pos2)
                return torch.norm(pos1 - pos2)  # 计算欧几里得距离

            # 为每个节点添加边，确保图是连通的
            for node in G.nodes():
                degree = random.randint(min_degree, max_degree)  # 获取当前节点的度

                if use_random_edges:
                    # 使用随机选择的方式连接边
                    neighbors = [n for n in G.nodes() if n != node and G.degree(n) < max_degree]
                    if len(neighbors) < degree:
                        neighbors = random.sample(neighbors, len(neighbors))  # 随机选择
                    else:
                        neighbors = random.sample(neighbors, degree)  # 随机选择指定数量的邻居
                else:
                    # 使用优先队列选择最小距离的邻居
                    heap = []
                    for neighbor in G.nodes():
                        if neighbor != node and G.degree(neighbor) < max_degree:
                            dist = distance(node, neighbor).item()  # 获取当前节点和邻居之间的距离
                            heapq.heappush(heap, (dist, neighbor))

                    # 选择邻居
                    neighbors = []
                    for _ in range(degree):
                        if heap:
                            _, neighbor = heapq.heappop(heap)  # 弹出最近的邻居
                            neighbors.append(neighbor)

                # 添加边
                for neighbor in neighbors:
                    dhcost = random.randint(1, max_dhcost)
                    total_cost += dhcost
                    demand = random.randint(1, max_demand)
                    total_demand += demand
                    G.add_edge(node, neighbor, dhcost=dhcost, demand=demand)

            # 确保图是连通的
            is_done = False
            iteration_count = 0
            while not is_done and iteration_count < 100:
                iteration_count += 1
                if G.number_of_edges() < edge_size:
                    # 添加边，优先连接距离较近的节点
                    add_nodes = [n for n in G.nodes() if G.degree(n) < max_degree]
                    if not add_nodes:
                        break
                    add_node = random.sample(add_nodes, k=1)[0]
                    neighbors = sorted([n for n in G.nodes() if n != add_node and G.degree(n) < max_degree and not G.has_edge(n, add_node)],
                                    key=lambda n: distance(add_node, n))
                    if not neighbors:
                        break
                    neighbor = neighbors[0]
                    dhcost = random.randint(1, max_dhcost)
                    total_cost += dhcost
                    demand = random.randint(1, max_demand)
                    total_demand += demand
                    G.add_edge(add_node, neighbor, dhcost=dhcost, demand=demand)

                elif G.number_of_edges() > edge_size:
                    delete_nodes = [n for n in G.nodes() if G.degree(n) > min_degree]
                    if not delete_nodes:
                        break
                    delete_node = random.sample(delete_nodes, k=1)[0]
                    neighbors = [n for n in G.neighbors(delete_node) if G.degree(n) > min_degree]
                    if not neighbors:
                        break
                    neighbor = random.sample(neighbors, k=1)[0]
                    edge_data = G.get_edge_data(delete_node, neighbor)
                    if edge_data and 'dhcost' in edge_data:
                        total_cost -= edge_data['dhcost']
                        total_demand -= edge_data['demand']
                    G.remove_edge(delete_node, neighbor)

                if G.number_of_edges() == edge_size:
                    is_done = True

            # 随机将一个节点设为depot
            depot = random.choice(list(G.nodes))

            if nx.is_connected(G) and G.number_of_edges() == edge_size:
                return G, depot, total_cost, total_demand
            

def adjust_edge_count(G, target_edge_count, max_dhcost, max_demand):
    """
    调整图中的边数，使其与目标的 edge_size 相等。如果边数不足就添加边，边数超出则删除边。
    
    参数:
    ----------
    G : networkx.Graph
        要调整的图
    target_edge_count : int
        目标边数
    max_dhcost : int
        最大边成本
    max_demand : int
        最大需求值

    返回:
    ----------
    G : networkx.Graph
        调整后的图
    """
    
    current_edge_count = G.number_of_edges()

    # 如果边数不足，添加边
    while current_edge_count < target_edge_count:
        # 随机选择两个尚未相连的节点
        u, v = random.sample(G.nodes(), 2)
        if not G.has_edge(u, v):
            dhcost = random.randint(1, max_dhcost)
            demand = random.randint(1, max_demand)
            G.add_edge(u, v, dhcost=dhcost, demand=demand)
            current_edge_count += 1

    # 如果边数超出，删除边
    while current_edge_count > target_edge_count:
        # 随机选择一条边进行删除
        u, v = random.choice(list(G.edges()))
        G.remove_edge(u, v)
        current_edge_count -= 1

    return G

def generate_clustered_coords(vertex_size, distribution):
    """
    生成簇分布的节点坐标，输出形式为 NumPy 数组。
    """
    n_cluster = distribution['n_cluster'] # 簇的数量
    batch_size = 1  # 原函数仅生成一个实例，因此这里 batch_size 设置为 1
    problem_size = vertex_size - 1  # 减去一个用于表示仓库（depot）

    # 生成 cluster 的中心坐标
    center = np.random.rand(batch_size, n_cluster * 2)  # 随机生成 cluster 中心的坐标
    center = distribution['lower'] + (distribution['upper'] - distribution['lower']) * center  # 标准化到指定范围
    std = distribution['std']

    coords = torch.zeros(problem_size + 1, 2)  # 初始化坐标张量，包含 depot
    mean_x, mean_y = center[0, ::2], center[0, 1::2]  # 获取每个簇的 x 和 y 坐标的均值

    # 为每个簇生成点
    for i in range(n_cluster):
        if i < n_cluster - 1:
            # 为第 i 个簇生成点
            coords[int((problem_size + 1) / n_cluster) * i:int((problem_size + 1) / n_cluster) * (i + 1)] = \
                torch.cat((torch.FloatTensor(int((problem_size + 1) / n_cluster), 1).normal_(mean_x[i], std),
                           torch.FloatTensor(int((problem_size + 1) / n_cluster), 1).normal_(mean_y[i], std)), dim=1)
        else:
            # 为最后一个簇生成剩余的点
            coords[int((problem_size + 1) / n_cluster) * i:] = \
                torch.cat((torch.FloatTensor((problem_size + 1) - int((problem_size + 1) / n_cluster) * i, 1).normal_(
                    mean_x[i], std),
                           torch.FloatTensor((problem_size + 1) - int((problem_size + 1) / n_cluster) * i, 1).normal_(
                    mean_y[i], std)), dim=1)

    # 将坐标限制在 [0, 1] 范围内
    coords = torch.where(coords > 1, torch.ones_like(coords), coords)
    coords = torch.where(coords < 0, torch.zeros_like(coords), coords)

    # 随机选择一个点作为仓库 (depot)
    depot_idx = int(np.random.choice(range(coords.shape[0]), 1))
    node_coords = coords[torch.arange(coords.size(0)) != depot_idx]  # 去掉 depot 的节点
    depot_coords = coords[depot_idx].unsqueeze(0)  # depot 的坐标

    # 拼接 node 和 depot 坐标，保持输出形式不变
    full_coords = torch.cat((depot_coords, node_coords), dim=0)

    # 转换为 NumPy 数组并返回
    return full_coords[:vertex_size]

def generate_mixed_coords(vertex_size, distribution):
    """
    生成混合分布的节点坐标，输出形式为 NumPy 数组。
    """
    batch_size = 1  # 原函数只生成一个实例
    problem_size = vertex_size  # 这里的 problem_size 实际上就是 vertex_size
    n_cluster_mix = distribution['n_cluster_mix']
    
    # 确保 n_cluster_mix 不超过 vertex_size
    if n_cluster_mix > vertex_size:
        raise ValueError("n_cluster_mix cannot be greater than vertex_size")

    # 生成中心点
    center = np.random.rand(batch_size, n_cluster_mix * 2)  # 随机生成 cluster 中心的坐标
    center = distribution['lower'] + (distribution['upper'] - distribution['lower']) * center  # 标准化到指定范围
    std = distribution['std']

    # 初始化节点坐标，随机生成一半的节点
    coords = torch.FloatTensor(problem_size, 2).uniform_(0, 1)
    
    for j in range(batch_size):
        mean_x, mean_y = center[j, ::2], center[j, 1::2]  # 获取每个簇的 x 和 y 坐标的均值

        # 随机选择一半的坐标进行簇分布的变化
        mutate_idx = np.random.choice(range(problem_size), int(problem_size / 2), replace=False)

        # 为每个簇生成点
        for i in range(n_cluster_mix):
            if i < n_cluster_mix - 1:
                coords[mutate_idx[int(problem_size / n_cluster_mix / 2) * i:int(problem_size / n_cluster_mix / 2) * (i + 1)]] = \
                    torch.cat((torch.FloatTensor(int(problem_size / n_cluster_mix / 2), 1).normal_(mean_x[i], std),
                               torch.FloatTensor(int(problem_size / n_cluster_mix / 2), 1).normal_(mean_y[i], std)), dim=1)
            elif i == n_cluster_mix - 1:
                coords[mutate_idx[int(problem_size / n_cluster_mix / 2) * i:]] = \
                    torch.cat((torch.FloatTensor(int(problem_size / 2) - int(problem_size / n_cluster_mix / 2) * i, 1).normal_(mean_x[i], std),
                               torch.FloatTensor(int(problem_size / 2) - int(problem_size / n_cluster_mix / 2) * i, 1).normal_(mean_y[i], std)), dim=1)

    # 将坐标限制在 [0, 1] 范围内
    coords = torch.where(coords > 1, torch.ones_like(coords), coords)
    coords = torch.where(coords < 0, torch.zeros_like(coords), coords)

    # 转换为 NumPy 数组并返回
    return coords[:vertex_size]


def augment_data_by_8_fold(depot_features, customer_features, customer_demand, graph_info, D, A):
    """
    数据增强，扩展8倍。

    参数:
    ----------
    depot_features : torch.Tensor
        仓库的节点特征 (batch, num_features)
    customer_features : torch.Tensor
        客户的节点特征 (batch, edge_size, num_features)
    graph_dynamic : torch.Tensor
        图的动态信息 (batch, edge_size)
    graph_info : torch.Tensor
        图的信息，包含原始图中的节点和边信息 (batch, 4, edge_size + 1)
    D : torch.Tensor
        每张图的节点之间的最短路径邻接矩阵 (batch, vertex_size, vertex_size)
    A : torch.Tensor
        图的邻接矩阵 (batch, edge_size + 1, edge_size + 1)

    返回:
    ----------
    aug_depot_features : torch.Tensor
        增强后的仓库节点特征 (8 * batch, num_features)
    aug_customer_features : torch.Tensor
        增强后的客户节点特征 (8 * batch, edge_size, num_features)
    aug_graph_dynamic : torch.Tensor
        增强后的图的动态信息 (8 * batch, edge_size)
    aug_graph_info : torch.Tensor
        增强后的图的信息 (8 * batch, 4, edge_size + 1)
    aug_D : torch.Tensor
        增强后的最短路径邻接矩阵 (8 * batch, vertex_size, vertex_size)
    aug_A : torch.Tensor
        增强后的邻接矩阵 (8 * batch, edge_size + 1, edge_size + 1)
    """
    
     # 初始化列表，保存增强后的特征
    depot_features_list = [depot_features.clone() for _ in range(8)]
    customer_features_list = [customer_features.clone() for _ in range(8)]
    
    for i in range(8):
        dep_feat = depot_features_list[i]
        cust_feat = customer_features_list[i]
        
        depot_x = dep_feat[:, [2]]
        depot_y = dep_feat[:, [3]]
        customer_x = cust_feat[:, :, [2]]
        customer_y = cust_feat[:, :, [3]]
        
        if i == 0:
            pass  # 原始坐标，不变
        elif i == 1:
            dep_feat[:, 2] = 1 - depot_x.squeeze(1)
            cust_feat[:, :, 2] = 1 - customer_x.squeeze(2)
        elif i == 2:
            dep_feat[:, 3] = 1 - depot_y.squeeze(1)
            cust_feat[:, :, 3] = 1 - customer_y.squeeze(2)
        elif i == 3:
            dep_feat[:, 2] = 1 - depot_x.squeeze(1)
            dep_feat[:, 3] = 1 - depot_y.squeeze(1)
            cust_feat[:, :, 2] = 1 - customer_x.squeeze(2)
            cust_feat[:, :, 3] = 1 - customer_y.squeeze(2)
        elif i == 4:
            dep_feat[:, 2], dep_feat[:, 3] = depot_y.squeeze(1), depot_x.squeeze(1)
            cust_feat[:, :, 2], cust_feat[:, :, 3] = customer_y.squeeze(2), customer_x.squeeze(2)
        elif i == 5:
            dep_feat[:, 2], dep_feat[:, 3] = 1 - depot_y.squeeze(1), depot_x.squeeze(1)
            cust_feat[:, :, 2], cust_feat[:, :, 3] = 1 - customer_y.squeeze(2), customer_x.squeeze(2)
        elif i == 6:
            dep_feat[:, 2], dep_feat[:, 3] = depot_y.squeeze(1), 1 - depot_x.squeeze(1)
            cust_feat[:, :, 2], cust_feat[:, :, 3] = customer_y.squeeze(2), 1 - customer_x.squeeze(2)
        elif i == 7:
            dep_feat[:, 2], dep_feat[:, 3] = 1 - depot_y.squeeze(1), 1 - depot_x.squeeze(1)
            cust_feat[:, :, 2], cust_feat[:, :, 3] = 1 - customer_y.squeeze(2), 1 - customer_x.squeeze(2)
    
    # 合并增强后的特征
    aug_depot_features = torch.cat(depot_features_list, dim=0)
    aug_customer_features = torch.cat(customer_features_list, dim=0)
    
    # 重复其他数据
    aug_customer_demand = customer_demand.repeat(8, 1)
    aug_graph_info = graph_info.repeat(8, 1, 1)
    aug_D = D.repeat(8, 1, 1)
    aug_A = A.repeat(8, 1, 1)
    
    return aug_depot_features, aug_customer_features, aug_customer_demand, aug_graph_info, aug_D, aug_A