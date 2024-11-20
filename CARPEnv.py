from dataclasses import dataclass
import torch
import networkx as nx
from CARProblemDef import get_random_carp_problems, augment_data_by_8_fold

@dataclass
class Reset_State:
    depot_features: torch.Tensor = None  # 仓库特征
    customer_features: torch.Tensor = None  # 客户特征
    customer_demand: torch.Tensor = None  # 客户需求
    graph_info: torch.Tensor = None
    A: torch.Tensor = None  # 邻接矩阵
    D: torch.Tensor = None 

@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor = None
    POMO_IDX: torch.Tensor = None
    selected_count: int = None
    load: torch.Tensor = None
    current_edge: torch.Tensor = None
    current_customer: torch.Tensor = None
    ninf_mask: torch.Tensor = None
    finished: torch.Tensor = None

class CARPEnv:
    def __init__(self, **env_params):
        self.env_params = env_params
        self.vertex_size = env_params['vertex_size']
        self.edge_size = env_params['edge_size']
        self.pomo_size = env_params['pomo_size']
        self.distribution = env_params['distribution']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 保存加载的问题
        self.FLAG__use_saved_problems = False
        self.saved_depot_features = None
        self.saved_customer_features = None
        self.saved_customer_demand = None  # 保存客户需求
        self.saved_graph_info = None
        self.saved_D = None
        self.saved_A = None
        self.saved_index = None

        # 常量 @ Load_Problem
        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        self.depot_features = None
        self.customer_features = None
        self.customer_demand = None  # 客户需求
        self.A = None  # 邻接矩阵
        self.D = None

        # 动态状态
        self.selected_count = None
        self.current_customer = None
        self.selected_customer_list = None
        self.load = None
        self.visited_ninf_flag = None
        self.ninf_mask = None
        self.finished = None

        # 状态返回
        self.reset_state = Reset_State()
        self.step_state = Step_State()

    def use_saved_problems(self, filename, device):
        self.FLAG__use_saved_problems = True
        loaded_dict = torch.load(filename, map_location=device)

        self.saved_depot_features = loaded_dict['depot_features']
        self.saved_customer_features = loaded_dict['customer_features']
        self.saved_customer_demand = loaded_dict['customer_demand']  # 加载客户需求
        self.saved_graph_info = loaded_dict['graph_info']
        self.saved_D = loaded_dict['D']
        self.saved_A = loaded_dict['A']
        self.saved_index = 0

    def load_problems(self, batch_size, aug_factor=1, distribution=None, load_path=None, copy=None, episode=None):
        self.batch_size = batch_size
        if distribution is not None:
            self.distribution['data_type'] = distribution

        # 如果没有传入复制的数据
        if copy is None:
            if not self.FLAG__use_saved_problems:
                # 从头生成数据
                depot_features, customer_features, customer_demand, graph_info, D, A = get_random_carp_problems(
                    batch_size, self.vertex_size, self.edge_size, self.device, self.distribution, load_path=load_path, episode=episode
                )
            else:
                # 从保存的数据中加载
                depot_features = self.saved_depot_features[self.saved_index:self.saved_index + batch_size]
                customer_features = self.saved_customer_features[self.saved_index:self.saved_index + batch_size]
                customer_demand = self.saved_customer_demand[self.saved_index:self.saved_index + batch_size]  # 加载客户需求
                graph_info = self.saved_graph_info[self.saved_index:self.saved_index + batch_size]
                D = self.saved_D[self.saved_index:self.saved_index + batch_size]
                A = self.saved_A[self.saved_index:self.saved_index + batch_size]
                self.saved_index += batch_size
        else:
            # 复制传入的数据
            depot_features, customer_features, customer_demand, graph_info, D, A = copy

        if aug_factor > 1:
            if aug_factor == 8:
                # 数据增强处理
                self.batch_size = self.batch_size * 8
                depot_features, customer_features, customer_demand, graph_info, D, A = augment_data_by_8_fold(
                    depot_features, customer_features, customer_demand, graph_info, D, A
                )
            else:
                raise NotImplementedError

        # 将数据移动到 GPU 上
        self.depot_features = depot_features.to(self.device)
        self.customer_features = customer_features.to(self.device)
        self.customer_demand = customer_demand.to(self.device)  # 客户需求
        self.graph_info = graph_info.to(self.device)
        self.A = A.to(self.device)
        self.D = D.to(self.device)

        depot_demand = torch.zeros(size=(self.batch_size, 1)).to(self.device)
        # 确保 depot_demand 和 customer_demand 都在同一个设备上
        depot_demand = depot_demand.to(self.device)
        customer_demand = customer_demand.to(self.device)

        self.depot_customer_demand = torch.cat((depot_demand, customer_demand), dim=1)

        # 初始化 BATCH_IDX 和 POMO_IDX
        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size).to(self.device)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size).to(self.device)

        # 更新重置状态
        self.reset_state.depot_features = depot_features
        self.reset_state.customer_features = customer_features
        self.reset_state.customer_demand = customer_demand  # 将客户需求加入到 reset_state 中
        self.reset_state.graph_info = graph_info
        self.reset_state.A = A  # 使用新的邻接矩阵 A
        self.reset_state.D = D

        # 更新 step 状态
        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX = self.POMO_IDX

    def reset(self):
        self.selected_count = 0
        self.current_edge = None
        self.current_customer = None
        self.selected_edge_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long)
        self.selected_customer_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long)

        self.load = torch.ones(size=(self.batch_size, self.pomo_size))
        self.visited_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.edge_size + 1)).to(self.device)
        self.ninf_mask = torch.zeros(size=(self.batch_size, self.pomo_size, self.edge_size + 1)).to(self.device)
        self.finished = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.bool).to(self.device)

        reward = None
        done = False
        return self.reset_state, reward, done
    
    # 在每次执行 step 方法之前调用，用来初始化或者记录一些环境的状态
    def pre_step(self):
        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_edge = self.current_edge
        self.step_state.current_customer = self.current_customer
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished

        reward = None
        done = False
        return self.step_state, reward, done
    
    # 在每一步更新POMO的状态，包括选择的节点或边、负载状态、访问标记等
    def step(self, selected):
        # selected.shape: (batch, pomo)
        # Dynamic-1
        ####################################
        self.selected_count += 1
        self.current_edge = selected
        # shape: (batch, pomo)
        self.selected_edge_list = torch.cat((self.selected_edge_list, self.current_edge[:, :, None]), dim=2)
        # shape: (batch, pomo, 0~)

        # Dynamic-2
        ####################################
        self.at_the_depot = (selected == 0)

        demand_list = self.depot_customer_demand[:, None, :].expand(self.batch_size, self.pomo_size, -1)
        # shape: (batch, pomo, edge_size+1)
        gathering_index = selected[:, :, None]
        # shape: (batch, pomo, 1)
        selected_demand = demand_list.gather(dim=2, index=gathering_index).squeeze(dim=2)
        # shape: (batch, pomo)
        self.load -= selected_demand
        self.load[self.at_the_depot] = 1 # refill loaded at the depot

        self.visited_ninf_flag[self.BATCH_IDX, self.POMO_IDX, selected] = float('-inf')
        # shape: (batch, pomo, problem+1)
        self.visited_ninf_flag[:, :, 0][~self.at_the_depot] = 0  # depot is considered unvisited, unless you are AT the depot

        self.ninf_mask = self.visited_ninf_flag.clone()
        round_error_epsilon = 0.00001
        demand_too_large = self.load[:, :, None] + round_error_epsilon < demand_list
        # shape: (batch, pomo, edge_size+1)
        self.ninf_mask[demand_too_large] = float('-inf')
        # shape: (batch, pomo, edge_size+1)

        newly_finished = (self.visited_ninf_flag == float('-inf')).all(dim=2)
        # shape: (batch, pomo)
        self.finished = self.finished + newly_finished
        # shape: (batch, pomo)

        # do not mask depot for finished episode.
        self.ninf_mask[:, :, 0][self.finished] = 0

        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_edge = self.current_edge
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished

        # returning values
        done = self.finished.all()
        if done:
            reward = -self._get_travel_distance()  # note the minus sign!
        else:
            reward = None
        return self.step_state, reward, done

    def _get_travel_distance(self):

        total_dhcost = torch.zeros(self.batch_size*self.pomo_size)

        pi = self.selected_edge_list
        pi_num_samples, pomo_size, tour_length = pi.size()
        pi = pi.view(pi_num_samples * pomo_size, tour_length)

        idx = pi.unsqueeze(1).expand(-1, self.graph_info.size(1), -1)
        graph_info = self.graph_info.unsqueeze(1).repeat(1, pomo_size, 1, 1)
        graph_info_num_samples, feature_size, edge_size = self.graph_info.size()
        graph_info = graph_info.view(graph_info_num_samples * pomo_size, feature_size, edge_size)
        tour = torch.gather(graph_info, 2, idx).to(int)

        D = self.D.unsqueeze(1).expand(-1, pomo_size, -1, -1)
        D_num_samples, _, node_size = self.D.size()
        D = D.reshape(D_num_samples * pomo_size, node_size, node_size).to(self.device)
        
        num_samples, _, tour_length = tour.size()
        f_1 = torch.zeros(num_samples)
        f_2 = torch.zeros(num_samples)
        depot = graph_info[:, 1, 0].long()
        indices = torch.arange(num_samples)

        for i in range(1, tour_length + 1):
            if i == 1:
                node_1_front = tour[:, 1, -i - 1]
                node_2_front = tour[:, 2, -i - 1]
                node_1_behind = tour[:, 1, -i]
                node_2_behind = tour[:, 2, -i]
                f_1 = tour[indices, -2, -i] + torch.min(
                    D[indices, node_2_front, node_1_behind] + D[indices, node_2_behind, depot],
                    D[indices, node_2_front, node_2_behind] + D[indices, node_1_behind, depot])

                f_2 = tour[indices, -2, -i] + torch.min(
                    D[indices, node_1_front, node_1_behind] + D[indices, node_2_behind, depot],
                    D[indices, node_1_front, node_2_behind] + D[indices, node_1_behind, depot])

            elif i == tour_length:
                node_1 = tour[:, 1, -i]
                node_2 = tour[:, 2, -i]
                total_dhcost = tour[indices, -2, -i] + torch.min(D[indices, depot, node_1] + f_1,
                                                                      D[indices, depot, node_2] + f_2)
                total_dhcost = total_dhcost.view(self.batch_size, self.pomo_size)

            else:
                node_1_front = tour[indices, 1, -i - 1]
                node_2_front = tour[indices, 2, -i - 1]
                node_1_behind = tour[indices, 1, -i]
                node_2_behind = tour[indices, 2, -i]
                f_1_ = tour[indices, -2, -i] + torch.min(D[indices, node_2_front, node_1_behind] + f_1,
                                                         D[indices, node_2_front, node_2_behind] + f_2)
                f_2_ = tour[indices, -2, -i] + torch.min(D[indices, node_1_front, node_1_behind] + f_1,
                                                         D[indices, node_1_front, node_2_behind] + f_2)
                f_1 = f_1_
                f_2 = f_2_
        del graph_info, D
        torch.cuda.empty_cache()        
        return total_dhcost