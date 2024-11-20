import torch
import torch.nn as nn
import torch.nn.functional as F
from GCN import GCN

class CARPModel(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        # 初始化编码器，使用GCN编码图结构
        self.encoder = CARP_Encoder(**model_params)
        # 初始化解码器，使用多头注意力机制进行解码
        self.decoder = CARP_Decoder(**model_params)
        # 存储编码器生成的节点或边的表示
        self.encoded_nodes = None

    def pre_forward(self, reset_state, attn_type=None):
        # 仓库特征 (batch, 1, num_features)
        depot = reset_state.depot_features
        # 客户特征 (batch, problem, num_features)
        customer = reset_state.customer_features
        # 客户的需求 (batch, problem)
        customer_demand = reset_state.customer_demand  # 使用传入的客户需求
        # 图的邻接矩阵 A (batch, problem+1, problem+1)
        A = reset_state.A
        
        
        # 使用编码器将仓库、客户、客户需求和图的邻接矩阵编码为高维表示，存储在 self.encoded_nodes 中
        # 编码后的节点表示 (batch, problem+1, embedding_dim)
        
        self.encoded_nodes = self.encoder(depot, customer, customer_demand, A)

        # 将编码后的节点信息传递给解码器
        self.decoder.set_kv(self.encoded_nodes)

    def forward(self, state, route=None, return_probs=False, teacher=False):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)

        # 第一次选择，从仓库出发
        if state.selected_count == 0:
            selected = torch.zeros(size=(batch_size, pomo_size), dtype=torch.long)  # 所有车辆从仓库出发
            prob = torch.ones(size=(batch_size, pomo_size))  # 初始选择概率为均匀分布
            if return_probs:
                probs = torch.ones(size=(batch_size, pomo_size, self.encoded_nodes.size(1)))
        elif state.selected_count == 1:  # 第二次选择
            selected = torch.arange(start=1, end=pomo_size+1)[None, :].expand(batch_size, pomo_size)
            prob = torch.ones(size=(batch_size, pomo_size))  # 概率均匀分布
            if return_probs:
                probs = torch.ones(size=(batch_size, pomo_size, self.encoded_nodes.size(1)))
        else:
            # 获取上一次选择的节点的编码表示
            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_edge)
            # 使用解码器计算下一步选择的概率分布
            probs = self.decoder(encoded_last_node, state.load, ninf_mask=state.ninf_mask)

            if route is None:
                if self.training or self.model_params['eval_type'] == 'softmax':
                    while True:  # to fix pytorch.multinomial bug on selecting 0 probability elements
                        with torch.no_grad():
                            selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1) \
                                .squeeze(dim=1).reshape(batch_size, pomo_size)
                            # shape: (batch, pomo)
                        prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size)
                            # shape: (batch, pomo)
                        if (prob != 0).all():
                            break    

                else:
                    if teacher:
                        selected = probs.argmax(dim=2)
                        # shape: (batch, pomo)
                        prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size)
                    else:
                        selected = probs.argmax(dim=2)
                        # shape: (batch, pomo)
                        prob = None  # value not needed. Can be anything.    
                        
            # 学生利用教师模型的路径
            else:
                selected = route[:, :,state.selected_count].reshape(batch_size, pomo_size).long()
                prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size)
                
        if return_probs:
            return selected, prob, probs
        
        return selected, prob


# 从编码的节点表示中提取当前选择节点的特征
def _get_encoding(encoded_nodes, node_index_to_pick):
    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)

    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)

    return picked_nodes


########################################
# ENCODER
########################################

class CARP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        # 使用GCN进行图结构数据的编码
        self.gcn_model = GCN(6, embedding_dim)
        # 多层编码器
        encoder_layer_num = model_params['encoder_layer_num']
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])

    def forward(self, depot, customer, customer_demand, A):
        
        # 检查和调整 depot 的维度
        if depot.dim() == 4:
            depot = depot.squeeze(1)  # 将 (batch_size, 1, 1, num_features) 转换为 (batch_size, 1, num_features)

        # 确保 depot 的形状为 (batch_size, 1, num_features)
        if depot.dim() == 2:
            depot = depot.unsqueeze(1)  # 如果是 2D 张量 (batch_size, num_features)，则添加维度

        # 检查和调整 customer 的维度
        if customer.dim() == 4:
            customer = customer.squeeze(1)  # 将 (batch_size, 1, edge_size, num_features) 转换为 (batch_size, edge_size, num_features)

        # 将仓库节点与客户节点拼接
        node_feature = torch.cat((depot, customer), dim=1)  # 拼接后的形状为 (batch_size, problem+1, num_features)
        # 通过GCN对节点特征进行编码
        out = self.gcn_model(A, node_feature)

        # 多层编码器对GCN生成的节点嵌入进行进一步处理
        for layer in self.layers:
            out = layer(out)

        return out  # 返回编码后的节点表示


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = model_params['embedding_dim']
        head_num = model_params['head_num']
        qkv_dim = model_params['qkv_dim']

        # 初始化多头注意力机制的线性层
        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        # 多头注意力的组合层
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.add_n_normalization_1 = AddAndInstanceNormalization(**model_params)
        self.feed_forward = FeedForward(**model_params)
        self.add_n_normalization_2 = AddAndInstanceNormalization(**model_params)

    def forward(self, input1):
        head_num = self.model_params['head_num']

        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)

        out_concat = multi_head_attention(q, k, v)
        multi_head_out = self.multi_head_combine(out_concat)

        out1 = self.add_n_normalization_1(input1, multi_head_out)
        out2 = self.feed_forward(out1)
        out3 = self.add_n_normalization_2(out1, out2)
        
        return out3  # 返回经过注意力和前馈网络处理后的节点表示


########################################
# DECODER
########################################

class CARP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = model_params['embedding_dim']
        head_num = model_params['head_num']
        qkv_dim = model_params['qkv_dim']

        # 初始化解码器中的线性层
        self.Wq_last = nn.Linear(embedding_dim + 1, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.k = None  # 保存的键
        self.v = None  # 保存的值
        self.single_head_key = None  # 用于计算单头注意力的键

    def set_kv(self, encoded_nodes):
        head_num = self.model_params['head_num']
        self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=head_num)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=head_num)
        self.single_head_key = encoded_nodes.transpose(1, 2)

    def forward(self, encoded_last_node, load, ninf_mask):
        head_num = self.model_params['head_num']

        input_cat = torch.cat((encoded_last_node, load[:, :, None]), dim=2)
        q_last = reshape_by_heads(self.Wq_last(input_cat), head_num=head_num)

        out_concat = multi_head_attention(q_last, self.k, self.v, rank3_ninf_mask=ninf_mask)
        mh_atten_out = self.multi_head_combine(out_concat)

        score = torch.matmul(mh_atten_out, self.single_head_key)

        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping = self.model_params['logit_clipping']

        score_scaled = score / sqrt_embedding_dim
        score_clipped = logit_clipping * torch.tanh(score_scaled)

        score_masked = score_clipped + ninf_mask
        probs = F.softmax(score_masked, dim=2)

        return probs


########################################
# NN SUB CLASS / FUNCTIONS
########################################

def reshape_by_heads(qkv, head_num):
    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    q_transposed = q_reshaped.transpose(1, 2)

    return q_transposed


def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)

    score = torch.matmul(q, k.transpose(2, 3))
    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))

    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, k.size(2))
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, k.size(2))

    weights = nn.Softmax(dim=3)(score_scaled)
    out = torch.matmul(weights, v)

    out_transposed = out.transpose(1, 2)
    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)

    return out_concat


class AddAndInstanceNormalization(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input1, input2):
        added = input1 + input2
        transposed = added.transpose(1, 2)
        normalized = self.norm(transposed)
        back_trans = normalized.transpose(1, 2)

        return back_trans


class FeedForward(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        return self.W2(F.relu(self.W1(input1)))