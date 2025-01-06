import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()

        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        # Weight parameter for the attention mechanism
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        # Attention mechanism parameter
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        self.batch_size, _, _ = h.shape  # h.shape: (batch_size, num_nodes, features)

        W = self.W.unsqueeze(0).expand(self.batch_size, -1, -1)
        Wh = torch.matmul(h, W)  # h.shape: (batch_size, num_nodes, in_features)
                                 # Wh.shape: (batch_size, num_nodes, out_features)

        e = self._prepare_attentional_mechanism_input(Wh)   # 每条边权重分数

        # Apply attention mask
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)

        # Normalize attention weights
        attention = F.softmax(attention, dim=1)

        # Apply dropout on attention weights
        attention = F.dropout(attention, self.dropout, training=self.training)

        # Compute the final node representation
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape: (batch_size, num_nodes, out_features)
        # self.a.shape: (2 * out_features, 1)
        # Wh1 & Wh2.shape: (batch_size, num_nodes, 1)

        a = self.a.unsqueeze(0).expand(self.batch_size, -1, -1)

        Wh1 = torch.matmul(Wh, a[:, :self.out_features, :])
        Wh2 = torch.matmul(Wh, a[:, self.out_features:, :])

        # Broadcasting addition to get attention scores
        e = Wh1 + Wh2.transpose(1, 2)

        return self.leakyrelu(e)

    def __repr__(self):
        return f'{self.__class__.__name__} ({self.in_features} -> {self.out_features})'


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GAT, self).__init__()

        self.dropout = dropout

        # Create multiple attention layers, one for each head
        self.attentions = [
            GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True)
            for _ in range(nheads)
        ]
        for i, attention in enumerate(self.attentions):
            self.add_module(f'attention_{i}', attention)

        # Output attention layer, taking the concatenated results of all heads
        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        # Apply dropout to input
        x = F.dropout(x, self.dropout, training=self.training)

        # Apply each attention layer and concatenate the outputs
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)

        # Apply dropout to the concatenated features
        x = F.dropout(x, self.dropout, training=self.training)

        # Apply the final output attention layer
        x = F.elu(self.out_att(x, adj))
        return x
