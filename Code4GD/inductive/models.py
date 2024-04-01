import torch
import torch.nn as nn
from torch_scatter import scatter

class GNNLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, attn_dim, n_rel, act=lambda x:x):
        super(GNNLayer, self).__init__()
        self.n_rel = n_rel # 关系数
        self.in_dim = in_dim # 输入维度
        self.out_dim = out_dim # 输出维度
        self.attn_dim = attn_dim # 注意力维度
        self.act = act # 激活函数

        # 关系嵌入层
        self.rela_embed = nn.Embedding(2*n_rel+1, in_dim)

        self.Ws_attn = nn.Linear(in_dim, attn_dim, bias=False) # 节点的注意力
        self.Wr_attn = nn.Linear(in_dim, attn_dim, bias=False) # 关系的注意力
        self.Wqr_attn = nn.Linear(in_dim, attn_dim) # 查询的注意力
        self.w_alpha = nn.Linear(attn_dim, 1) # 计算最终的注意力权重

        self.W_h = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, q_sub, q_rel, hidden, edges, n_node, old_nodes_new_idx):
        # edges:  [batch_idx, head, rela, tail, old_idx, new_idx]
        sub = edges[:,4] # 起始节点
        rel = edges[:,2] # 关系
        obj = edges[:,5] # 目标节点

        hs = hidden[sub] # 
        hr = self.rela_embed(rel) # 获取关系的嵌入向量

        r_idx = edges[:,0] # 
        h_qr = self.rela_embed(q_rel)[r_idx]

        # 消息传递
        message = hs + hr 
        # 注意力机制
        alpha = torch.sigmoid(self.w_alpha(nn.ReLU()(self.Ws_attn(hs) + self.Wr_attn(hr) + self.Wqr_attn(h_qr))))
        message = alpha * message
        # 消息聚合
        message_agg = scatter(message, index=obj, dim=0, dim_size=n_node, reduce='sum')
        hidden_new = self.act(self.W_h(message_agg)) # 公式(3)

        return hidden_new

class RED_GNN_induc(torch.nn.Module):
    def __init__(self, params, loader):
        super(RED_GNN_induc, self).__init__()
        self.n_layer = params.n_layer # GNN 层数
        self.hidden_dim = params.hidden_dim # 隐藏层维度
        self.attn_dim = params.attn_dim # 注意力维度
        self.n_rel = params.n_rel # 关系数量
        self.loader = loader # 数据加载
        acts = {'relu': nn.ReLU(), 'tanh': torch.tanh, 'idd': lambda x:x} # 激活函数
        act = acts[params.act] # 默认恒等映射

        # 创建GNN层
        self.gnn_layers = []
        for i in range(self.n_layer):
            self.gnn_layers.append(GNNLayer(self.hidden_dim, self.hidden_dim, self.attn_dim, self.n_rel, act=act))
        self.gnn_layers = nn.ModuleList(self.gnn_layers)
       
        # 创建dropout层和输出层
        self.dropout = nn.Dropout(params.dropout)
        self.W_final = nn.Linear(self.hidden_dim, 1, bias=False)         # get score
        self.gate= nn.GRU(self.hidden_dim, self.hidden_dim) # 更新隐藏状态

    def forward(self, subs, rels, mode='transductive'):
        n = len(subs)

        n_ent = self.loader.n_ent if mode=='transductive' else self.loader.n_ent_ind

        q_sub = torch.LongTensor(subs).cuda() # 查询的起始节点
        q_rel = torch.LongTensor(rels).cuda() # 查询的关系

        h0 = torch.zeros((1, n,self.hidden_dim)).cuda() # 初始化隐藏状态
        nodes = torch.cat([torch.arange(n).unsqueeze(1).cuda(), q_sub.unsqueeze(1)], 1) # 节点信息
        hidden = torch.zeros(n, self.hidden_dim).cuda() # 初始化节点特征

        for i in range(self.n_layer):
            # 获取邻居节点信息
            nodes, edges, old_nodes_new_idx = self.loader.get_neighbors(nodes.data.cpu().numpy(), mode=mode)

            # 消息传递+聚合
            hidden = self.gnn_layers[i](q_sub, q_rel, hidden, edges, nodes.size(0), old_nodes_new_idx)
            h0 = torch.zeros(1, nodes.size(0), hidden.size(1)).cuda().index_copy_(1, old_nodes_new_idx, h0) # 更新状态
            hidden = self.dropout(hidden)
            hidden, h0 = self.gate(hidden.unsqueeze(0), h0) # 更新状态
            hidden = hidden.squeeze(0)
        # 分数
        scores = self.W_final(hidden).squeeze(-1) 
        scores_all = torch.zeros((n, n_ent)).cuda()
        scores_all[[nodes[:,0], nodes[:,1]]] = scores
        return scores_all


