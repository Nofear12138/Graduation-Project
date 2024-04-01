import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter

class RatingModel(nn.Module):
    '''MLP for rating'''
    def __init__(self,num_classes, input_dim, hidden_dim, output_dim, embed_dim, n_ent):
        super(RatingModel, self).__init__()
        self.num_classes = num_classes
        self.entity_embedding = nn.Embedding(n_ent, embed_dim)
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        user, item = x[:, 0], x[:, 1]
        user_embed = self.entity_embedding(user)
        item_embed = self.entity_embedding(item)
        input = torch.cat([user_embed, item_embed], dim=1)
        input = self.linear1(input)
        input = self.relu(input)
        input = self.dropout1(input)
        input = self.linear2(input)
        input = self.relu2(input)
        input = self.dropout2(input)
        input = self.linear3(input)
        return input
    
class GNNLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, attn_dim, n_rel, act=lambda x:x):
        super(GNNLayer, self).__init__()
        self.n_rel = n_rel
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.attn_dim = attn_dim
        self.act = act

        # self.user_embed = nn.Embedding(n_user,in_dim)
        # self.item_embed = nn.Embedding(n_item, in_dim) # Q: Can't add it
        self.rela_embed = nn.Embedding(2*n_rel+1, in_dim)
        self.Ws_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wr_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wqr_attn = nn.Linear(in_dim, attn_dim)
        # self.Wrating_attn = nn.Linear(in_dim, attn_dim)
        self.w_alpha  = nn.Linear(attn_dim, 1)

        self.W_h = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, q_sub, q_rel, hidden, edges, n_node, old_nodes_new_idx):
        # edges:  [batch_idx, head, rela, tail, old_idx, new_idx]
        # print("edges num:",edges.shape)
        # for edge in edges:
        #     print("GNNLayer forward edge:" ,edge)

        sub = edges[:,4]
        rel = edges[:,2]
        obj = edges[:,5]

        # print("GNNLayer forward sub, rel, obj:",sub, rel, obj)
        # hss = torch.zeros((len(sub), hidden.shape[1])).cuda()
        # mask = (sub >= 1430)
        # hss[mask] = 0.5 * hidden[sub[mask]]
        # hss[~mask] = hidden[sub[~mask]]
        hs = hidden[sub] # hidden layers, consist of all entities
    
        # print("here")
        # print("GNNLayer forward hidden and hs:", hidden.shape, hs.shape)

        # add mask to reduce the weight! 
        hr = self.rela_embed(rel) # 相似度？
        r_idx = edges[:,0]
        h_qr = self.rela_embed(q_rel)[r_idx]
        # hrating = self.rating_embed(q_rel)[r_idx]

        message = hs + hr # 上一层的起始点信息+这一层的关系信息

        # print("GNNLayer forward message:", message, message.shape)

        alpha = torch.sigmoid(self.w_alpha(nn.ReLU()(self.Ws_attn(hs) + self.Wr_attn(hr) + self.Wqr_attn(h_qr))))
        
        # print("GNNLayer forward alpha:", alpha, alpha.shape)
        
        message = alpha * message

        # print("GNNLayer forward message2:", message, message.shape)

        message_agg = scatter(message, index=obj, dim=0, dim_size=n_node, reduce='sum') # size changed

        # print("GNNLayer forward message_agg", message_agg, message_agg.shape) 

        hidden_new = self.act(self.W_h(message_agg))

        # print("GNNLayer forward hidden_new:", hidden_new, hidden_new.shape)

        return hidden_new

class RED_GNN(torch.nn.Module):
    def __init__(self, params, loader):
        super(RED_GNN, self).__init__()
        self.n_layer = params.n_layer
        self.hidden_dim = params.hidden_dim
        self.attn_dim = params.attn_dim
        self.n_rel = params.n_rel
        self.loader = loader
        acts = {'relu': nn.ReLU(), 'tanh': torch.tanh, 'idd': lambda x:x}
        act = acts[params.act]

        self.gnn_layers = []
        for i in range(self.n_layer):
            self.gnn_layers.append(GNNLayer(self.hidden_dim, self.hidden_dim, self.attn_dim, self.n_rel, act=act))
        self.gnn_layers = nn.ModuleList(self.gnn_layers)

        self.rating_model = RatingModel(num_classes=5, input_dim=128, hidden_dim=256, output_dim=5, embed_dim=64, n_ent=loader.n_ent)
        
        self.dropout = nn.Dropout(params.dropout)
        self.W_final = nn.Linear(self.hidden_dim, 1, bias=False)         # get score
        self.gate = nn.GRU(self.hidden_dim, self.hidden_dim)

    def forward(self, triples, neg_objs=None, mode='train'):
        if mode == 'train':
            subs, rels, objs = triples[:, 0], triples[:, 1], triples[:, 2]
            users = torch.tensor(triples[:,0], dtype=torch.long).clone().detach().cuda()
            items = torch.tensor(triples[:,2], dtype=torch.long).clone().detach().cuda()
            pred_rating = self.rating_model(torch.stack((users, items), dim=1))

            # print("REDGNN forward subs, rels, objs, users, items:", subs, rels, objs, users, items)

        else:
            subs, rels = triples[:, 0], triples[:, 1] # 实际传入为二元组
            pred_rating = None

        n = len(subs) # number of users
        q_sub = torch.LongTensor(subs).cuda() # 给定用户、关系，去获取物品得分
        q_rel = torch.LongTensor(rels).cuda()

        # print("REDGNN forward q_sub, q_rel", q_sub, q_rel)

        h0 = torch.zeros((1, n,self.hidden_dim)).cuda() # 初始状态
        nodes = torch.cat([torch.arange(n).unsqueeze(1).cuda(), q_sub.unsqueeze(1)], 1) # [batch_size, index, sub_id]

        hidden = torch.zeros(n, self.hidden_dim).cuda()

        scores_all = []
        for i in range(self.n_layer):
            # nodes, edges, old_nodes_new_idx = self.loader.get_neighbors(nodes.data.cpu().numpy(), mode=mode)
            nodes, edges, old_nodes_new_idx = self.loader.get_neighbors_withoutmatrix(nodes.data.cpu().numpy(), mode=mode)
            hidden = self.gnn_layers[i](q_sub, q_rel, hidden, edges, nodes.size(0), old_nodes_new_idx) # 聚合

            # print("REDGNN gnn hidden:", i, hidden, hidden.shape)

            h0 = torch.zeros(1, nodes.size(0), hidden.size(1)).cuda().index_copy_(1, old_nodes_new_idx, h0)

            # print("RED_GNN h0",i, h0, h0.shape)

            hidden = self.dropout(hidden)

            hidden, h0 = self.gate (hidden.unsqueeze(0), h0)
            # print("REDGNN hidden and h0", hidden, hidden.shape, h0, h0.shape)

            hidden = hidden.squeeze(0) # 迭代
        scores = self.W_final(hidden).squeeze(-1)
        # print("REDGNN forward scores:", scores, scores.shape)

        scores_all = torch.zeros((n, self.loader.n_ent)).cuda()         # non_visited entities have 0 scores
        
        # print("nodes, nodes[:,0], nodes[:,1]", nodes, nodes.shape, nodes[:,0], nodes[:,1])
        scores_all[[nodes[:,0], nodes[:,1]]] = scores
        # 建立新score，包括正例负例的分数，第一列为正例,即实现[b,n_ent] -> [b, 1+neg_size]
        if mode == 'train':
            neg_size  = len(neg_objs[0])
            # print("negsize =", neg_size)
            result_scores = torch.zeros((n, 1 + neg_size))
            row = 0
            for tr in triples:
                result_scores[row, 0] = scores_all[row, tr[-1]]
                for j in range(1,neg_size+1):
                    result_scores[row, j] = scores_all[row, neg_objs[row][j - 1]]
                row += 1
            # print("result_scores:", result_scores)
            return result_scores, pred_rating
        else:
            return scores_all





