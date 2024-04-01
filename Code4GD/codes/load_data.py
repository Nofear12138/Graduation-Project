import os
import torch
import random
from scipy.sparse import csr_matrix
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import sys

class Dataset_Rating(Dataset):
    '''zzl added, 1.29 使用相同输入数据, 同对应dataloader作废'''
    def __init__(self, dataset, task_dir):
        self.dataset = dataset
        self.task_dir = task_dir
        
        # get entity index
        with open(os.path.join(task_dir, 'entities.txt')) as f:
            self.et2id = {}
            for line in f.readlines():
                entity, index = line.strip().split()
                self.et2id[entity] = int(index)

        self.n_ent = len(self.et2id)
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        return self.et2id[self.dataset[index][0]], self.et2id[self.dataset[index][1]], int(self.dataset[index][2])-1
    
class DataLoader_Rating:
    def __init__(self, task_dir, batch_size):
        self.task_dir = task_dir
        self.batch_size = batch_size
        self.train_ratio = 0.8
        self.valid_ratio = 0.1
        self.test_ratio = 0.1
        self.dataset = self.load_dataset()
        self.split_dataset(self.dataset)
        with open(os.path.join(task_dir, 'entities.txt')) as f:
            self.et2id = {}
            for line in f.readlines():
                entity, index = line.strip().split()
                self.et2id[entity] = int(index)
        self.n_ent = len(self.et2id)

    def load_dataset(self):
        dataset = []
        with open(os.path.join(self.task_dir, 'RatingDataset.txt')) as f:
            for line in f.readlines():
                userData = list(line.strip().split('\t'))
                dataset.append(userData)
                user, item, rating = userData[0], userData[1], userData[2]
        return dataset
    
    def split_dataset(self, dataset):
        train_data, temp_data = train_test_split(dataset, test_size=0.2, random_state=42)
        valid_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

        self.train_dataset = Dataset_Rating(train_data, self.task_dir)
        self.valid_dataset = Dataset_Rating(valid_data, self.task_dir)
        self.test_dataset = Dataset_Rating(test_data, self.task_dir)

        self.train_loader = DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.valid_loader = DataLoader(dataset=self.valid_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=True)
    

class DataLoader_Rec:
    def __init__(self, args):
        self.task_dir = args.data_path # where the data path
        with open(os.path.join(self.task_dir, 'new_entities.txt')) as f:
            self.entity2id = dict()
            for line in f:
                entity, eid = line.strip().split()
                self.entity2id[entity] = int(eid) # 实体映射id, 先user后item

        with open(os.path.join(self.task_dir, 'relations.txt')) as f:
            self.relation2id = dict()
            for line in f:
                relation, rid = line.strip().split()
                self.relation2id[relation] = int(rid) # 关系映射id

        self.n_ent = len(self.entity2id) # 实体数量
        self.n_rel = len(self.relation2id) #关系数量
        self.n_user, self.n_item = self.get_nums('facts.txt') # 用户数量和物品数量
        self.filters = defaultdict(lambda:set()) # 记录(h,r,t)在(h,r)确定下的所有t
        print('n_ent:', self.n_ent, 'n_user:', self.n_user, 'n_item:', self.n_item)
        self.fact_triple  = self.read_triples('facts.txt') # 图谱三元组
        self.train_triple = self.read_triples('train.txt') # 训练集三元组
        self.valid_triple = self.read_triples('valid.txt') # 验证集三元组
        self.test_triple  = self.read_triples('test.txt')  # 测试集三元组
    
        # add inverse 1.29 rec format: user - rating -item
        self.fact_data  = self.double_triple(self.fact_triple) # double, 用于图信息传递

        # self.train_data = np.array(self.double_triple(self.train_triple))
        # self.valid_data = self.double_triple(self.valid_triple)
        # self.test_data  = self.double_triple(self.test_triple)
        self.train_data = self.train_triple # 不double，保证是user到item
        self.valid_data = self.valid_triple
        self.test_data  = self.test_triple

        self.load_graph(self.fact_data) # KG
        # self.load_test_graph(self.double_triple(self.fact_triple)+self.double_triple(self.train_triple)) # tKG (previous)
        self.load_test_graph(self.double_triple(self.fact_triple)) # now: tKG = KG
        self.load_train_user2rating(self.train_data) # 对于每个用户获取其打的分数
        
        self.valid_q, self.valid_a = self.load_query(self.valid_data) # record the query and answer for valid data
        self.test_q,  self.test_a  = self.load_query(self.test_data) # record the query and answer for test data

        self.n_train = len(self.train_data) 
        self.n_valid = len(self.valid_q)
        self.n_test  = len(self.test_q)

        # added by zzl for negative sample
        self.neg_size = args.neg_num # for train
        self.num_neighbor = args.num_neighbor
        self.v_t_neg_size = 49 # for valid and test
        self.train_user_set, self.valid_user_set, self.test_user_set = self.get_user_set() # record the set item of every user
        
        print("load_len", len(self.test_user_set)) 
        
        self.get_neg_set()

        for filt in self.filters:
            self.filters[filt] = list(self.filters[filt])
        print('n_train:', self.n_train, 'n_valid:', self.n_valid, 'n_test:', self.n_test)

    def read_triples(self, filename):
        triples = []
        with open(os.path.join(self.task_dir, filename)) as f:
            for line in f:
                h, r, t = line.strip().split()
                h, r, t = self.entity2id[h], self.relation2id[r], self.entity2id[t]
                triples.append([h,r,t])
                self.filters[(h,r)].add(t)
                self.filters[(t,r+self.n_rel)].add(h) 
        return triples

    def get_nums(self, filename):
        # added for getting the number of users and items
        user = []
        item = []
        with open(os.path.join(self.task_dir, filename)) as f:
            for line in f:
                u, r, i = line.strip().split()
                user.append(u)
                item.append(i)
        return len(set(user)), len(set(item))
    
    def double_triple(self, triples):
        new_triples = []
        for triple in triples:
            h, r, t = triple
            new_triples.append([t, r+self.n_rel, h]) 
        return triples + new_triples

    def load_graph(self, triples):
        # 自关系
        idd = np.concatenate([np.expand_dims(np.arange(self.n_ent),1), 2*self.n_rel*np.ones((self.n_ent, 1)), np.expand_dims(np.arange(self.n_ent),1)], 1)
        
        # print("load_graph: idd:", idd)
        # 生成图谱
        self.KG = np.concatenate([np.array(triples), idd], 0)
        self.n_fact = len(self.KG)
        self.M_sub = csr_matrix((np.ones((self.n_fact,)), (np.arange(self.n_fact), self.KG[:,0])), shape=(self.n_fact, self.n_ent))

        # print("load_graph: KG:", self.KG)

    def load_test_graph(self, triples):
        idd = np.concatenate([np.expand_dims(np.arange(self.n_ent),1), 2*self.n_rel*np.ones((self.n_ent, 1)), np.expand_dims(np.arange(self.n_ent),1)], 1)

        # print("load_test_graph: idd:", idd)
        self.tKG = np.concatenate([np.array(triples), idd], 0)
        self.tn_fact = len(self.tKG)
        self.tM_sub = csr_matrix((np.ones((self.tn_fact,)), (np.arange(self.tn_fact), self.tKG[:,0])), shape=(self.tn_fact, self.n_ent))

        # print("load_test_graph: tKG:", self.tKG)
        # print("load_test_graph: tM_sub", self.tM_sub)

    def load_query(self, triples):
        triples.sort(key=lambda x:(x[0], x[1])) # sorted by user and item
        trip_hr = defaultdict(lambda:list())

        for trip in triples:
            h, r, t = trip
            trip_hr[(h,r)].append(t) # record the 'answer'
        
        queries = []
        answers = []
        for key in trip_hr:
            queries.append(key) 
            answers.append(np.array(trip_hr[key]))
        return queries, answers

    def get_neighbors(self, nodes, mode='train'):
        if mode=='train':
            KG = self.KG
            M_sub = self.M_sub
        else:
            KG = self.tKG
            M_sub = self.tM_sub

        # nodes: n_node x 2 with (batch_idx, node_idx)
        node_1hot = csr_matrix((np.ones(len(nodes)), (nodes[:,1], nodes[:,0])), shape=(self.n_ent, nodes.shape[0]))

        # print("get_neighbors node_1hot:", node_1hot, node_1hot.shape)

        edge_1hot = M_sub.dot(node_1hot)

        # print("edge_1hot:", edge_1hot, edge_1hot.shape)

        edges = np.nonzero(edge_1hot)

        # print("edges:", edges)

        sampled_edges = np.concatenate([np.expand_dims(edges[1],1), KG[edges[0]]], axis=1)     # (batch_idx, head, rela, tail)
        sampled_edges = torch.LongTensor(sampled_edges).cuda()

        # print("sampled_edges", sampled_edges.shape, sampled_edges)

        # index to nodes
        head_nodes, head_index = torch.unique(sampled_edges[:,[0,1]], dim=0, sorted=True, return_inverse=True)
        tail_nodes, tail_index = torch.unique(sampled_edges[:,[0,3]], dim=0, sorted=True, return_inverse=True)

        # print("head_nodes, head_index", head_nodes, head_index)
        # print("tail_nodes, tail_index", tail_nodes, tail_index)

        sampled_edges = torch.cat([sampled_edges, head_index.unsqueeze(1), tail_index.unsqueeze(1)], 1)
       
        # print("after sampled_edges", sampled_edges, sampled_edges.shape)
        # exit(0)

        mask = sampled_edges[:,2] == (self.n_rel*2)

        # print("mask", mask, mask.shape)

        _, old_idx = head_index[mask].sort()

        # print("head_index[mask].sort:", head_index[mask].sort())

        old_nodes_new_idx = tail_index[mask][old_idx]

        # print("old_nodes_new_idx", old_nodes_new_idx)
    
        return tail_nodes, sampled_edges, old_nodes_new_idx
   

    # def get_neighbors(self, nodes, mode='train'):
    #     if mode=='train':
    #         KG = self.KG
    #         M_sub = self.M_sub
    #     else:
    #         KG = self.tKG
    #         M_sub = self.tM_sub

    #     # nodes: n_node x 2 with (batch_idx, node_idx)
    #     node_1hot = csr_matrix((np.ones(len(nodes)), (nodes[:,1], nodes[:,0])), shape=(self.n_ent, nodes.shape[0]))
    #     edge_1hot = M_sub.dot(node_1hot)
    #     edges = np.nonzero(edge_1hot)
    #     sampled_edges = np.concatenate([np.expand_dims(edges[1],1), KG[edges[0]]], axis=1)  # (batch_idx, head, rela, tail)
    #     sampled_edges = torch.LongTensor(sampled_edges).cuda()

    #     # index to nodes
    #     head_nodes, head_index = torch.unique(sampled_edges[:, [0, 1]], dim=0, sorted=True, return_inverse=True)
    #     tail_nodes, tail_index = torch.unique(sampled_edges[:, [0, 3]], dim=0, sorted=True, return_inverse=True)

    #     # Sample up to num_neighbor edges for each head node
    #     sampled_edges = torch.cat([sampled_edges, head_index.unsqueeze(1), tail_index.unsqueeze(1)], 1)

    #     # Create a mask to filter edges based on the head node and limit the number of neighbors
    #     # mask = sampled_edges[:, 2] < self.n_rel * 2  # Exclude self-loops and edges with relation index greater than or equal to n_rel * 2
    #     # filtered_edges = sampled_edges[mask]

    #     # Sort edges by head node and then by relation index to ensure a consistent order
    #     sort_indices = torch.argsort(sampled_edges[:, 0] * self.n_rel + sampled_edges[:, 2], dim=0)
    #     sorted_edges = sampled_edges[sort_indices]
    #     print("sorted edges:", sorted_edges)

    #     # Group edges by head node and sample up to num_neighbor edges for each group
    #     head_node_groups, group_indices = torch.unique(sorted_edges[:, 0], return_inverse=True, dim=0)
    #     sampled_groups = []
    #     for head_node in head_node_groups:
    #         group_edges = sorted_edges[group_indices == head_node]
    #         if len(group_edges) > self.num_neighbor:
    #             sampled_group = group_edges[np.random.choice(len(group_edges), self.num_neighbor, replace=False)]
    #         else:
    #             sampled_group = group_edges
    #         sampled_groups.append(sampled_group)

    #     # Concatenate the sampled groups to form the final list of edges
    #     final_sampled_edges = torch.cat(sampled_groups, dim=0)
    #     print("final sampled_edges", final_sampled_edges)
    #     exit(0)
    #     # Create a mapping from old node indices to new indices after sampling
    #     old_nodes_new_idx = {}
    #     for i, edge in enumerate(final_sampled_edges):
    #         old_nodes_new_idx[(edge[1], edge[3])] = i

    #     return tail_nodes, final_sampled_edges, old_nodes_new_idx

    def get_batch(self, batch_idx, steps=2, data='train'): # this for rec
        if data=='train':
            train_triple = np.array(self.train_data)[batch_idx]
            train_negitem = []
            for t in train_triple:
                train_negitem.append(self.train_neg_set[t[0]])
            train_negitem = np.array(train_negitem)
            return train_triple, train_negitem

        # if data=='test': # to be changed by the format of dataset!
        #     test_triples = np.array(self.test_data)[batch_idx]
            
        # return test_triples
        if data=='test':
            query, answer = np.array(self.test_q), np.array(self.test_a)

        subs = []
        rels = []
        subs = query[batch_idx, 0]
        rels = query[batch_idx, 1]
        combined_tuples = np.hstack((subs.reshape(-1, 1), rels.reshape(-1, 1)))
        return combined_tuples

    # def get_batch(self, batch_idx, steps=2, data='train'):
    #     if data=='train':
    #         train_triple = np.array(self.train_data)[batch_idx]
    #         train_negitem = []
    #         for t in train_triple:
    #             train_negitem.append(self.train_neg_set[t[0]])
    #         train_negitem = np.array(train_negitem)
    #         return train_triple, train_negitem
    #     if data=='valid':
    #         query, answer = np.array(self.valid_q), np.array(self.valid_a)
    #     if data=='test':
    #         query, answer = np.array(self.test_q), np.array(self.test_a)

    #     subs = []
    #     rels = []
    #     objs = []
    #     # to be done : get the neg_item in valid and test
    #     subs = query[batch_idx, 0]
    #     rels = query[batch_idx, 1]
    #     combined_tuples = np.hstack((subs.reshape(-1, 1), rels.reshape(-1, 1)))

    #     # objs = np.zeros((len(batch_idx), self.n_ent)) # 原objs起label作用
    #     # for i in range(len(batch_idx)):
    #     #     objs[i][answer[batch_idx[i]]] = 1
    #     objs = np.zeros((len(batch_idx), 1 + self.v_t_neg_size))
    #     objs[:, 0] = 1
    #     # get the neg_item
    #     v_t_neg_item = []
    #     for u in subs: # 遍历用户
    #         if data=='valid':
    #             v_t_neg_item.append(self.valid_neg_set[u])
    #         if data == 'test':
    #             v_t_neg_item.append(self.test_neg_set[u])
                
    #     # print("valid and test combined_tuples:", combined_tuples)
    #     # print("objs:", objs)
    #     # print("v_t_neg_item", v_t_neg_item)

    #     return combined_tuples, objs, v_t_neg_item

    # def shuffle_train(self,):
    #     # fact_triple = np.array(self.fact_triple)
    #     # train_triple = np.array(self.train_triple)
    #     # all_triple = np.concatenate([fact_triple, train_triple], axis=0)
    #     all_triple = np.array(self.fact_triple)
    #     n_all = len(all_triple)
    #     rand_idx = np.random.permutation(n_all)
    #     all_triple = all_triple[rand_idx]

    #     # increase the ratio of fact_data, e.g., 3/4->4/5, can increase the performance
    #     self.fact_data = self.double_triple(all_triple[:n_all*3//4].tolist()) # 图谱，建图，先保留反向边
    #     # self.train_data = np.array(self.double_triple(all_triple[n_all*3//4:].tolist()))
    #     self.train_data = np.array(all_triple[n_all*3//4:].tolist()) # keep user-rating-item
    #     self.n_train = len(self.train_data)
    #     self.load_graph(self.fact_data)
    def shuffle_data(self):
        train_triple2 = np.array(self.train_triple)
        n_train = len(train_triple2)
        rand_idx = np.random.permutation(n_train)
        train_triple2 = train_triple2[rand_idx]
        self.train_data = np.array(train_triple2.tolist())


    def get_user_set(self,):
        train_user_set = defaultdict(list) # init
        train_user_setfrom0 = defaultdict(list)
        valid_user_set = defaultdict(list)
        test_user_set = defaultdict(list)
        for train_triple in self.train_data:
            user, item = train_triple[0], train_triple[2]
            train_user_set[int(user)].append(int(item)) # 记录每个user对应的item
            train_user_setfrom0[int(user)].append(int(item-self.n_user))
        for valid_triple in self.valid_data:
            valid_user_set[int(valid_triple[0])].append(int(valid_triple[2]))
        for test_triple in self.test_data:# begin from 0
            test_user_set[int(test_triple[0])].append(int(test_triple[2]-self.n_user))
        self.train_user_setfrom0 = train_user_setfrom0
        return train_user_set, valid_user_set, test_user_set


    def get_neg_set(self):
        # 为每个用户生成负例
        train_neg_set = defaultdict(list) # key->user, value->itemList
        train_user_set = self.train_user_set # user + pos_itemList
        for user in range(self.n_user): # 对每个用户生成neg_size个，用于训练
            if train_neg_set[user]: # avoid repetitive operations
                continue
            while True and len(train_neg_set[user]) < self.neg_size:
                neg_item = random.randint(self.n_user, self.n_ent - 1) # 从所有实体中随机选择
                if neg_item not in train_user_set[user]: # 如果负例不和用户有交互
                    train_user_set[int(user)].append(neg_item) # 现在产生交互
                    train_neg_set[user].append(neg_item) # 不考虑rels
        self.train_neg_set = train_neg_set
        # print("get_neg_set: train_neg_set:", train_neg_set)
        
        # valid_neg_set = defaultdict(list)
        # valid_user_set = self.valid_user_set
        # for vuser in range(self.n_user):
        #     if valid_neg_set[vuser]:
        #         continue
        #     while True and len(valid_neg_set[vuser]) < self.v_t_neg_size:
        #         vneg_item = random.randint(self.n_user, self.n_ent - 1)
        #         if vneg_item not in valid_user_set[vuser]:
        #             valid_user_set[int(vuser)].append(vneg_item)
        #             valid_neg_set[vuser].append(vneg_item)
        # self.valid_neg_set = valid_neg_set

        # # print("get_neg_set: valid_neg_set:", valid_neg_set)
       
        # test_neg_set = defaultdict(list)
        # test_user_set = self.test_user_set
        # for tuser in range(self.n_user):
        #     if test_neg_set[tuser]:
        #         continue
        #     while True and len(test_neg_set[tuser]) < self.v_t_neg_size:
        #         tneg_item = random.randint(self.n_user, self.n_ent - 1)
        #         if tneg_item not in test_user_set[tuser]:
        #             test_user_set[int(tuser)].append(tneg_item)
        #             test_neg_set[tuser].append(tneg_item)
        # self.test_neg_set = test_neg_set  
        
        # print("get_neg_set: test_neg_set:", test_neg_set)

    def load_train_user2rating(self, train_data):
        user2rating = {}
        self.rel2rating=[1,2,3,4,5,1,2,3,4,5,5]

        for triple in train_data:
            u,r= triple[0], triple[1]
            rating = self.rel2rating[r]
            # print("load_train_user2rating: r == rating?", r, rating)
            if u not in user2rating:
                user2rating[u] = []
            user2rating[u].append(rating)

        self.user2rating = user2rating
        
class DataLoader_Amazon:
    def __init__(self, args):
        self.args = args
        self.task_dir = args.data_path
        with open(os.path.join(self.task_dir, 'relations.txt')) as f:
            self.relation2id = dict()
            for line in f:
                relation, rid = line.strip().split()
                self.relation2id[relation] = int(rid) # 关系映射id

        self.n_rel = len(self.relation2id) # 关系个数
        self.filters = defaultdict(lambda:set()) # 记录(h,r,t)在(h,r)确定下的所有t
        
        self.fact_triple  = self.read_triples('facts.txt') # 图谱三元组
        self.train_triple = self.read_triples('train.txt') # 训练集三元组
        self.test_triple  = self.read_triples('test_full.txt')  # 全部的测试集三元组

        self.n_user = max(max(t[0] for t in self.train_triple), max(t[0] for t in self.test_triple)) + 1 # 用户个数
        self.n_ent = max(max(t[2] for t in self.train_triple), max(t[2] for t in self.test_triple)) + 1 # 实体个数
        self.n_item = self.n_ent - self.n_user # item个数
        print("n_user:",self.n_user, "n_item:", self.n_item, "n_ent:", self.n_ent)

        self.fact_data  = self.double_triple(self.fact_triple) # double, 用于图信息传递
        # self.train_data = np.array(self.double_triple(self.train_triple))
        # self.test_data  = self.double_triple(self.test_triple)
        self.train_data = self.train_triple # 不double，保证是user到item
        self.test_data  = self.test_triple # test中的全部三元组，随机选取进行测试

        self.load_graph(self.fact_data) # KG
        # self.load_test_graph(self.double_triple(self.fact_triple)+self.double_triple(self.train_triple)) # tKG (previous)
        self.load_test_graph(self.double_triple(self.fact_triple)) # now: tKG = KG
        self.user2rating, self.test_user2rating = self.load_user2rating(self.train_data, self.test_data) # 对于每个用户获取其打的分数，不存在分数的记录为[]
        
        # self.valid_q, self.valid_a = self.load_query(self.valid_data) # record the query and answer for valid data
        # self.test_q,  self.test_a  = self.load_query(self.test_data) # record the query and answer for test data

        self.neg_size = args.neg_num # for train，每个正例找一个负例
        self.num_neighbor = args.num_neighbor # 图中找邻居的最大个数
        # self.train_user_set, self.valid_user_set, self.test_user_set = self.get_user_set() # record the set item of every user
        self.train_user_set, self.test_user_set = self.get_user_set()
        self.train_data = self.train_data[:1000] # 小样本，看test是否有问题
        self.n_train = len(self.train_data) # 训练集个数
        self.n_test  = len(self.test_user_set.keys()) # 对存在正例的用户进行测试
        self.get_neg_set()

        for filt in self.filters:
            self.filters[filt] = list(self.filters[filt])

        print('n_train:', self.n_train, 'n_test:', self.n_test)

    def read_triples(self, filename): # 读取三元组
        triples = []
        with open(os.path.join(self.task_dir, filename)) as f:
            for line in f:
                h, r, t = line.strip().split()
                r = self.relation2id[r]
                h, t = int(h), int(t)
                # h, r, t = self.entity2id[h], self.relation2id[r], self.entity2id[t]
                triples.append([h,r,t])
                self.filters[(h,r)].add(t)
                self.filters[(t,r+self.n_rel)].add(h) 
        return triples
    
    def double_triple(self, triples): # double，增加反向边
        new_triples = []
        for triple in triples:
            h, r, t = triple
            new_triples.append([t, r+self.n_rel, h]) 
        return triples + new_triples

    def load_graph(self, triples):
        # 自关系
        idd = np.concatenate([np.expand_dims(np.arange(self.n_ent),1), 2*self.n_rel*np.ones((self.n_ent, 1)), np.expand_dims(np.arange(self.n_ent),1)], 1)
        # 生成图谱
        self.KG = np.concatenate([np.array(triples), idd], 0)
        myKG = defaultdict(list)
        for triple in triples: # 不加自环
            head, rel, tail = int(triple[0]), int(triple[1]), int(triple[2])

            myKG[head].append([rel, tail])
        self.myKG = myKG
        self.n_fact = len(self.KG)
        self.M_sub = csr_matrix((np.ones((self.n_fact,)), (np.arange(self.n_fact), self.KG[:,0])), shape=(self.n_fact, self.n_ent))

    def load_test_graph(self, triples):
        idd = np.concatenate([np.expand_dims(np.arange(self.n_ent),1), 2*self.n_rel*np.ones((self.n_ent, 1)), np.expand_dims(np.arange(self.n_ent),1)], 1)

        self.tKG = np.concatenate([np.array(triples), idd], 0)
        self.tn_fact = len(self.tKG)
        self.tM_sub = csr_matrix((np.ones((self.tn_fact,)), (np.arange(self.tn_fact), self.tKG[:,0])), shape=(self.tn_fact, self.n_ent))

    # def load_query(self, triples):
    #     triples.sort(key=lambda x:(x[0], x[1])) # sorted by user and item
    #     trip_hr = defaultdict(lambda:list())

    #     for trip in triples:
    #         h, r, t = trip
    #         trip_hr[(h,r)].append(t) # record the 'answer'
        
    #     queries = []
    #     answers = []
    #     for key in trip_hr:
    #         queries.append(key) 
    #         answers.append(np.array(trip_hr[key]))
    #     return queries, answers
    def get_neighbors_withoutmatrix(self, nodes, mode='train'):
        if mode=='train':
            KG = self.KG
            M_sub = self.M_sub
        else:
            KG = self.tKG
            M_sub = self.tM_sub

        # nodes: n_node x 2 with (batch_idx, node_idx) 编号+头结点id
        id_heads = []
        rels_tails = []
        for node in nodes: # batch_idx node_idx
            batch_idx, head = node[0], node[1]
            myKG_head = self.myKG[head]
            if len(myKG_head) > self.num_neighbor - 1:
                id_heads.extend([(batch_idx, head)] * self.num_neighbor) # 多，保证有一个给自环
                rels_tails_sample = random.sample(myKG_head, self.num_neighbor-1)
                rels_tails.extend(rels_tails_sample)
            else: # 不足，全部加入
                id_heads.extend([(batch_idx, head)] * (len(myKG_head) + 1)) # 给自环
                rels_tails.extend(myKG_head)
            rels_tails.append([self.n_rel*2, head]) # 自环
        id_heads = np.array(id_heads)
        rels_tails = np.array(rels_tails)
        my_sampled_edges = np.hstack((id_heads, rels_tails))
        my_sampled_edges = torch.LongTensor(my_sampled_edges).cuda()
        # index to nodes
        my_head_nodes, my_head_index = torch.unique(my_sampled_edges[:,[0,1]], dim=0, sorted=True, return_inverse=True)
        my_tail_nodes, my_tail_index = torch.unique(my_sampled_edges[:,[0,3]], dim=0, sorted=True, return_inverse=True)

        my_sampled_edges = torch.cat([my_sampled_edges, my_head_index.unsqueeze(1), my_tail_index.unsqueeze(1)], 1)
        my_mask = my_sampled_edges[:,2] == (self.n_rel*2)
        _, my_old_idx = my_head_index[my_mask].sort()
        my_old_nodes_new_idx = my_tail_index[my_mask][my_old_idx]
        return my_tail_nodes, my_sampled_edges, my_old_nodes_new_idx
        
    def get_neighbors(self, nodes, mode='train'):
        def sample_edges_for_heads(edges, num=self.num_neighbor): # 取样
            sampled_edges = [] # result
            head_edges = defaultdict(list) # 记录头结点出发的所有边
            for edge in edges:
                head = edge[1]
                tail = edge[3]
                if head == tail: # 自环
                    sampled_edges.append(edge) # 保证自环的存在，后面步骤需要自环
                else:
                    head_edges[head].append(edge) # 加入到边里
            
            for head, edges in head_edges.items():
                if len(edges) <= num - 1: # 已经有一个自环
                    sampled_edges.extend(edges)
                else: 
                    import random
                    sampled_edges.extend(random.sample(edges, num - 1)) # 随机选取
            return np.array(sampled_edges, dtype=np.int32)

        if mode=='train':
            KG = self.KG
            M_sub = self.M_sub
        else:
            KG = self.tKG
            M_sub = self.tM_sub

        # nodes: n_node x 2 with (batch_idx, node_idx) 编号+头结点id
        node_1hot = csr_matrix((np.ones(len(nodes)), (nodes[:,1], nodes[:,0])), shape=(self.n_ent, nodes.shape[0])) # (node_idx, batch_idx) 1.0
        edge_1hot = M_sub.dot(node_1hot) # [batch_idx, 头结点id]
        edges = np.nonzero(edge_1hot) 
        sampled_edges = np.concatenate([np.expand_dims(edges[1],1), KG[edges[0]]], axis=1)     # (batch_idx, head, rela, tail)
        # sampled_edges_tolist = sampled_edges.tolist()
        # print("sampled before:", len(sampled_edges))
        sampled_edges_tolist = [[int(x) for x in row] for row in sampled_edges]
        sampled_edges = sample_edges_for_heads(sampled_edges_tolist)
        sampled_edges = torch.LongTensor(sampled_edges).cuda() # 实际上一个idx对应一个head
        # print("sampled after:", len(sampled_edges))
        # index to nodes
        head_nodes, head_index = torch.unique(sampled_edges[:,[0,1]], dim=0, sorted=True, return_inverse=True)
        tail_nodes, tail_index = torch.unique(sampled_edges[:,[0,3]], dim=0, sorted=True, return_inverse=True)

        sampled_edges = torch.cat([sampled_edges, head_index.unsqueeze(1), tail_index.unsqueeze(1)], 1)
       
        mask = sampled_edges[:,2] == (self.n_rel*2)
        _, old_idx = head_index[mask].sort()
        old_nodes_new_idx = tail_index[mask][old_idx]
        return tail_nodes, sampled_edges, old_nodes_new_idx

    # def get_neighbors(self, nodes, mode='train'):
    #     if mode=='train':
    #         KG = self.KG
    #         M_sub = self.M_sub
    #     else:
    #         KG = self.tKG
    #         M_sub = self.tM_sub

    #     # nodes: n_node x 2 with (batch_idx, node_idx)
    #     node_1hot = csr_matrix((np.ones(len(nodes)), (nodes[:,1], nodes[:,0])), shape=(self.n_ent, nodes.shape[0]))
    #     edge_1hot = M_sub.dot(node_1hot)
    #     edges = np.nonzero(edge_1hot)
    #     sampled_edges = np.concatenate([np.expand_dims(edges[1],1), KG[edges[0]]], axis=1)     # (batch_idx, head, rela, tail)
    #     sampled_edges = torch.LongTensor(sampled_edges).cuda()
        
    #     # index to nodes
    #     head_nodes, head_index = torch.unique(sampled_edges[:,[0,1]], dim=0, sorted=True, return_inverse=True)
    #     tail_nodes, tail_index = torch.unique(sampled_edges[:,[0,3]], dim=0, sorted=True, return_inverse=True)

    #     sampled_edges = torch.cat([sampled_edges, head_index.unsqueeze(1), tail_index.unsqueeze(1)], 1)
       
    #     mask = sampled_edges[:,2] == (self.n_rel*2)
    #     _, old_idx = head_index[mask].sort()
    #     old_nodes_new_idx = tail_index[mask][old_idx]
    
    #     return tail_nodes, sampled_edges, old_nodes_new_idx

    # def get_batch(self, batch_idx, steps=2, data='train'):
    #     if data=='train':
    #         train_triple = np.array(self.train_data)[batch_idx]
    #         train_negitem = []
    #         for t in train_triple:
    #             train_negitem.append(self.train_neg_set[t[0]])
    #         train_negitem = np.array(train_negitem)
    #         return train_triple, train_negitem
    #     if data=='valid':
    #         query, answer = np.array(self.valid_q), np.array(self.valid_a)
    #     if data=='test':
    #         query, answer = np.array(self.test_q), np.array(self.test_a)

    #     subs = []
    #     rels = []
    #     objs = []
    #     # to be done : get the neg_item in valid and test
    #     subs = query[batch_idx, 0]
    #     rels = query[batch_idx, 1]
    #     combined_tuples = np.hstack((subs.reshape(-1, 1), rels.reshape(-1, 1)))

    #     # objs = np.zeros((len(batch_idx), self.n_ent)) # 原objs起label作用
    #     # for i in range(len(batch_idx)):
    #     #     objs[i][answer[batch_idx[i]]] = 1
    #     objs = np.zeros((len(batch_idx), 1 + self.v_t_neg_size))
    #     objs[:, 0] = 1
    #     # get the neg_item
    #     v_t_neg_item = []
    #     for u in subs: # 遍历用户
    #         if data=='valid':
    #             v_t_neg_item.append(self.valid_neg_set[u])
    #         if data == 'test':
    #             v_t_neg_item.append(self.test_neg_set[u])
    #     return combined_tuples, objs, v_t_neg_item
    def random_select(self, test_data, batch_idx):
        subs = []
        rels = []
        test_users = list(self.test_user_set.keys())
        start = int(batch_idx[0])
        end = int(batch_idx[-1]) + 1
        subs = test_users[start:end] # 用户按顺序获取即可
        for sub in subs: # 对每个用户，随机找寻一个关系
            rels.append(random.choice(self.test_user2rating[sub]))
        return np.array(subs), np.array(rels)

    def get_batch(self, batch_idx, steps=2, data='train'):
        if data=='train': # 训练，一个正例对应一个负例
            train_triple = np.array(self.train_data)[batch_idx] # 正例三元组
            train_negitem = []
            for t in train_triple:
                train_negitem.append(self.train_neg_set[t[0]]) # 把用户对应的负例加进去
            train_negitem = np.array(train_negitem)
            return train_triple, train_negitem

        if data=='test': # test的话，每个用户随机选择一个关系进行测试，得到所有物品分数
            subs, rels = self.random_select(self.test_data, batch_idx)
            combined_tuples = np.hstack((subs.reshape(-1, 1), rels.reshape(-1, 1))) # 组合一下
            return combined_tuples
        

    # def shuffle_train(self,):
    #     # fact_triple = np.array(self.fact_triple)
    #     # train_triple = np.array(self.train_triple)
    #     # all_triple = np.concatenate([fact_triple, train_triple], axis=0)
    #     all_triple = np.array(self.fact_triple)
    #     n_all = len(all_triple)
    #     rand_idx = np.random.permutation(n_all)
    #     all_triple = all_triple[rand_idx]

    #     # increase the ratio of fact_data, e.g., 3/4->4/5, can increase the performance
    #     self.fact_data = self.double_triple(all_triple[:n_all*4//5].tolist()) # 图谱，建图，先保留反向边
    #     # self.train_data = np.array(self.double_triple(all_triple[n_all*3//4:].tolist()))
    #     self.train_data = np.array(all_triple[n_all*4//5:].tolist()) # keep user-rating-item
    #     self.n_train = len(self.train_data)
    #     self.load_graph(self.fact_data)

    def shuffle_data(self): # 打乱训练集顺序
        train_triple2 = np.array(self.train_triple)
        n_train = len(train_triple2)
        rand_idx = np.random.permutation(n_train)
        train_triple2 = train_triple2[rand_idx]
        self.train_data = np.array(train_triple2.tolist())


    def get_user_set(self,):
        train_user_set = defaultdict(list) # init, item从n_user开始，用于从models的分数中直接取列
        train_user_setfrom0 = defaultdict(list) # item从0开始记录，用于test
        test_user_set = defaultdict(list) 
        for train_triple in self.train_data:
            user, item = train_triple[0], train_triple[2]
            train_user_set[int(user)].append(int(item)) # 记录每个user对应的item
            train_user_setfrom0[int(user)].append(int(item-self.n_user)) # 与KGAT中的编号形式对应
        for test_triple in self.test_data:# begin from 0，用于test
            test_user_set[int(test_triple[0])].append(int(test_triple[2]-self.n_user))
        self.train_user_setfrom0 = train_user_setfrom0
        return train_user_set, test_user_set

    def get_neg_set(self):
        # 为每个用户生成负例
        train_neg_set = defaultdict(list) # key->user, value->itemList
        train_user_set = self.train_user_set # user + pos_itemList
        for user in range(self.n_user): # 对每个用户生成neg_size个，用于训练
            if train_neg_set[user]: # avoid repetitive operations
                continue
            while True and len(train_neg_set[user]) < self.neg_size:
                neg_item = random.randint(self.n_user, self.n_ent - 1) # 从所有实体中随机选择
                if neg_item not in train_user_set[user]: # 如果负例不和用户有交互
                    train_user_set[int(user)].append(neg_item) # 现在产生交互
                    train_neg_set[user].append(neg_item) # 不考虑rels
        self.train_neg_set = train_neg_set

    
    def load_user2rating(self, train_data, test_data):
        user2rating = defaultdict(list)
        test_user2rating = defaultdict(list)
        self.rel2rating=[1,2,3,4,5,1,2,3,4,5,5] # 训练，需要用到真实的rating值，而非映射
        for triple in train_data:
            u,r= triple[0], triple[1]
            rating = self.rel2rating[r] # r = 0, 真实打分为1分
            user2rating[u].append(rating)
        for t in test_data:
            u, r = t[0], t[1]
            test_user2rating[u].append(r)
        return user2rating, test_user2rating

        


        
    

