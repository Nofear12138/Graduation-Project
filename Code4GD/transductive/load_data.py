import os
import torch
from scipy.sparse import csr_matrix
import numpy as np
from collections import defaultdict

class DataLoader:
    def __init__(self, task_dir):
        self.task_dir = task_dir

        with open(os.path.join(task_dir, 'entities.txt')) as f:
            self.entity2id = dict()
            n_ent = 0
            for line in f:
                entity = line.strip()
                self.entity2id[entity] = n_ent
                n_ent += 1
        
        with open(os.path.join(task_dir, 'relations.txt')) as f:
            self.relation2id = dict()
            n_rel = 0
            for line in f:
                relation = line.strip()
                self.relation2id[relation] = n_rel
                n_rel += 1

        self.n_ent = n_ent
        self.n_rel = n_rel

        self.filters = defaultdict(lambda:set())

        self.fact_triple  = self.read_triples('facts.txt')
        self.train_triple = self.read_triples('train.txt')
        self.valid_triple = self.read_triples('valid.txt')
        self.test_triple  = self.read_triples('test.txt')
        
        # add inverse
        self.fact_data  = self.double_triple(self.fact_triple)
        self.train_data = np.array(self.double_triple(self.train_triple))
        self.valid_data = self.double_triple(self.valid_triple)
        self.test_data  = self.double_triple(self.test_triple)
        
        self.load_graph(self.fact_data)
        self.load_test_graph(self.double_triple(self.fact_triple)+self.double_triple(self.train_triple))

        self.valid_q, self.valid_a = self.load_query(self.valid_data)
        self.test_q,  self.test_a  = self.load_query(self.test_data)

        self.n_train = len(self.train_data)
        self.n_valid = len(self.valid_q)
        self.n_test  = len(self.test_q)
        self.num_neighbor = 5
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

    def double_triple(self, triples):
        new_triples = []
        for triple in triples:
            h, r, t = triple
            new_triples.append([t, r+self.n_rel, h]) 
        return triples + new_triples


    def load_graph(self, triples):
        idd = np.concatenate([np.expand_dims(np.arange(self.n_ent),1), 2*self.n_rel*np.ones((self.n_ent, 1)), np.expand_dims(np.arange(self.n_ent),1)], 1)

        self.KG = np.concatenate([np.array(triples), idd], 0)
        myKG = defaultdict(list)
        for triple in self.KG:
            head, rel, tail = int(triple[0]), int(triple[1]), int(triple[2])

            myKG[head].append([rel, tail])
        self.myKG = myKG
        self.n_fact = len(self.KG)
        # print(self.myKG)
        # exit(0)
        self.M_sub = csr_matrix((np.ones((self.n_fact,)), (np.arange(self.n_fact), self.KG[:,0])), shape=(self.n_fact, self.n_ent))  # 记录[index, head_entities]

    def load_test_graph(self, triples):
        idd = np.concatenate([np.expand_dims(np.arange(self.n_ent),1), 2*self.n_rel*np.ones((self.n_ent, 1)), np.expand_dims(np.arange(self.n_ent),1)], 1)

        self.tKG = np.concatenate([np.array(triples), idd], 0)
        self.tn_fact = len(self.tKG)
        self.tM_sub = csr_matrix((np.ones((self.tn_fact,)), (np.arange(self.tn_fact), self.tKG[:,0])), shape=(self.tn_fact, self.n_ent))

    def load_query(self, triples):
        triples.sort(key=lambda x:(x[0], x[1]))
        trip_hr = defaultdict(lambda:list())

        for trip in triples:
            h, r, t = trip
            trip_hr[(h,r)].append(t)

        queries = []
        answers = []
        for key in trip_hr:
            queries.append(key)
            answers.append(np.array(trip_hr[key]))           
        return queries, answers
    
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
            id_heads.extend([(batch_idx, head)] * len(self.myKG[head]))
            rels_tails.extend(self.myKG[head])
        id_heads = np.array(id_heads)
        rels_tails = np.array(rels_tails)
        my_sampled_edges = np.hstack((id_heads, rels_tails))
        # print(len(sampled_edges))
        # print(len(my_sampled_edges))
        # exit(0)
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
        def sample_edges_for_heads(edges, num=10):
            sampled_edges = []
            head_edges = defaultdict(list)
            for edge in edges:
                head = edge[1]
                tail = edge[3]
                if head == tail:
                    sampled_edges.append(edge) # 保证自环的存在
                else:
                    head_edges[head].append(edge)
            
            for head, edges in head_edges.items():
                if len(edges) <= num - 1:
                    sampled_edges.extend(edges)
                else: 
                    import random
                    sampled_edges.extend(random.sample(edges, num))
            # print("in function:", sampled_edges)
            # exit(0)
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

        # print("sampled before:",len(sampled_edges))
        # sampled_edges_tolist = sampled_edges.tolist()
        # sampled_edges_tolist = [[int(x) for x in row] for row in sampled_edges]
        # sampled_edges = sample_edges_for_heads(sampled_edges_tolist)
        # print("sampled after:",len(sampled_edges))
        sampled_edges = torch.LongTensor(sampled_edges).cuda() # 实际上一个idx对应一个head

        # index to nodes
        head_nodes, head_index = torch.unique(sampled_edges[:,[0,1]], dim=0, sorted=True, return_inverse=True)
        tail_nodes, tail_index = torch.unique(sampled_edges[:,[0,3]], dim=0, sorted=True, return_inverse=True) # 所有的尾部结点，按batch_idx和尾实体排序
        # print(head_index, head_nodes)
        # exit(0)
        sampled_edges = torch.cat([sampled_edges, head_index.unsqueeze(1), tail_index.unsqueeze(1)], 1)
        mask = sampled_edges[:,2] == (self.n_rel*2)
        _, old_idx = head_index[mask].sort()
        old_nodes_new_idx = tail_index[mask][old_idx]
        return tail_nodes, sampled_edges, old_nodes_new_idx


    def get_batch(self, batch_idx, steps=2, data='train'):
        if data=='train':
            return np.array(self.train_data)[batch_idx]
        if data=='valid':
            query, answer = np.array(self.valid_q), np.array(self.valid_a)
        if data=='test':
            query, answer = np.array(self.test_q), np.array(self.test_a)

        subs = []
        rels = []
        objs = []
        
        subs = query[batch_idx, 0]
        rels = query[batch_idx, 1]
        objs = np.zeros((len(batch_idx), self.n_ent))
        for i in range(len(batch_idx)):
            objs[i][answer[batch_idx[i]]] = 1
        return subs, rels, objs

    def shuffle_train(self,):
        fact_triple = np.array(self.fact_triple)
        train_triple = np.array(self.train_triple)
        all_triple = np.concatenate([fact_triple, train_triple], axis=0)
        n_all = len(all_triple)
        rand_idx = np.random.permutation(n_all)
        all_triple = all_triple[rand_idx]

        # increase the ratio of fact_data, e.g., 3/4->4/5, can increase the performance
        self.fact_data = self.double_triple(all_triple[:n_all*3//4].tolist())
        self.train_data = np.array(self.double_triple(all_triple[n_all*3//4:].tolist()))
        self.n_train = len(self.train_data)
        self.load_graph(self.fact_data)

