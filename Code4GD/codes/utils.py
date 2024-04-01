import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import rankdata
import subprocess
import logging
import heapq

# def get_performance(user_pos_test, r, Ks):
#     recall, ndcg = [], []

#     for K in Ks:
#         recall.append(recall_at_k(r, K, len(user_pos_test)))
#         ndcg.append(ndcg_at_k(r, K, user_pos_test))

#     return {'recall': np.array(recall), 'ndcg': np.array(ndcg) }

def recall_at_k(r, k, all_pos_num):
    r = np.asfarray(r)[:k]
    return np.sum(r) / all_pos_num 


def dcg_at_k(r, k, method=1):
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.

# def ndcg_at_k(r, k, method=1):
#     """Score is normalized discounted cumulative gain (ndcg)
#     Relevance is positive real values.  Can use binary
#     as the previous methods.
#     Returns:
#         Normalized discounted cumulative gain
#     """
#     dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
#     if not dcg_max:
#         return 0.
#     return dcg_at_k(r, k, method) / dcg_max
    
def ndcg_at_k(r, k, ground_truth, method=1):
    GT = set(ground_truth)
    if len(GT) > k :
        sent_list = [1.0] * k
    else:
        sent_list = [1.0]*len(GT) + [0.0]*(k-len(GT))
    dcg_max = dcg_at_k(sent_list, k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max

def ranklist_by_heapq(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)
    # top_values, top_indices = torch.topk(rating, k=10, largest=True)
    # print("Top 10 values:", top_values)
    # print("Indices of top 10 values:", top_indices)
    # exit(0)
    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = 0.
    return r, auc

def get_performance(user_pos_test, r, auc, Ks):
    recall, ndcg= [], []

    for K in Ks:
        recall.append(recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(ndcg_at_k(r, K, user_pos_test))

    return {'recall': np.array(recall), 'ndcg': np.array(ndcg)}



# def ranklist_by_heapq(user_pos_test, test_items, scores, Ks):
#     item_score = {}
#     for i in test_items:
#         item_score[i] = scores[i] # 物品对应得分；

#     K_max = max(Ks)
#     K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)
    
#     # 使用torch.topk获取前五个最大值和对应的下标
#     # top_values, top_indices = torch.topk(scores, k=10, largest=True)
#     # print("Top 10 values:", top_values)
#     # print("Indices of top 10 values:", top_indices)
#     # exit(0)
#     r = []
#     for i in K_max_item_score:
#         if i in user_pos_test:
#             r.append(1)
#         else:
#             r.append(0)
#     return r


# def cal_ranks(scores, labels, filters):
#     scores = scores - np.min(scores, axis=1, keepdims=True) + 1e-8
#     full_rank = rankdata(-scores, method='average', axis=1)
#     filter_scores = scores * filters
#     filter_rank = rankdata(-filter_scores, method='min', axis=1)
#     ranks = (full_rank - filter_rank + 1) * labels      # get the ranks of multiple answering entities simultaneously
#     ranks = ranks[np.nonzero(ranks)]
#     return list(ranks)


# def cal_performance(ranks):
#     mrr = (1. / ranks).sum() / len(ranks)
#     h_1 = sum(ranks<=1) * 1.0 / len(ranks)
#     h_10 = sum(ranks<=10) * 1.0 / len(ranks)
#     return mrr, h_1, h_10

# def recalls_and_ndcgs_at_k(scores, labels, ks):
#     metrics = {}
#     scores = scores.cpu()
#     labels = labels.cpu()
#     answer_count = labels.sum(1)
#     answer_count_float = answer_count.float()
#     labels_float = labels.float()
#     rank = (-scores).argsort(dim=1)
#     cut = rank
#     print("here is calculating the metrics!!:")
#     print("labels:", labels, labels.shape)
#     print("scores:", scores, scores.shape)
#     for k in sorted(ks, reverse=True):
#         cut = cut[:, :k]
#         hits = labels_float.gather(1, cut)
#         metrics['Recall@%d' % k] = (hits.sum(1) / answer_count_float).mean().item()
#         position = torch.arange(2, 2+k)
#         weights = 1 / torch.log2(position.float())
#         dcg = (hits * weights).sum(1)
#         idcg = torch.Tensor([weights[:min(int(n), k)].sum() for n in answer_count]) 
#         ndcg = (dcg / idcg).mean()
#         metrics['NDCG@%d' % k] = ndcg
#     return metrics




def select_gpu():
    return 1
    # nvidia_info = subprocess.run('nvidia-smi', stdout=subprocess.PIPE)
    # gpu_info = False
    # gpu_info_line = 0
    # proc_info = False
    # gpu_mem = []
    # gpu_occupied = set()
    # i = 0
    # for line in nvidia_info.stdout.split(b'\n'):
    #     line = line.decode().strip()
    #     if gpu_info:
    #         gpu_info_line += 1
    #         if line == '':
    #             gpu_info = False
    #             continue
    #         if gpu_info_line % 3 == 2:
    #             mem_info = line.split('|')[2]
    #             used_mem_mb = int(mem_info.strip().split()[0][:-3])
    #             gpu_mem.append(used_mem_mb)
    #     if proc_info:
    #         if line == '|  No running processes found                                                 |':
    #             continue
    #         if line == '+-----------------------------------------------------------------------------+':
    #             proc_info = False
    #             continue
    #         proc_gpu = int(line.split()[1])
    #         #proc_type = line.split()[3]
    #         gpu_occupied.add(proc_gpu)
    #     if line == '|===============================+======================+======================|':
    #         gpu_info = True
    #     if line == '|=============================================================================|':
    #         proc_info = True
    #     i += 1
    # for i in range(0,len(gpu_mem)):
    #     if i not in gpu_occupied:
    #         logging.info('Automatically selected GPU %d because it is vacant.', i)
    #         return i
    # for i in range(0,len(gpu_mem)):
    #     if gpu_mem[i] == min(gpu_mem):
    #         logging.info('All GPUs are occupied. Automatically selected GPU %d because it has the most free memory.', i)
    #         return i
