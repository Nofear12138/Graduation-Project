import torch
import numpy as np
import multiprocessing
import heapq
from tqdm import tqdm
from time import time
from utils import get_performance, ranklist_by_heapq
cores = multiprocessing.cpu_count() // 2

def test_one_user(x):
    # user u's ratings for user u
    rating = x[0]
    # uid
    u = x[1]
    # user u's items in the training set
    try:
        training_items = train_user_set[u]
    except Exception:
        training_items = []
    # user u's items in the test set
    user_pos_test = test_user_set[u] # 测试集里用户的正例

    all_items = set(range(0, n_items))

    test_items = list(all_items - set(training_items)) # 其余的都是负例

    r, auc = ranklist_by_heapq(user_pos_test, test_items, rating, Ks)
    
    metrics = get_performance(user_pos_test, r, auc, Ks)
    print(metrics)
    return metrics

# def test_one_user(x): # 目前的问题：每一个的指标太离谱
#     scores = x[0]
#     user = x[1]
#     try:
#         training_items = train_user_set_from0[user]
#     except Exception:
#         training_items = []

#     user_pos_test = test_user_set[user]

#     all_items = set(range(0, n_items)) # [0, n_item-1]
    
#     test_items = list(all_items - set(training_items))
    
#     r = ranklist_by_heapq(user_pos_test, test_items, scores, Ks)
#     metrics = get_performance(user_pos_test, r, Ks)
#     print(metrics)
#     # if metrics['recall'][0] != metrics['ndcg'][0]:
#     #     print("error!!")
#     #     print("r:",r, "user:",user, "user_pos_test:", user_pos_test)
#     #     print(metrics)
#     return metrics

def test(BaseModel, loader, opts):
    global Ks
    Ks = eval(loader.args.Ks)
    result = {'recall': np.zeros(len(Ks)),
                'ndcg': np.zeros(len(Ks)) }
    global n_users, n_items
    n_users = loader.n_user
    n_items = loader.n_item

    global train_user_set, test_user_set
    train_user_set = loader.train_user_setfrom0 # 从0开始
    test_user_set = loader.test_user_set

    pool = multiprocessing.Pool(cores)
    count = 0

    test_users = list(test_user_set.keys())
    n_test_users = len(test_users)

    batch_size = opts.n_tbatch
    # n_data = loader.n_test
    n_batch = n_test_users // batch_size + (n_test_users % batch_size > 0)
    model = BaseModel.model
    model.eval()
    for i in tqdm(range(n_batch)): # 每个batch计算一遍metrics
        if i == 2:
            exit(0)
        start = i * batch_size
        end = min(n_test_users, (i + 1) * batch_size)
        batch_idx = np.arange(start, end)
        test_triple = loader.get_batch(batch_idx, data='test') # 这里没对应起来！
        # print("test_triple",test_triple)
        # user_list_batch = test_users[start: end] # Q:不是0-user的顺序会出问题吗？有序才能用这个
        user_list_batch = test_triple[:, 0]
        # print("user_list_batch:",user_list_batch)
        # exit(0)
        scores = model(test_triple, mode='test') # 2.5s
        scores = scores[:,n_users:] # 所有的分数
        scores = scores.detach().cpu()
        tensor_str = scores[0].numpy().tolist()
        with open('check.txt', 'w') as f:
            f.write(str(tensor_str))
        # st = time.time()
        # batch_result = []
        # print(user_list_batch)
        # for u in range(len(user_list_batch)):
        #     performance = self.test_one_user(scores[u], user_list_batch[u])
        #     batch_result.append(performance) # a list
        # ed = time.time()
        # print(f"代码执行时间: {ed - st} 秒")
        # exit(0)
        user_batch_scores_uid = zip(scores, user_list_batch)
        batch_result = pool.map(test_one_user, user_batch_scores_uid)
        count += len(batch_result)

        for re in batch_result:
            result['recall'] += re['recall']/n_test_users
            result['ndcg'] += re['ndcg']/n_test_users
            
    assert count == n_test_users
    pool.close()
    return result
