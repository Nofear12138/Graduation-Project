import torch
import torch.nn as nn
import numpy as np
import time
import os
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from models import RED_GNN
from load_data import DataLoader_Rec
from utils import get_performance, ranklist_by_heapq
from tqdm import tqdm
import multiprocessing

class BaseModel(object):
    def __init__(self, args, loader):
        self.model = RED_GNN(args, loader)
        self.model.cuda()

        self.loader = loader
        self.n_ent = loader.n_ent     # number of entities
        self.n_rel = loader.n_rel     # number of relations
        self.n_user = loader.n_user   # number of users
        self.n_batch = args.n_batch   #  batch_size 4 train
        self.n_tbatch = args.n_tbatch # batch_size 4 eval
        self.n_train = loader.n_train # length of train_data
        # self.n_valid = loader.n_valid # length of valid_data
        self.n_test  = loader.n_test  # length of test_data
        self.n_layer = args.n_layer   # number of layer 4 gnn
        self.Ks = eval(args.Ks)
        self.args = args

        self.optimizer = Adam(self.model.parameters(), lr=args.lr, weight_decay=args.lamb)
        self.scheduler = ExponentialLR(self.optimizer, args.decay_rate)
        
        self.smooth = 1e-5
        self.t_time = 0
        self.lambda1 = 0.05
        self.lambda2 = 0.1
        self.test_user_set = loader.test_user_set
        print("__init__ len:",len(self.test_user_set))
        
        self.cores = multiprocessing.cpu_count() // 2
    
    def train_batch(self, epoch):
        self.loader.shuffle_data()
        epoch_loss = 0
        i = 0
        batch_size = self.n_batch
        n_batch = self.n_train // batch_size + (self.n_train % batch_size > 0)
        train_s_time = time.time()

        self.model.train()
        criterion = nn.CrossEntropyLoss() 
        criterion2 = nn.MSELoss(reduce=None)
        for i in tqdm(range(n_batch)):
            
            start = i*batch_size
            end = min(self.loader.n_train, (i+1)*batch_size)
            batch_idx = np.arange(start, end) # batch_idx 
            tr_triples, tr_neg_items = self.loader.get_batch(batch_idx) # get the pos and neg item
            '''loss for rec'''
            self.model.zero_grad()
            scores, pred_rating = self.model(tr_triples, tr_neg_items)
            # pos_scores = scores[[torch.arange(len(scores)).cuda(),torch.LongTensor(triple[:,2]).cuda()]]
            pos_scores = scores[:, 0] # 第一列为正例分数
            neg_scores = scores[:, 1:] # 其余为负例分数
            max_n = torch.max(scores, 1, keepdim=True)[0]
            loss_rec = torch.mean( - pos_scores + max_n + torch.log(torch.sum(torch.exp(scores - max_n), 1))) # 防止溢出，保证数值稳定性  

            # print("loss part: -pos_scores", -pos_scores )
            # print("part sum:", - pos_scores )
            # print("loss part: log_sum:",torch.log(torch.sum(torch.exp(neg_scores), 1)))  
            # print("loss:", loss_rec.item())     
            '''loss for rating'''
            ratings = torch.tensor(tr_triples[:,1], dtype=torch.long).clone().detach().cuda()
            loss_rating = criterion(pred_rating.float(), ratings)
            # regularizer, use MSELOSS, rating1为预测任务里的评分，rating2为确定用户下随机选择的一个评分
            rating2s = [] # get other rating
            for t in tr_triples:
                userid = t[0]
                rating2 = np.array(self.loader.user2rating[userid])
                rating2 = np.random.choice(rating2) 
                rating2s.append(torch.tensor(rating2))
            rating2s = torch.stack(rating2s).cuda()
            rating1s = torch.tensor([self.loader.rel2rating[val] for val in tr_triples[:,1]]).cuda() # batch中评分
            loss_l2 = criterion2(rating1s.float(), rating2s.float())
            # print("loss_rec:" ,loss_rec.item())
            # print("loss_rating:", loss_rating.item(),"weighted loss_rating:",self.lambda1 * loss_rating.item())
            # print("loss_l2", loss_l2.item(), "weighted loss_l2:",self.lambda2*loss_l2.item())
            loss = loss_rec + self.lambda1 * loss_rating + self.lambda2 * loss_l2 # 新loss，只保留第一部分看结果
            # loss = loss_rec
            loss.backward()
            self.optimizer.step()
            # avoid NaN
            for p in self.model.parameters():
                X = p.data.clone()
                flag = X != X
                X[flag] = np.random.random()
                p.data.copy_(X)
            epoch_loss += loss.item()
        self.scheduler.step()
        train_e_time = time.time()
        self.t_time += time.time() - train_s_time
        epoch_train_time = train_e_time - train_s_time
        train_result = {}
        # print("using time %.4f, loss at epoch %d: %.4f" % (epoch_train_time, epoch, epoch_loss))
        train_result['epoch'] = epoch
        train_result['epoch_loss'] = epoch_loss
        train_result['training time'] = epoch_train_time
        return train_result
        # self.loader.shuffle_train()
    
    def _evaluate(self, epoch_loss):
        '''Valid Part'''
        batch_size = self.n_tbatch
        # n_data = self.n_valid
        # n_batch = n_data // batch_size + (n_data % batch_size > 0)
        # self.model.eval()
        # i_time = time.time()
        # all_scores = []
        # all_objs = []
        # for i in range(n_batch):
        #     start = i*batch_size
        #     end = min(n_data, (i+1)*batch_size)
        #     batch_idx = np.arange(start, end)
        #     combined_tuples, objs, neg_items = self.loader.get_batch(batch_idx, data='valid')
        #     scores, _ = self.model(combined_tuples, neg_items, mode='valid')
        #     scores = scores.detach().cpu().numpy()
        #     # filters = []
        #     # subs, rels = combined_tuples[:, 0], combined_tuples[:, 1] 
        #     # for i in range(len(subs)):
        #     #     filt = self.loader.filters[(subs[i], rels[i])]
        #     #     filt_1hot = np.zeros((self.n_ent, ))
        #     #     filt_1hot[np.array(filt)] = 1
        #     #     filters.append(filt_1hot)
             
        #     # filters = np.array(filters)
        #     all_scores.append(scores)
        #     all_objs.append(objs)      
        # all_scores = np.concatenate(all_scores, axis=0)
        # all_objs = np.concatenate(all_objs, axis=0)              
        # v_metrics = recalls_and_ndcgs_at_k(torch.tensor(all_scores), torch.tensor(all_objs), self.Ks)
        '''Testing Part'''
        n_data = self.n_test
        n_batch = n_data // batch_size + (n_data % batch_size > 0)
        self.model.eval()
        t_all_scores = []
        t_all_objs = []
        for i in range(n_batch):
            start = i*batch_size
            end = min(n_data, (i+1)*batch_size)
            batch_idx = np.arange(start, end)
            combined_tuples, objs, neg_items = self.loader.get_batch(batch_idx, data='test')
            scores, _ = self.model(combined_tuples, neg_items, mode='test')
            scores = scores.detach().cpu().numpy()

            # filters = []
            # subs, rels = combined_tuples[:, 0], combined_tuples[:, 1] 
            # for i in range(len(subs)):
            #     filt = self.loader.filters[(subs[i], rels[i])]
            #     filt_1hot = np.zeros((self.n_ent, ))
            #     filt_1hot[np.array(filt)] = 1
            #     filters.append(filt_1hot)
             
            # filters = np.array(filters)
            t_all_scores.append(scores)
            t_all_objs.append(objs)
        t_all_scores = np.concatenate(t_all_scores, axis=0)
        t_all_objs = np.concatenate(t_all_objs, axis=0)                          
        t_metrics = recalls_and_ndcgs_at_k(torch.tensor(t_all_scores), torch.tensor(t_all_objs), self.Ks)
        i_time = time.time() - i_time
        out_str = '[Train] epoch_loss:%.4f\t' %(epoch_loss)
        # out_str += '[VALID] '
        # for k, v in v_metrics.items():
        #     out_str += f"{k}: {v} "
        # out_str+='\n'
        out_str += '[TEST] '
        for k, v in t_metrics.items():
            out_str += f"{k}: {v} "
        
        return out_str
    
    # def test_one_user(self, scores, user):
    #     training_items = self.loader.train_user_setfrom0[user]
    #     user_pos_test = self.test_user_set[user]
    #     all_items = set(range(0, self.n_ent-self.n_user)) # [0, n_item-1]
    #     test_items = list(all_items - set(training_items))
    #     r = ranklist_by_heapq(user_pos_test, test_items, scores, self.Ks, self.n_user)
    #     metrics = get_performance(user_pos_test, r, self.Ks)
    #     # if metrics['recall'][0] != metrics['ndcg'][0]:
    #     #     print("error!!")
    #     #     print("r:",r, "user:",user, "user_pos_test:", user_pos_test)
    #     #     print(metrics)
    #     return metrics

    # def _Test(self):
    #     result = {'recall': np.zeros(len(self.Ks)),
    #               'ndcg': np.zeros(len(self.Ks)) }

    #     pool = multiprocessing.Pool(self.cores)
    #     count = 0
    #     test_users = list(self.loader.test_user_set.keys())
    #     n_test_users = len(test_users) # 有序才能用这个
    #     batch_size = self.n_tbatch
    #     n_data = self.n_test
    #     n_batch = n_data // batch_size + (n_data % batch_size > 0)
    #     self.model.eval()
    #     for i in tqdm(range(n_batch)): # 每个batch计算一遍metrics
    #         start = i*batch_size
    #         end = min(n_data, (i+1)*batch_size)
    #         batch_idx = np.arange(start, end)
    #         test_triple = self.loader.get_batch(batch_idx, data='test') # 这里没对应起来！
    #         # print("test_triple",test_triple)
    #         # user_list_batch = test_users[start: end] # Q:不是0-user的顺序会出问题吗？有序才能用这个
    #         user_list_batch = test_triple[:, 0]
    #         # print("user_list_batch:",user_list_batch)
    #         # exit(0)
    #         scores = self.model(test_triple, mode='test') # 2.5s
    #         scores = scores[:,self.n_user:] # 所有的分数
    #         # print("scores:", scores)
    #         # st = time.time()
    #         # batch_result = []
    #         # print(user_list_batch)
    #         # for u in range(len(user_list_batch)):
    #         #     performance = self.test_one_user(scores[u], user_list_batch[u])
    #         #     batch_result.append(performance) # a list
    #         # ed = time.time()
    #         # print(f"代码执行时间: {ed - st} 秒")
    #         # exit(0)
    #         user_batch_scores_uid = zip(scores, user_list_batch)
    #         batch_result = pool.map(self.test_one_user, user_batch_scores_uid)
    #         count += len(batch_result)

    #         for re in batch_result:
    #             result['recall'] += re['recall']/n_test_users
    #             result['ndcg'] += re['ndcg']/n_test_users
    #             # result['recall'] += re['recall']/n_data
    #             # result['ndcg'] += re['ndcg']/n_data
    #     assert count == n_test_users
    #     pool.close()
    #     return result



    # def evaluate(self, epoch_loss):
    #     batch_size = self.n_tbatch

    #     n_data = self.n_valid
    #     n_batch = n_data // batch_size + (n_data % batch_size > 0)
    #     # ranking = []
    #     self.model_rec.eval()
    #     i_time = time.time()
    #     for i in range(n_batch):
    #         start = i*batch_size
    #         end = min(n_data, (i+1)*batch_size)
    #         batch_idx = np.arange(start, end)
    #         subs, rels, objs = self.loader1.get_batch(batch_idx, data='valid')
    #         scores = self.model_rec(subs, rels, mode='valid').data.cpu().numpy()
    #         filters = []
    #         for i in range(len(subs)):
    #             filt = self.loader1.filters[(subs[i], rels[i])]
    #             filt_1hot = np.zeros((self.n_ent, ))
    #             filt_1hot[np.array(filt)] = 1
    #             filters.append(filt_1hot)
             
    #         filters = np.array(filters)
    #         # ranks = cal_ranks(scores, objs, filters)
    #         # ranking += ranks
    #     # ranking = np.array(ranking)
    #     # v_mrr, v_h1, v_h10 = cal_performance(ranking)
    #     scores = torch.tensor(scores)
    #     objs = torch.tensor(objs)

    #     v_rmse = cal_rmse(scores, objs)
    #     v_mae = cal_mae(scores, objs)


    #     n_data = self.n_test
    #     n_batch = n_data // batch_size + (n_data % batch_size > 0)
    #     # ranking = []
    #     self.model_rec.eval()
    #     for i in range(n_batch):
    #         start = i*batch_size
    #         end = min(n_data, (i+1)*batch_size)
    #         batch_idx = np.arange(start, end)
    #         subs, rels, objs = self.loader1.get_batch(batch_idx, data='test')
    #         scores = self.model_rec(subs, rels, mode='test').data.cpu().numpy()
    #         filters = []
    #         for i in range(len(subs)):
    #             filt = self.loader1.filters[(subs[i], rels[i])]
    #             filt_1hot = np.zeros((self.n_ent, ))
    #             filt_1hot[np.array(filt)] = 1
    #             filters.append(filt_1hot)
             
    #         filters = np.array(filters)
    #     #     ranks = cal_ranks(scores, objs, filters)
    #     #     ranking += ranks
    #     # ranking = np.array(ranking)
    #     # t_mrr, t_h1, t_h10 = cal_performance(ranking)
    #     scores = torch.tensor(scores)
    #     objs = torch.tensor(objs)
    #     t_rmse = cal_rmse(scores, objs)
    #     t_mae = cal_mae(scores, objs)
    #     i_time = time.time() - i_time
    #     train_loss = epoch_loss / len(self.loader2.train_dataset)
    #     # out_str = '[VALID] MRR:%.4f H@1:%.4f H@10:%.4f\t [TEST] MRR:%.4f H@1:%.4f H@10:%.4f \t[TIME] train:%.4f inference:%.4f\n'%(v_mrr, v_h1, v_h10, t_mrr, t_h1, t_h10, self.t_time, i_time)
    #     out_str = '[TEST] train_loss:%.4f [VALID] RMSE:%.4f MAE:%.4f\t [TEST] RMSE:%.4f MAE:%.4f\t[TIME] train:%.4f inference:%.4f\n'%(train_loss, v_rmse, v_mae, t_rmse, t_mae, self.t_time, i_time)
        
    #     # return v_mrr, out_str
    #     return v_rmse, v_mae, out_str

