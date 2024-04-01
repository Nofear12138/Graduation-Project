import os
import argparse
import torch
import torch.distributed as dist
import numpy as np
from load_data import DataLoader_Rec, DataLoader_Amazon
from base_model import BaseModel
from utils import select_gpu
from datetime import datetime
from evaluate import test
parser = argparse.ArgumentParser(description="Parser for RED-GNN")
# parser.add_argument('--data_path', type=str, default='data/Amazon/Musical_test')
# parser.add_argument('--data_path', type=str, default='/root/Code4GD/codes/data/Amazon/Small')
parser.add_argument('--data_path', type=str, default='data/Amazon/Books')
parser.add_argument('--seed', type=str, default=3407)
parser.add_argument('--neg_num', type=int, default=1) # 负例个数
parser.add_argument('--Ks', nargs='?', default='[1, 20, 40, 80, 100]', help='Output sizes of every layer')
parser.add_argument('--num_neighbor', type=int, default=20) 
args = parser.parse_args()

class Options(object):
    pass

if __name__ == '__main__':
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset_rec = args.data_path
    dataset_rec = dataset_rec.split('/')
    if len(dataset_rec[-1]) > 0:
        dataset_rec = dataset_rec[-1]
    else:
        dataset_rec = dataset_rec[-2]

    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    opts = Options
    opts.perf_file = os.path.join(results_dir,  dataset_rec + '_result_full_gpu1_womatrix.txt')

    gpu = select_gpu()
    torch.cuda.set_device(gpu)
    print('gpu:', gpu)
    
    # loader = DataLoader_Rec(args)
    loader = DataLoader_Amazon(args)
    opts.n_ent = loader.n_ent # 实体数
    opts.n_rel = loader.n_rel # 关系数

    if dataset_rec == 'Small':
        opts.lr = 0.0001
        opts.decay_rate = 0.994
        opts.lamb = 0.00014
        opts.hidden_dim = 64
        opts.attn_dim = 5
        opts.n_layer = 5
        opts.dropout = 0.02
        opts.act = 'idd'
        opts.n_batch = 1
        opts.n_tbatch = 1

    if dataset_rec == 'Musical_test':
        opts.lr = 0.0001
        opts.decay_rate = 0.994
        opts.lamb = 0.00014
        opts.hidden_dim = 64
        opts.attn_dim = 5
        opts.n_layer = 5
        opts.dropout = 0.02
        opts.act = 'idd'
        opts.n_batch = 50
        opts.n_tbatch = 50
        opts.Ks = args.Ks
        opts.num_neighbor = args.num_neighbor
    
    if dataset_rec == 'Books': # same as fb
        opts.lr = 0.0009
        opts.decay_rate = 0.9938
        opts.lamb = 0.000080
        opts.hidden_dim = 48
        opts.attn_dim = 5
        opts.n_layer = 3
        opts.dropout = 0.0391
        opts.act = 'idd'
        opts.n_batch = 500
        opts.n_tbatch = 500
        opts.Ks = args.Ks
        opts.num_neighbor = args.num_neighbor

    if dataset_rec == 'BookTest': # same as fb
        opts.lr = 0.0009
        opts.decay_rate = 0.9938
        opts.lamb = 0.000080
        opts.hidden_dim = 32
        opts.attn_dim = 5
        opts.n_layer = 3
        opts.dropout = 0.0391
        opts.act = 'relu'
        opts.n_batch = 40
        opts.n_tbatch = 40
        opts.Ks = args.Ks

    config_str = '%.4f, %.4f, %.6f, %d, %d, %d, %d, %.4f, %s\n' % (opts.lr, opts.decay_rate, opts.lamb, opts.hidden_dim, opts.attn_dim, opts.n_layer, opts.n_batch, opts.dropout, opts.act)
    print(config_str)
    current_time = datetime.today().strftime("%Y-%m-%d %H:%M")
    with open(opts.perf_file, 'a+') as f:
        f.write(current_time+'\n')
        f.write(config_str+'\n')

    model = BaseModel(opts, loader)
    
    for epoch in range(50):
        train_result = model.train_batch(epoch)
        res = test(model, loader, opts)
        result = {**train_result, **res}
        re_str = str(result)
        with open(opts.perf_file, 'a+') as f:
            f.write(re_str+'\n')
        print(re_str)




