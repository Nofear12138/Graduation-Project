import os
import argparse
import torch
import numpy as np
from load_data import DataLoader_Rec, DataLoader_Rating
from base_model import BaseModel
from utils import select_gpu

parser = argparse.ArgumentParser(description="Parser for RED-GNN")
parser.add_argument('--data_path', type=str, default='data/WN18RR_v1')
parser.add_argument('--seed', type=str, default=1234)

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
    opts.perf_file = os.path.join(results_dir,  dataset_rec + '_perf.txt')

    gpu = select_gpu()
    torch.cuda.set_device(gpu)
    print('gpu:', gpu)

    loader1 = DataLoader_Rec(args.data_path)
    loader2 = DataLoader_Rating(args.data_path, batch_size=80)
     
    opts.n_ent = loader1.n_ent
    opts.n_rel = loader1.n_rel

    if dataset_rec == 'Musical_Instruments':
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
        
    config_str = '%.4f, %.4f, %.6f,  %d, %d, %d, %d, %.4f,%s\n' % (opts.lr, opts.decay_rate, opts.lamb, opts.hidden_dim, opts.attn_dim, opts.n_layer, opts.n_batch, opts.dropout, opts.act)
    print(config_str)
    with open(opts.perf_file, 'a+') as f:
        f.write(config_str)

    model = BaseModel(opts, loader1, loader2)
    
    best_mae = 10e6
    best_rmse = 10e6
    for epoch in range(50):
        rmse, mae, out_str = model.train_batch()
        with open(opts.perf_file, 'a+') as f:
            f.write(out_str)
        if mae < best_mae:
            best_mae = min(mae, best_mae)
            best_str_mae = out_str
            print(str(epoch) + '\t' + best_str_mae)
        if rmse < best_rmse:
            best_rmse = min(rmse, best_rmse)
            best_str_rmse = out_str
            print(str(epoch) + '\t' + best_str_rmse)
    print(best_str_mae)
    print(best_str_rmse)




