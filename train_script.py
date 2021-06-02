import numpy as np
import os, glob
import time

import pyarrow as pa
import pyarrow.parquet as pq


import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import *
import torch_resnet_concat as networks
from modules import *

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
plt.rcParams["figure.figsize"] = (5,5)
plt.switch_backend('agg')

run = 0
np.random.seed(run)


def get_args():
    parser = argparse.ArgumentParser(description='Training parameters.')
    parser.add_argument('-batch_size', '--batch_size', type=int, default = 32, help='Batch size')
    parser.add_argument('-epochs', '--epochs' , default = 10, type=int, help='Number of epochs')    
    parser.add_argument('-load_epoch', '--load_epoch' , default = 0, type=int, help='Load epoch')    

    parser.add_argument('-lr', '--lr' , default = 0.001, type=float, help='Learning rate')
    
    parser.add_argument('-resblocks', '--resblocks' , default = 3, type=str, help='Number of blocks in ResNet')
    
    
#     parser.add_argument('-CSC_data_path', '--CSC_data_path',  type=str, help='Path to the input')
#     parser.add_argument('-y_file_path', '--y_file_path', type=str, help='Path to the targets')
#     parser.add_argument('-res_dir', '--res_dir', default = '.', type=str, help='Path to the targets')

    return parser.parse_args()



def main():
    
    args = get_args()
    print ('Got the arguments')
    train_cut = int(len(datasets) * 0.2)
    mass_bins = np.arange(3600,17000+670,670)/1000. # for histogram in eval()
    
    is_cuda, run_logger = True, True
    expt_name = 'TopGun'
    
    if run_logger:
        global f
        if not os.path.isdir('LOGS'):
            os.makedirs('LOGS')
        f = open('LOGS/%s.log'%(expt_name), 'w')
        for d in ['MODELS', 'PLOTS']:
            if not os.path.isdir('%s/%s'%(d, expt_name)):
                os.makedirs('%s/%s'%(d, expt_name))
            
            
    resnet = networks.ResNet(3, args.resblocks, [16, 32])
    if is_cuda: 
        resnet.cuda()
    optimizer = optim.Adam(resnet.parameters(), lr=args.lr)
    
    train_loader, val_loader = train_val_loader(datasets, train_cut, args.batch_size, random_sampler=False)
    train(load_epoch, resnet, optimizer, epochs, train_loader, val_loader, run_logger=True)

if __name__ == "__main__":
    main()