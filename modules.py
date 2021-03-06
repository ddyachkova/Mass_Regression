# import numpy as np
# run = 0
# np.random.seed(run)
# import os, glob
# import time
# import pyarrow as pa
# import pyarrow.parquet as pq
# import torch
# import torch.nn.functional as F
# import torch.optim as optim
# import matplotlib.pyplot as plt
# plt.switch_backend('agg')

# plt.rcParams["figure.figsize"] = (5,5)
# from torch.utils.data import *
# import os

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

def mae_loss_wgtd(pred, true, is_cuda, wgt=1.):
    loss = wgt*(pred-true).abs()
    if is_cuda: 
        loss = loss.cuda()
    return loss.mean()

def transform_y(y, m0_scale = 17):
    return y/m0_scale

def inv_transform(y, m0_scale = 17):
    return y*m0_scale

class ParquetDataset(Dataset):
    def __init__(self, filename, label):
        self.parquet = pq.ParquetFile(filename)
        self.cols = None 
        self.label = label
    def __getitem__(self, index):
        data = self.parquet.read_row_group(index, columns=self.cols).to_pydict()
        data['X_jet'] = np.float32(data['X_jet'][0]) 
        data['jetM'] = transform_y(np.float32(data['jetM']))
        data['iphi'] = np.float32(data['iphi'])/360.
        data['ieta'] = np.float32(data['ieta'])/170.
        data['jetPt'] = np.float32(data['jetPt'])
        data['label'] = self.label

        # Preprocessing
        data['X_jet'] = data['X_jet'][:, 20:105, 20:105]
        data['X_jet'][data['X_jet'] < 1.e-3] = 0. # Zero-Suppression
        return dict(data)
    def __len__(self):
        return self.parquet.num_row_groups
    
    
def train_val_loader(datasets, train_cut, batch_size, random_sampler=True):
    dset = ConcatDataset([ParquetDataset(dataset, datasets.index(dataset)) for dataset in datasets])
    idxs = np.random.permutation(len(dset))
    if random_sampler: 
        train_sampler = sampler.SubsetRandomSampler(idxs[:train_cut])
        val_sampler = sampler.SubsetRandomSampler(idxs[train_cut:])
    else: 
        train_sampler, val_sampler = None, None 
    train_loader = DataLoader(dataset=dset, batch_size=batch_size, shuffle=False, num_workers=0, sampler=train_sampler, pin_memory=True)
    val_loader = DataLoader(dataset=dset, batch_size=120, shuffle=False, num_workers=0, sampler=val_sampler, pin_memory=True)
    return train_loader, val_loader


def logger(s, f):
#     global f, run_logger
#     if run_logger:
    print(s)
    f.write('%s\n'%str(s))
        
def load_model(model_name, resnet, optimizer):
    print(model_name)
    logger('Loading weights from %s'%model_name)
    checkpoint = torch.load(model_name)
    resnet.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return resnet, optimizer

def do_eval(resnet, val_loader, epoch, run_logger):
    global expt_name
    loss_ = 0.
    m_pred_, m_true_, mae_, pt_, iphi_, ieta_, label_ = [], [], [], [], [], [], []
    now = time.time()
    ma_low = transform_y(3.6) # convert from GeV to network units
#     resnet.eval()
    for i, data in enumerate(val_loader):
        if i >= 2: 
            break
        X, m0, pt, iphi, ieta = data['X_jet'].cuda(), data['jetM'].cuda(), data['jetPt'].cuda(), data['iphi'].cuda(), data['ieta'].cuda()
        
        
        logits = resnet([X, iphi, ieta])        # logits
        loss_ += mae_loss_wgtd(logits, m0, is_cuda=True).item()
        
        # Undo preproc on mass
        logits, m0 = inv_transform(logits), inv_transform(m0)
        mae = (logits-m0).abs()

        # Store batch metrics:
        
        m_pred_.append(logits.tolist())
        m_true_.append(m0.tolist())
        mae_.append(mae.tolist())
        
        pt_.append(pt.tolist())
        iphi_.append(iphi.tolist())
        ieta_.append(ieta.tolist())
        label_.append(data['label'].tolist())

    now = time.time() - now
    label_ = np.concatenate(label_)
    m_true_ = np.concatenate(m_true_) #[label_==tgt_label]
    m_pred_ = np.concatenate(m_pred_) #[label_==tgt_label]
    mae_ = np.concatenate(mae_) #[label_==tgt_label]
    pt_ = np.concatenate(pt_) #[label_==tgt_label]
    iphi_ = np.concatenate(iphi_) #[label_==tgt_label]
    ieta_ = np.concatenate(ieta_) #[label_==tgt_label]

    logger('%d: Val m_pred: %s...'%(epoch, str(np.squeeze(m_pred_[:5]))), f)
    logger('%d: Val m_true: %s...'%(epoch, str(np.squeeze(m_true_[:5]))), f)
    logger('%d: Val time:%.2fs in %d steps for N=%d'%(epoch, now, len(val_loader), len(m_true_)), f)
    logger('%d: Val loss:%f, mae:%f'%(epoch, loss_/len(val_loader), np.mean(mae_)), f)
    score_str = 'epoch%d_mae%.4f'%(epoch, np.mean(mae_))

    if epoch %10 ==0:
        # Check 1D m_pred
        hst = np.histogram(np.squeeze(m_pred_), bins=mass_bins)[0]
        logger('%d: Val m_pred, [3600,17000,670] MeV: %s'%(epoch, str(np.uint(hst))))
        mlow = hst[0]
        mrms = np.std(hst)
        logger('%d: Val m_pred, [3600,17000,670] MeV: low:%d, rms: %f'%(epoch, mlow, mrms))
        norm = 1.*len(m_pred_)/len(m0)
        plt.hist(m_true_, range=(-1,17), bins=20, histtype='step', label=r'$\mathrm{m_{true}}$', linestyle='--', color='grey', alpha=0.6)
        plt.hist(m_pred_, range=(-1,17), bins=20, histtype='step', label=r'$\mathrm{m_{pred}}$', linestyle='--', color='C0', alpha=0.6)
        plt.xlim(-1,17)
        plt.xlabel(r'$\mathrm{m}$', size=16)
        plt.legend(loc='upper right')
        plt.show()
        plt.savefig('PLOTS/%s/mpred_%s.png'%(expt_name, score_str), bbox_inches='tight')
        plt.close()
    return np.mean(mae_)


        
def train(load_epoch, resnet, optimizer, epochs, train_loader, val_loader, run_logger=True):
    
    if load_epoch != 0:
        epoch_string = 'MODELS/Tops_ResNet_blocks_3/model_epoch_%d'%(load_epoch)
        model_name = glob.glob('%s*pkl'%(epoch_string))
        resnet, optimizer = load_model(model_name, resnet, optimizer)
    print_step = 100
    logger(">> Training <<<<<<<<", f)
    resnet.train()
    for e in range(epochs):
        epoch = e+1+load_epoch
        epoch_wgt, n_trained= 0., 0
        logger('>> Epoch %d <<<<<<<<'%(epoch), f)

        # Run training
        resnet.train()
        now = time.time()
        for i, data in enumerate(train_loader):
            if i >= 1: 
                break 
            X, m0, iphi, ieta = data['X_jet'].cuda(), data['jetM'].cuda(), data['iphi'].cuda(), data['ieta'].cuda()
            optimizer.zero_grad()

#             print('X shape', X.shape)
            logits = resnet([X, iphi, ieta])
            loss = mae_loss_wgtd(logits, m0, is_cuda=False)
            loss.backward()
            optimizer.step()
            epoch_wgt += len(m0) 
            n_trained += 1

            if i % print_step == 0:
                logits, m0 = inv_transform(logits), inv_transform(m0)
                mae = (logits-m0).abs().mean()
                logger('%d: (%d/%d) m_pred: %s'%(epoch, i, len(train_loader), str(np.squeeze(logits.tolist()[:5]))), f)
                logger('%d: (%d/%d) m_true: %s'%(epoch, i, len(train_loader), str(np.squeeze(m0.tolist()[:5]))), f)
                logger('%d: (%d/%d) Train loss: %f, mae: %f'%(epoch, i, len(train_loader), loss.item(), mae.item()), f)

        now = time.time() - now
        logits, m0 = inv_transform(logits), inv_transform(m0)
        mae = (logits-m0).abs().mean()
        logger('%d: Train time: %.2fs in %d steps for N:%d, wgt: %.f'%(epoch, now, len(train_loader), n_trained, epoch_wgt), f)
        logger('%d: Train loss: %f, mae: %f'%(epoch, loss.item(), mae.item()), f)

        # Run Validation
        resnet.eval()
        _ = do_eval(resnet, val_loader, epoch, True)

    if run_logger:
        f.close()
