import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, recall_score
from CR_VAE.models.cgru_error import arrange_input
from torchmetrics.regression import RelativeSquaredError


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def data_loader(inp_dir:str):
    """loading the input time series.
    Args:
        inp_dir: the directory to the time series file
    Returns:
        x: the numpy array of shape [total_signal_length, n_channels]
    """
    if inp_dir.endswith('.npy'):
        x = np.load(inp_dir).T
    if inp_dir.endswith('.csv'):
        x = pd.read_csv(inp_dir).values
    return x

def train_test_split(inp_arr, inp_split_ratio=0.8):
    """loading the input time series.
    Args:
        inp_arr: input time series signal, a numpy array of shape [total_signal_length, n_channels]
    Returns:
        x_train: the numpy array of shape [train_signal_length, n_channels]
        x_test: the numpy array of shape [test_signal_length, n_channels]
    """
    x_train = inp_arr[:int(inp_arr.shape[0]*inp_split_ratio),:]
    x_test = inp_arr[int(inp_arr.shape[0]*inp_split_ratio):,:]
    return x_train, x_test

def data_processor(inp_np_array, inp_sec_length=20):
    """preprocessing the input time series.
    Args:
        inp_np_array: the input numpy array with shape [total_signal_length, n_channels]
        inp_sec_length: the length for the overlapping short sequences
    Returns:
        X_all: a sectioned torch tensor of shape [n_samples, section_length, n_channels]
    """
    X_ = torch.tensor(inp_np_array[np.newaxis], dtype=torch.float32, device=device)
    X, Y = zip(*[arrange_input(x, inp_sec_length) for x in X_])
    X_all = torch.cat(X, dim=0)
    Y_all = torch.cat(Y, dim=0)
    return X_all




def gt_causal_graph(inp_graph_file, inp_ts_file):
    '''
    Args:
        inp_graph_file: directory to the causal graph csv file (first col: parent, second col: children)
        inp_ts_file: directory to the time series csv file (first col: parent, second col: children)
    Returns:
        gc = a numpy array of shape [p,p] where p is the number of the channels
    '''
    dim = pd.read_csv(inp_ts_file).shape[-1]
    gc = np.zeros([dim,dim])
    
    idx = pd.read_csv(inp_graph_file, header=None).values[:,:2]
    for i in idx:
        gc[i[1],i[0]] = 1.0
    return gc
    


def rrse(inp_x, inp_pred):
    relative_squared_error = RelativeSquaredError().to('cuda')
    rrse = torch.stack([relative_squared_error(torch.squeeze(inp_x)[:,i], inp_pred[:,i]) for i in range(inp_pred.shape[-1])])
    return torch.sqrt(torch.mean(rrse)).detach().to('cpu').numpy()


def MMD(x, y, kernel="multiscale"):
    # ref: https://www.kaggle.com/code/onurtunali/maximum-mean-discrepancy?scriptVersionId=29651280&cellId=4
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.
    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    Returns:
        a single value as MMD
    """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    dxx = rx.t() + rx - 2. * xx # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz # Used for C in (1)
    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))
    if kernel == "multiscale":   
        bandwidth_range = [0.01, 0.1, 1, 10, 100]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1         
    if kernel == "rbf":  
        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)
    return torch.mean(XX + YY - 2. * XY)
    
    
    
    
    
    
def evaluate_auc_tpr(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Compute AUC and TPR between two binary adjacency matrices.
    Args:
        y_true (np.ndarray): Ground-truth binary matrix
        y_pred (np.ndarray): Predicted binary matrix

    Returns:
        AUC, TPR
    """
    y_true_flat = y_true.ravel()
    y_pred_flat = y_pred.ravel()
    # AUC
    auc = roc_auc_score(y_true_flat, y_pred_flat)
    # True Positive Rate (Recall)
    tpr = recall_score(y_true_flat, y_pred_flat, zero_division=0)
    return auc, tpr