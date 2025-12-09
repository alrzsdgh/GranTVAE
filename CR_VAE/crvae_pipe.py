# -*- coding: utf-8 -*-

"""
Created on Sat Aug  6 20:00:04 2022

@author: 61995
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.io

from CR_VAE.models.cgru_error import CRVAE, VRAE4E, train_phase1, train_phase2
from utils import data_loader



device = torch.device('cuda')

def crvae_training(data_name, inp_x, inp_graph):
    '''
    Args:
        inp_X: the training time series, a numpy array of shape [train_signal_length, n_channels]
    Returns:
        model: the developed model
        GC_est: the discovered causal graph by the model
    '''
    inp_x = torch.tensor(inp_x[np.newaxis], dtype=torch.float32, device=device)
    full_connect = np.ones(inp_graph.shape)
    cgru = CRVAE(inp_x.shape[-1], full_connect, hidden=64).cuda(device=device)
    vrae = VRAE4E(inp_x.shape[-1], hidden=64).cuda(device=device)
    print(f'Training CR-VAE on {data_name} phase 1 ...')
    train_loss_list = train_phase1(
        cgru, inp_x, context=20, lam=0.1, lam_ridge=0, lr=5e-2, max_iter=1000,
        check_every=50, verbose=0)#0.1
    GC_est = cgru.GC().cpu().data.numpy()
    cgru = CRVAE(inp_x.shape[-1], GC_est, hidden=64).cuda(device=device)
    vrae = VRAE4E(inp_x.shape[-1], hidden=64).cuda(device=device)
    print(f'Training CR-VAE on {data_name} phase 2 ...')
    model, train_loss_list = train_phase2(
        cgru, vrae, inp_x, context=20, lam=0., lam_ridge=0, lr=5e-2, max_iter=10000,
        check_every=50, verbose=0)
    return model, GC_est

def crvae_prediction(inp_model, inp_x):
    '''
    Args:
        inp_model: the trained crvae model
        inp_X: the testing time series, a numpy array of shape [test_signal_length, n_channels]
    Returns:
        pred: the predicted time step for each test sample
    '''
    pred = inp_model(inp_x, mode='test')
    pred = torch.squeeze(pred[:,-1,:])
    return pred
    

