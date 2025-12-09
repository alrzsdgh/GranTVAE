# Modules
import os
import torch
import numpy as np
import pandas as pd
import configparser
from utils import MMD, rrse, evaluate_auc_tpr, data_loader, train_test_split, data_processor, gt_causal_graph
from CR_VAE.crvae_pipe import crvae_training, crvae_prediction
from Transformer import TransformerVAE, training_, inference_
from G_transformer import GranTVAE, gratra_training, gratra_inference

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reading config file
CONFIG_PATH = 'configs.ini'
config = configparser.ConfigParser()
config.read(CONFIG_PATH)
dataset_name = config['Configuration']['data_name']
split_ratio = float(config['Configuration']['split_ratio'])
num_repeats = int(config['Configuration']['num_repeats'])
model_name = config['Models']['model_name']





def main():

    # Baselines
    data_dir = f'datasets/{dataset_name}.csv'
    x = data_loader(data_dir)
    if x.shape[0] < 2000:
        train_section, test_section = train_test_split(x,split_ratio)
    else:
        train_section, test_section = train_test_split(x[-2000:],split_ratio)
    n_channels = x.shape[1]
    x_NG = pd.read_csv(data_dir).values
    X_NG = torch.tensor(x_NG[np.newaxis], dtype=torch.float32, device='cuda')

    ctVAE_x_train = data_processor(train_section)[:-1]
    ctVAE_y_train = data_processor(train_section, 21)
    x_test = data_processor(test_section)

    if dataset_name.startswith('Henon'):
        dim = x.shape[-1]
        GC = np.zeros([dim,dim])
        for i in range(dim):
            GC[i,i] = 1
            if i!=0:
                GC[i,i-1] = 1

    if dataset_name.startswith('Lorenz'):
        dim = x.shape[-1]
        GC = np.zeros([dim,dim], dtype=int)
        for i in range(dim):
            GC[i, i] = 1
            GC[i, (i + 1) % dim] = 1
            GC[i, (i - 1) % dim] = 1
            GC[i, (i - 2) % dim] = 1           

    if dataset_name.startswith('VAR'):
        GC = pd.read_csv(f'datasets/VAR_graph.csv').values


    if dataset_name.startswith('fMRI'):
        GC = gt_causal_graph(f'datasets/fMRI_graph.csv', f'datasets/fMRI.csv')



    if model_name == 'TCDF':
        auc_lst, tpr_lst =[], []
        os.chdir('TCDF')
        from TDCF_pipe import runTCDF
        for i in range(num_repeats):
            GC_est = runTCDF(f'../{data_dir}')
            auc_lst.append(evaluate_auc_tpr(GC, GC_est)[0])
            tpr_lst.append(evaluate_auc_tpr(GC, GC_est)[1])
        print('_______________________________')
        print(f'Performance of {model_name} on {dataset_name}')
        print('AUC',f"{np.mean(np.array(auc_lst)):.3f}±{np.std(np.array(auc_lst)):.3f}")
        print('TPR',f"{np.mean(np.array(tpr_lst)):.3f}±{np.std(np.array(tpr_lst)):.3f}")
        print('_______________________________')
        os.chdir('..')



    if model_name == 'CR-VAE':
        rrse_lst, mmd_lst =[], []
        for i in range(num_repeats):
            model, GC_est=crvae_training('henon', train_section, GC)
            prediction = crvae_prediction(model, x_test)
            mmd_lst.append(MMD(torch.Tensor(test_section[20:]).to(device), prediction).detach().to('cpu').numpy())
            rrse_lst.append(rrse(torch.Tensor(test_section[20:]).to(device), prediction))

        print('_______________________________')
        print(f'Performance of {model_name} on {dataset_name}')
        print('MMD:', f"{np.mean(np.array(mmd_lst)):.3f}±{np.std(np.array(mmd_lst)):.3f}")
        print('RRSE:', f"{np.mean(np.array(rrse_lst)):.3f}±{np.std(np.array(rrse_lst)):.3f}")
        print('_______________________________')


    if model_name == 'nonGranTVAE':
        print(f'nonGranTVAE on {dataset_name}')
        tst_rrse_lst, tst_mmd_lst, tr_rrse_lst, tr_mmd_lst =[], [], [], []
        mmd_rel_ch_lst, rrse_rel_ch_lst = [], []
        for i in range(num_repeats): 
            model = TransformerVAE(channels=n_channels).to('cuda')
            model, loss = training_(model, ctVAE_x_train, ctVAE_y_train, dataset_name)
            torch.save(model.state_dict(), f"saved_models/nonGranTVAE_{dataset_name}_{i+1}.pth")

            train_prediction = inference_(model, ctVAE_x_train)
            tr_mmd = MMD(torch.Tensor(train_section[21:]).to(device), train_prediction).detach().to('cpu').numpy()
            tr_rrse = rrse(torch.Tensor(train_section[21:]).to(device), train_prediction)
            tr_mmd_lst.append(tr_mmd)
            tr_rrse_lst.append(tr_rrse)

            test_prediction = inference_(model, x_test)
            tst_mmd = MMD(torch.Tensor(test_section[20:]).to(device), test_prediction).detach().to('cpu').numpy()
            tst_rrse = rrse(torch.Tensor(test_section[20:]).to(device), test_prediction)
            tst_mmd_lst.append(tst_mmd)
            tst_rrse_lst.append(tst_rrse)

            mmd_rel_ch_lst.append((tst_mmd - tr_mmd)*100 / tr_mmd)
            rrse_rel_ch_lst.append((tst_rrse - tr_rrse)*100 / tr_rrse)

        print('_______________________________')
        print(f'Performance of {model_name} on {dataset_name}')
        print('Train MMD:', f"{np.mean(np.array(tr_mmd_lst)):.3f}±{np.std(np.array(tr_mmd_lst)):.3f}")
        print('Train RRSE:', f"{np.mean(np.array(tr_rrse_lst)):.3f}±{np.std(np.array(tr_rrse_lst)):.3f}")
        print('Test MMD:', f"{np.mean(np.array(tst_mmd_lst)):.3f}±{np.std(np.array(tst_mmd_lst)):.3f}")
        print('Test RRSE:', f"{np.mean(np.array(tst_rrse_lst)):.3f}±{np.std(np.array(tst_rrse_lst)):.3f}")
        print('MMD Rel:', f"{np.mean(np.array(mmd_rel_ch_lst)):.3f}±{np.std(np.array(mmd_rel_ch_lst)):.3f}%")
        print('RRSE Rel:', f"{np.mean(np.array(rrse_rel_ch_lst)):.3f}±{np.std(np.array(rrse_rel_ch_lst)):.3f}%")
        print('_______________________________')



    if model_name == 'GranTVAE':
        print(f'GranTVAE on {dataset_name}')
        tst_rrse_lst, tst_mmd_lst, auc_lst, tpr_lst, tr_rrse_lst, tr_mmd_lst =[], [], [], [], [], []
        mmd_rel_ch_lst, rrse_rel_ch_lst = [], []
        for i in range(num_repeats): 
            model = GranTVAE(channels=n_channels).to('cuda')
            model, loss, GC_est = gratra_training(model, ctVAE_x_train, ctVAE_y_train, dataset_name)
            torch.save(model.state_dict(), f"saved_models/GranTVAE_{dataset_name}_{i+1}.pth")

            train_prediction = gratra_inference(model, ctVAE_x_train, GC_est)
            tr_mmd = MMD(torch.Tensor(train_section[21:]).to(device), train_prediction).detach().to('cpu').numpy()
            tr_rrse = rrse(torch.Tensor(train_section[21:]).to(device), train_prediction)
            tr_mmd_lst.append(tr_mmd)
            tr_rrse_lst.append(tr_rrse)

            test_prediction = gratra_inference(model, x_test, GC_est)
            tst_mmd = MMD(torch.Tensor(test_section[20:]).to(device), test_prediction).detach().to('cpu').numpy()
            tst_rrse = rrse(torch.Tensor(test_section[20:]).to(device), test_prediction)
            tst_mmd_lst.append(tst_mmd)
            tst_rrse_lst.append(tst_rrse)

            mmd_rel_ch_lst.append((tst_mmd - tr_mmd)*100 / tr_mmd)
            rrse_rel_ch_lst.append((tst_rrse - tr_rrse)*100 / tr_rrse)

            GC_est = GC_est.detach().to('cpu').numpy()
            auc_lst.append(evaluate_auc_tpr(GC, GC_est)[0])
            tpr_lst.append(evaluate_auc_tpr(GC, GC_est)[1]) 
        print('_______________________________')
        print(f'Performance of {model_name} on {dataset_name}')
        print('Train MMD:', f"{np.mean(np.array(tr_mmd_lst)):.3f}±{np.std(np.array(tr_mmd_lst)):.3f}")
        print('Train RRSE:', f"{np.mean(np.array(tr_rrse_lst)):.3f}±{np.std(np.array(tr_rrse_lst)):.3f}")
        print('Test MMD:', f"{np.mean(np.array(tst_mmd_lst)):.3f}±{np.std(np.array(tst_mmd_lst)):.3f}")
        print('Test RRSE:', f"{np.mean(np.array(tst_rrse_lst)):.3f}±{np.std(np.array(tst_rrse_lst)):.3f}")
        print('MMD Rel:', f"{np.mean(np.array(mmd_rel_ch_lst)):.3f}±{np.std(np.array(mmd_rel_ch_lst)):.3f}%")
        print('RRSE Rel:', f"{np.mean(np.array(rrse_rel_ch_lst)):.3f}±{np.std(np.array(rrse_rel_ch_lst)):.3f}%")
        print('AUC',f"{np.mean(np.array(auc_lst)):.3f}±{np.std(np.array(auc_lst)):.3f}")
        print('TPR',f"{np.mean(np.array(tpr_lst)):.3f}±{np.std(np.array(tpr_lst)):.3f}")
        print('_______________________________')


if __name__ == "__main__":
    main()