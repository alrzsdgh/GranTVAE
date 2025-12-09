import TCDF
import pandas as pd
import numpy as np

def runTCDF(datafile):
    """Loops through all variables in a dataset and return the discovered causes, time delays, losses, attention scores and variable names."""
    df_data = pd.read_csv(datafile)

    allcauses = dict()
    alldelays = dict()
    allreallosses=dict()
    allscores=dict()

    columns = list(df_data)
    for c in columns:
        idx = df_data.columns.get_loc(c)
        causes, causeswithdelay, realloss, scores = TCDF.findcauses(c, cuda='cuda', epochs=1000, 
        kernel_size=4, layers=1, log_interval=500, 
        lr=0.01, optimizername='Adam',
        seed=1111, dilation_c=4, significance=0.8, file=datafile)

        allscores[idx]=scores
        allcauses[idx]=causes
        alldelays.update(causeswithdelay)
        allreallosses[idx]=realloss
        
    effect_cause = np.array(list(alldelays.keys()))
    dim = df_data.shape[-1]
    gc = np.zeros([dim, dim])
    for i in effect_cause:
        gc[i[1],i[0]] = 1.0
    return gc
