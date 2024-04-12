#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 11:15:28 2019

@author: bapung
"""

import numpy as np
import pandas as pd
#from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
#from scipy.ndimage import gaussian_filter
from OsCheck import DataDirPath, figDirPath
#import scipy.signal as sg
#import scipy.stats as stats
#from scipy.signal import hilbert
import h5py
import seaborn as sns
#sns.set(style="darkgrid")


#plt.style.use('seaborn')
        

sourceDir = DataDirPath() + 'wake_new/'

arrays = {}
f= h5py.File(sourceDir + 'wake-basics.mat', 'r') 
for k, v in f.items():
    arrays[k] = np.array(v)

#spks = {}
fspikes= h5py.File(sourceDir + 'testVersion.mat', 'r') 
fbehav= h5py.File(sourceDir + 'wake-behavior.mat', 'r') 
fpos= h5py.File(sourceDir + 'wake-position.mat', 'r') 
fspeed= h5py.File(sourceDir + 'wake-speed.mat', 'r') 
#fICAStrength = h5py.File('/data/DataGen/ICAStrengthpy.mat', 'r') 

subjects = arrays['basics']
#spikes = spks['spikes']

#figFilename = figDirPath() +'PlaceCells.pdf'
#pdf = PdfPages(figFilename)

plt.clf()
for sub in [1]:
    sub_name = subjects[sub]
    print(sub_name)
    
    nUnits = len(fspikes['spikes'][sub_name]['time'])
    celltype, quality, stability, shankID={}, {}, {}, []
    for i in range(0,nUnits):
        celltype[i] = fspikes[fspikes['spikes'][sub_name]['time'][i,0]].value
        quality[i] = fspikes[fspikes['spikes'][sub_name]['quality'][i,0]].value
        stability[i] = fspikes[fspikes['spikes'][sub_name]['StablePrePost'][i,0]].value
        shankID.append(((fspikes[fspikes['spikes'][sub_name]['id'][i,0]].value).squeeze())[0])
    
    
    pyrid = [i for i in range(0,nUnits) if quality[i] < 4 and stability[i] == 1]
    nPyr = len(pyrid)
    cellpyr= [celltype[a] for a in pyrid]
    shankID =[shankID[a] for a in pyrid]
    
    # ====Selecting only pairs from different shanks ========= 
    diff_tetMat = np.asmatrix(shankID).T - np.asmatrix(shankID)
    diff_tetMat = np.tril(diff_tetMat, k=-1)
    
    
    
    behav = np.transpose(fbehav['behavior'][sub_name]['time'][:])
    states = np.transpose(fbehav['behavior'][sub_name]['list'][:])
    
    if sub==0:
        rem_post = states[(states[:,0] > behav[3,0]) & (states[:,2]==2) & (states[:,1]-states[:,0] > 250e3),0:2]
        Bins = np.arange(behav[1,0], behav[2,1], 250e3)        
    else:
        rem_post = states[(states[:,0] > behav[2,0]) & (states[:,2]==2) & (states[:,1]-states[:,0] > 250e3),0:2]
        Bins = np.arange(behav[1,0], behav[1,1], 250e3)
        
            
    
        
    spkCnt = [np.histogram(cellpyr[x], bins = Bins)[0] for x in range(0,len(cellpyr))]
    corr_mat = np.corrcoef(spkCnt)
    corr_mat = corr_mat[diff_tetMat!=0]
    
    # ===== Making quintiles from the maze correlations ==========
    sort_corr = np.argsort(corr_mat)
    corr_ind = [x for x in range(0,len(sort_corr))]
    n = int(np.ceil(len(sort_corr)/5))
    quintile_ind = [sort_corr[i * n:(i + 1) * n] for i in range((len(sort_corr) + n - 1) // n )]
    print(len(quintile_ind))

    
    

    # ====== correlations over POST ===========
    mean_remcorr = []
    allrem_corr = []
    q1, q2, q3, q4, q5 = [], [], [], [], []
    for ep_ind, epoch_strt in enumerate(np.arange(behav[2,0], behav[2,1], 40e6)):
        
        bin_ep = np.arange(epoch_strt, epoch_strt+60e6, 250e3)    
        spkCnt_ep = [np.histogram(cellpyr[x], bins = bin_ep)[0] for x in range(0,len(cellpyr))]
        corr_ep = np.corrcoef(spkCnt_ep)
        corr_ep = corr_ep[diff_tetMat!=0] # Pairs from different shanks
        
        q1.append(np.nanmean(corr_ep[quintile_ind[0]]))
        q2.append(np.nanmean(corr_ep[quintile_ind[1]]))
        q3.append(np.nanmean(corr_ep[quintile_ind[2]]))
        q4.append(np.nanmean(corr_ep[quintile_ind[3]]))
        q5.append(np.nanmean(corr_ep[quintile_ind[4]]))

#        mean_remcorr.append(np.nanmean(rem_corr))
#        allrem_corr.extend(rem_corr)
        
    # ====creating dataframe for correlations during REM ========
    t = (np.arange(behav[2,0], behav[2,1], 40e6)-behav[2,0])/3600e6
    df = pd.DataFrame({'time': t ,'q1' : q1, 'q2': q2, 'q3' : q3, 'q4': q4, 'q5' : q5})
#    df.dropna() # ignoring nan values
    
    

        
        
    fig = plt.subplot(7,1,sub+1)

    df.plot(x = 'time', y = ['q1','q2','q3','q4','q5'], ax = fig)
        
        
#    plt.xticks([])
#    plt.yticks([])

    plt.title(sub_name)
#    plt.tight_layout()
#    pdf.savefig(dpi=300)

#pdf.close()    
    