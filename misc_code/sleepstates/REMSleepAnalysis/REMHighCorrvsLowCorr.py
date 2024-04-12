#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 10:00:47 2019

@author: bapung
"""


import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from OsCheck import DataDirPath, figDirPath
#import scipy.signal as sg
#import scipy.stats as stats
#from scipy.signal import hilbert
import h5py
import seaborn as sns
sns.set(style="darkgrid")


plt.style.use('seaborn')
        

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
for sub in [0,1,2,3,4,5,6]:
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
    
    # Thresholding the maze correlations
    Thresh_corr = 0.1
    high_corr = np.where(corr_mat >= Thresh_corr)[0]
    low_corr = np.where(corr_mat < Thresh_corr)[0]
    corr_thresh = np.where(corr_mat >= Thresh_corr, 'high', 'low').tolist()
    
    
#    np.fill_diagonal(corr_mat,0)
    
    mean_remcorr = []
    allrem_corr = []
    high_remcorr, low_remcorr, rem_id = [], [], []
    for rem_epoch in range(0,len(rem_post)):
        bin_rem = np.arange(rem_post[rem_epoch,0], rem_post[rem_epoch,1], 250e3)    
        spkCnt_rem = [np.histogram(cellpyr[x], bins = bin_rem)[0] for x in range(0,len(cellpyr))]
        rem_corr = np.corrcoef(spkCnt_rem)
#        rem_corr = rem_corr[np.tril_indices(len(cellpyr), -1)]
        rem_corr = rem_corr[diff_tetMat!=0] # Pairs from different shanks
        
        high_remcorr.append((rem_corr[high_corr]))
        low_remcorr.append((rem_corr[low_corr]))
        rem_id.extend(len(rem_corr)*[rem_epoch+1])

        mean_remcorr.append(np.nanmean(rem_corr))
        allrem_corr.extend(rem_corr)
        
    # ====creating dataframe for correlations during REM ========
    
    df = pd.DataFrame({'corr' : allrem_corr, 'thresh': corr_thresh*len(rem_post), 'remid' : rem_id})
    df.dropna() # ignoring nan values
    
    
    
    
    
    
    
    # set width of bar
    barWidth = 0.25
    # Set position of bar on X axis
    r1 = np.arange(len(high_remcorr))
    r2 = [x + barWidth for x in r1]

 
    allrem_corr_corr = (pd.DataFrame(allrem_corr).T).corr()
    np.fill_diagonal(allrem_corr_corr.values,0)
        
        
    plt.subplot(3,3,sub+1)
#    plt.bar(np.arange(1,len(rem_post)+1,1),height=mean_remcorr)
#    plt.imshow(allrem_corr_corr)
#    plt.bar(r1, high_remcorr, color='#7f6d5f', width=barWidth, edgecolor='white', label='high')
#    plt.bar(r2, low_remcorr, color='#557f2d', width=barWidth, edgecolor='white', label='low')   
    sns.boxplot(x= 'remid', y = 'corr', hue = 'thresh',data = df, palette="Set2", fliersize= 0)
        
        
#    plt.xticks([])
#    plt.yticks([])

    plt.title(sub_name)
#    plt.tight_layout()
#    pdf.savefig(dpi=300)

#pdf.close()    
    