#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 12:31:15 2019

@author: bapung
"""

import numpy as np
import matplotlib.pyplot as plt
#from scipy.ndimage import gaussian_filter
#import scipy.signal as sg
#import scipy.stats as stats
#from scipy.signal import hilbert
import h5py


sourceDir = '/data/DataGen/wake_new/'

arrays = {}
f= h5py.File(sourceDir + 'wake-basics.mat', 'r') 
for k, v in f.items():
    arrays[k] = np.array(v)

#spks = {}
fspikes= h5py.File(sourceDir + 'testVersion.mat', 'r') 
fbehav= h5py.File(sourceDir + 'wake-behavior.mat', 'r') 
fpos= h5py.File(sourceDir + 'wake-position.mat', 'r') 
fICAStrength = h5py.File('/data/DataGen/ICAStrengthpy.mat', 'r') 

subjects = arrays['basics']
#spikes = spks['spikes']
plt.clf()
for sub in range(0,7):
    sub_name = subjects[sub]
    print(sub_name)
    
    nUnits = len(fspikes['spikes'][sub_name]['time'])
    celltype={}
    quality={}
    stability={}
    for i in range(0,nUnits):
        celltype[i] = fspikes[fspikes['spikes'][sub_name]['time'][i,0]].value
        quality[i] = fspikes[fspikes['spikes'][sub_name]['quality'][i,0]].value
        stability[i] = fspikes[fspikes['spikes'][sub_name]['StablePrePost'][i,0]].value
    
    
    pyrid = [i for i in range(0,nUnits) if quality[i] < 4 and stability[i] == 1]
    cellpyr= [celltype[a] for a in pyrid]
    
    behav = np.transpose(fbehav['behavior'][sub_name]['time'][:])
    states = np.transpose(fbehav['behavior'][sub_name]['list'][:])
    
    NCells = len(pyrid) 
    binS = behav.reshape(np.size(behav),1).squeeze() 
    cellSpkCounts = np.zeros((NCells,len(behav)))
    for cell in range(len(pyrid)):
        spkt = cellpyr[cell].squeeze()
        
        hist1, edges = np.histogram(spkt,binS)
        cellSpkCounts[cell,:] = hist1[::2] 
        
    behavTime = (np.diff(binS)[::2])/1e6
    
    if len(behav)>3:
        cellSpkCounts[:,1] = cellSpkCounts[:,1]+cellSpkCounts[:,2]
        behavTime[1]= behavTime[1]+behavTime[2]
        behavTime = np.delete(behavTime,2)
        cellSpkCounts = np.delete(cellSpkCounts,2,1)
    
    fRate = cellSpkCounts/behavTime
    
    
    plt.subplot(3,3,sub+1)
    plt.boxplot(fRate, labels = ['PRE','MAZE','POST'])
    plt.ylabel('Firing rate')
    plt.title(sub_name)
    
#plt.savefig('FRate.pdf')    