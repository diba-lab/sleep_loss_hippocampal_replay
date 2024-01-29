#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 16:34:35 2019

@author: bapung
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
#import scipy.signal as sg
#import scipy.stats as stats
#from scipy.signal import hilbert
import time as time
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
#fICAStrength = h5py.File('/data/DataGen/ICAStrengthpy.mat', 'r') 

subjects = arrays['basics']
#spikes = spks['spikes']

for sub in range(5,6):
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
    
#    spkt = [cellpyr[0],cellpyr[1]]
#    grp = [np.ones(len(cellpyr[0])),np.ones(len(cellpyr[1]))]
    t = time.time()
# do stuff
    
    for cell1 in range(0,1):
        for cell2 in range(1,2):
            spk1 = cellpyr[cell1].squeeze()
            spk2 = cellpyr[cell2].squeeze()
            
            diff1 = (spk1-spk2[:,np.newaxis]).squeeze()
            diff2 = (spk2-spk1[:,np.newaxis]).squeeze()

    elapsed = time.time() - t    