#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 11:56:12 2019

@author: bapung
"""


import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from OsCheck import DataDirPath, figDirPath
#import scipy.signal as sg
#import scipy.stats as stats
#from scipy.signal import hilbert
import h5py
        

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

figFilename = figDirPath() +'PlaceCells.pdf'
pdf = PdfPages(figFilename)

for sub in [6]:
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
    nPyr = len(pyrid)
    cellpyr= [celltype[a] for a in pyrid]
    
    behav = np.transpose(fbehav['behavior'][sub_name]['time'][:])
    states = np.transpose(fbehav['behavior'][sub_name]['list'][:])
    
    
    plt.clf()
    for cell in range(len(pyrid)):
        spkt = cellpyr[cell].squeeze()


#    ICA_strength = np.array(np.transpose(fICAStrength['ActStrength']['subjects'][sub_name]['wake'][:]))
        
        nRows = np.ceil(np.sqrt(nPyr))
        nCols = np.ceil(np.sqrt(nPyr))
    
        
    #    plt.plot(posx_mz,posy_mz,'.')
        plt.subplot(nRows,nCols,cell+1)
        plt.imshow(pfRate_smooth)
        
        plt.xticks([])
        plt.yticks([])
#        plt.subplot(1,2,2)
#        plt.imshow(pfRate_smooth)
    plt.suptitle(sub_name)
    plt.tight_layout()
    pdf.savefig(dpi=300)

pdf.close()    
    