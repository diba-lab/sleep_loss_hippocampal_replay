# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 23:46:13 2019

@author: Bapun
"""

import numpy as np
import matplotlib.pyplot as plt
from figCust import plotfig

x = np.linspace(0,10,100)
y = np.sin(x)

plotfig()
plt.subplot(121)
plt.plot(x,y)