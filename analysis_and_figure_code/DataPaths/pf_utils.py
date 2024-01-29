import numpy as np
from scipy import stats
import scipy.signal as sg
import pingouin as pg
import pandas as pd
from neuropy.analyses.placefields import Pf1D
from neuropy import core
from scipy import interpolate


def pf_position_normalized(
    neurons: core.Neurons, position: core.Position, nbins=50, frate_thresh=0
):
    pfmaze = Pf1D(neurons, position=position, frate_thresh=frate_thresh)
    maze_nbins = pfmaze.tuning_curves.shape[1]
    maze_x = np.linspace(0, 1, maze_nbins)
    fmaze = interpolate.interp1d(maze_x, pfmaze.tuning_curves)
    xnorm = np.linspace(0, 1, nbins)
    tc_maze = fmaze(xnorm)

    return tc_maze
