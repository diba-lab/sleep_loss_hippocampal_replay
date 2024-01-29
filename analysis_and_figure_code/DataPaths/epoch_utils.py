import numpy as np
from scipy import stats
import scipy.signal as sg
import pingouin as pg
import pandas as pd
from neuropy import core
from scipy import interpolate


def get_wake_epochs(brainstates: core.Epoch):
    wake = brainstates["AW"] + brainstates["QW"]
    wake.set_labels("WK")
    return wake.merge_neighbors()


def plot_hypnogram():
    pass
