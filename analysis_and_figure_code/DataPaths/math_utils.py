import numpy as np
from scipy import stats
import pandas as pd
from joblib import Parallel, delayed
import functools


def schmidt_trigger_threshold(arr, low_thresh, high_thresh):
    states = np.zeros_like(arr)
    first_low = np.where(arr <= low_thresh)[0][0]
    first_high = np.where(arr >= high_thresh)[0][0]
    first_ind = np.min([first_low, first_high])
    states[:first_ind] = np.argmin([first_low, first_high])
    current_state = states[first_ind - 1]

    for i in range(first_ind, len(arr)):
        if arr[i] >= high_thresh:
            current_state = 1
        if arr[i] <= low_thresh:
            current_state = 0

        states[i] = current_state

    return states
