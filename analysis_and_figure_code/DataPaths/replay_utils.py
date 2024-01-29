import numpy as np
import cupy as cp
from scipy import stats
import scipy.signal as sg

def get_jump_distance(arrs, estimator="max"):

    n_pos = arrs[0].shape[0]
    # norm_pos = np.linspace(0, 1, n_pos)
    dpos = 1 / n_pos

    match estimator:
        case 'mean': 
            return np.abs([np.mean(np.diff(np.argmax(_, axis=0)) * dpos) for _ in arrs])
        case 'median': 
            return np.abs([np.median(np.diff(np.argmax(_, axis=0)) * dpos) for _ in arrs])
        case 'max': 
            return np.abs([np.max(np.diff(np.argmax(_, axis=0)) * dpos) for _ in arrs])


def radon_transform_gpu(arr, nlines=10000, dt=1, dx=1, neighbours=1):

    arr = cp.asarray(arr)
    t = cp.arange(arr.shape[1])
    nt = len(t)
    tmid = (nt + 1) / 2 - 1

    pos = cp.arange(arr.shape[0])
    npos = len(pos)
    pmid = (npos + 1) / 2 - 1

    # using convolution to sum neighbours
    arr = cp.apply_along_axis(
        cp.convolve, axis=0, arr=arr, v=cp.ones(2 * neighbours + 1), mode="same"
    )

    # exclude stationary events by choosing phi little below 90 degree
    # NOTE: angle of line is given by (90-phi), refer Kloosterman 2012
    phi = cp.random.uniform(low=-np.pi / 2, high=np.pi / 2, size=nlines)
    diag_len = cp.sqrt((nt - 1) ** 2 + (npos - 1) ** 2)
    rho = cp.random.uniform(low=-diag_len / 2, high=diag_len / 2, size=nlines)

    rho_mat = cp.tile(rho, (nt, 1)).T
    phi_mat = cp.tile(phi, (nt, 1)).T
    t_mat = cp.tile(t, (nlines, 1))
    posterior = cp.zeros((nlines, nt))

    y_line = ((rho_mat - (t_mat - tmid) * cp.cos(phi_mat)) / cp.sin(phi_mat)) + pmid
    y_line = cp.rint(y_line).astype("int")

    # if line falls outside of array in a given bin, replace that with median posterior value of that bin across all positions
    t_out = cp.where((y_line < 0) | (y_line > npos - 1))
    t_in = cp.where((y_line >= 0) & (y_line <= npos - 1))
    posterior[t_out] = cp.median(arr[:, t_out[1]], axis=0)
    posterior[t_in] = arr[y_line[t_in], t_in[1]]

    # old_settings = np.seterr(all="ignore")
    posterior_mean = cp.nanmean(posterior, axis=1)

    best_line = cp.argmax(posterior_mean)
    score = posterior_mean[best_line]
    best_phi, best_rho = phi[best_line], rho[best_line]
    time_mid, pos_mid = nt * dt / 2, npos * dx / 2

    velocity = dx / (dt * cp.tan(best_phi))
    intercept = (
        (dx * time_mid) / (dt * cp.tan(best_phi))
        + (best_rho / cp.sin(best_phi)) * dx
        + pos_mid
    )
    # np.seterr(**old_settings)

    return score, -velocity, intercept


def is_continuous_posterior(posteriors, jump_thresh=40, time_thresh=60):
    """Classifies continuous versus discontinuous replay events based on the following criteria:
    1) Replay lasts at least 60 msec (3 - 20 msec time bins) during which...
    2) Maximum jump distance between adjacent time bins is 40cm.

    Jump distance is calculated as the distance between position bins with the peak posterior probability
    after decoding.

    :param: posteriors: list of posteriors for each replay, shape of each is npos_bins x ntime_bins. (npos_bins stays
    the same for all posteriors and should equal the length of the track in centimeters, ntime_bins varies for each and
    should match the duration of the PBE or SWR used to calculate the replay event.
    :param: jump_thresh: max jump distance in centimeters to be considered continuous
    :param: time_thresh: minimum duration in msec to be considered a replay. Should be in multiples
            of 20 msec (time bin size)."""

    assert np.mod(time_thresh, 20) == 0, "time_thresh must be in multiples of 20 msec"
    nbin_thresh = int(time_thresh/20)  # convert time in ms to # bins

    stacked_posteriors = np.hstack(posteriors)  # Stack up all posteriors horizontally
    n_bins = [_.shape[1] for _ in posteriors]  # get n time bins in each posterior
    cum_bins = np.cumsum(n_bins)[:-1] # identify ends of each event
    dist = np.abs(np.diff(np.argmax(stacked_posteriors, axis=0))) * 2.0  # get distance between each time bin
    dist[cum_bins - 1] = np.nan  # set jump distance between different replays to nan

    # Apply jump distance criteria
    dist = (dist < jump_thresh).astype("int")  # thresholds jump distance
    dist = np.pad(dist, (1,), "constant", constant_values=(0,))  # Pad a 0 to beginning and end

    # Apply time threshold (msec) criteria
    cont_bins = np.hsplit(
        (np.convolve(dist, np.ones(nbin_thresh), mode="same") >= nbin_thresh).astype("int"), cum_bins
    )

    cont_bool = np.array([1 in _ for _ in cont_bins])  # Make into nice array
    
    return cont_bool

def is_good_posterior(posteriors,zsc_thresh=2,frac=0.5):
    n_good_bins = lambda arr:(stats.zscore(arr,axis=None).max(axis=0)>=zsc_thresh).sum()
    good_bool = np.array([n_good_bins(arr)/arr.shape[1] for arr in posteriors])
    return good_bool >= frac

def get_max_length(arr):
    max_loc = np.argmax(arr, axis=0)
    dist = np.abs(np.diff(max_loc))
    dist_logical = np.where(dist < 40, 1, 0)
    pad_dist = np.pad(dist_logical, (1, 1), "constant", constant_values=(0, 0))
    peaks_dict = sg.find_peaks(pad_dist, height=1, width=1, plateau_size=1)[1]
    lengths = peaks_dict["plateau_sizes"] + 1
    
    try:
        max_length = np.max(lengths)
    except:
        max_length = 0

    return max_length



def get_distance(arr):
    max_loc = np.argmax(arr,axis=0)
    return max_loc[-1] - max_loc[0]

def get_continuous_distance(posteriors):
    stacked_posteriors = np.hstack(posteriors)
    n_bins = [_.shape[1] for _ in posteriors]
    cum_bins = np.cumsum(n_bins)[:-1]
    dist = np.abs(np.diff(np.argmax(stacked_posteriors, axis=0))) * 2.0
    


    dist[cum_bins - 1] = np.nan
    dist = (dist < 40).astype("int")
    dist = np.pad(dist, (1,), "constant", constant_values=(0,))
    cont_bins = np.hsplit(
        (np.convolve(dist, np.ones(3), mode="same") >= 3).astype("int"), cum_bins
    )

    cont_bool = np.array([1 in _ for _ in cont_bins])
    
    return cont_bool

