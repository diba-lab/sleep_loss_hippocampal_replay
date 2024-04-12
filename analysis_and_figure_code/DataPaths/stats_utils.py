import numpy as np
from scipy import stats
import pandas as pd
from joblib import Parallel, delayed
import functools
from tqdm import tqdm


def bootstrap(pairs, data: pd.DataFrame, x, y, hue, hue_order, n_boot=10000, **kwargs):
    rng = np.random.default_rng()
    subsample_mean = lambda arr: np.mean(rng.choice(arr, size=len(arr), replace=True))

    n_pairs = len(pairs)
    for i, p in enumerate(pairs):
        print(p)
        g1 = data[(data[x] == p[0][0]) & (data[hue] == p[0][1])][y].values
        g2 = data[(data[x] == p[1][0]) & (data[hue] == p[1][1])][y].values
        g1_means = np.array([subsample_mean(g1) for _ in range(n_boot)])
        g2_means = np.array([subsample_mean(g2) for _ in range(n_boot)])

        p_val = np.count_nonzero(g1_means >= g2_means) / n_boot
        print(f"{p[0][0]}_{p[0][1]} vs {p[1][0]}_{p[1][1]}: {p_val}")


def resample(df: pd.DataFrame, level="both", apply=None):
    """Resamples variables by sampling with replacement at hierarichy levels: 'session' and 'associated variables'. The associated variables are columns other than the 'session' columns. They could be pairwise correlations, ripple frequencies etc.

    session grp var1 var2 var3 ..........
    0       SD  0.2   0.3  0.4
    0       SD  0.5   0.3  0.4
    1       SD  0.2   0.73  0.2
    1       SD  0.2   0.73  0.2
    2       SD  0.1   0.2  0.23
    2       SD  0.1   0.2  0.23
    5       NSD  0.2   0.3  0.4
    5       NSD  0.2   0.3  0.4
    6       NSD  0.5   0.3  0.4
    6       NSD  0.2   0.73  0.2
    6       NSD  0.1   0.2  0.23

    Parameters
    ----------
    df : pd.DataFrame
        columns should specify session and variables
    level : str, optional
        _description_, by default "both"
    apply : callable/function, optional
        a function to operate on resampled dataframe, by default None

    Returns
    -------
    _type_
        _description_
    """
    sess_ids = df["session"].unique()
    n_sess = len(sess_ids)
    # print(sess_ids)
    if level in {"session", "both"}:
        # bootstrap session_ids
        rng = np.random.default_rng()
        sess_ids = rng.choice(sess_ids, size=n_sess, replace=True)

    new_df = []
    for i, idx in enumerate(sess_ids):
        idx_df = df[df.session == idx].copy()  # df of variables for that session
        idx_df.loc[:, "session"] = i  # make selected session independent

        if level in {"both", "samples"}:
            # bootstrap second level
            if "zt" in idx_df.columns:
                idx_df = (
                    idx_df.groupby(["zt"], sort=False)
                    .apply(pd.DataFrame.sample, frac=1, replace=True, ignore_index=True)
                    .reset_index(drop=True)
                )
            else:
                idx_df = idx_df.sample(frac=1, replace=True, ignore_index=True)

        new_df.append(idx_df)
    new_df = pd.concat(new_df, ignore_index=True)

    if apply is not None:
        assert callable(apply), "apply can only be a function"
        new_df = apply(new_df)

        # NRK add that didn't work but might need to be implemented in the future.
        # if redo_nans:  # check if nans result and recursively call resample if it does
        #     if np.any(np.isnan(new_df.mean(numeric_only=True))):
        #         new_df = resample(df, level=level, apply=apply, redo_nans=redo_nans)

    return new_df


def bootstrap_resample(df: pd.DataFrame, n_iter, n_jobs=1, apply=None, level="both"):
    groups = df["grp"].unique()

    partial_resample = functools.partial(resample, level=level, apply=apply)
    out_df = []
    for grp in groups:
        print(f"Running bootstraps for {grp} group")
        df_grp = df[df["grp"] == grp]
        data = [
            r
            for r in tqdm(
                Parallel(n_jobs=n_jobs, return_as="generator")(
                    delayed(partial_resample)(df_grp) for _ in range(n_iter)
                ),
                total=n_iter,
                # position=1,
                # leave=False,
            )
        ]
        data = pd.concat(data, ignore_index=True)
        data["grp"] = grp
        out_df.append(data)

    return pd.concat(out_df, ignore_index=True)


def get_bootstrap_prob(sample1, sample2):
    """
    get_direct_prob Returns the direct probability of items from sample2 being
    greater than or equal to those from sample1.
       Sample1 and Sample2 are two bootstrapped samples and this function
       directly computes the probability of items from sample 2 being greater
       than or equal to those from sample1. Since the bootstrapped samples are
       themselves posterior distributions, this is a way of computing a
       Bayesian probability. The joint matrix can also be returned to compute
       directly upon.
    obtained from: https://github.com/soberlab/Hierarchical-Bootstrap-Paper/blob/master/Bootstrap%20Paper%20Simulation%20Figure%20Codes.ipynb

    References
    ----------
    Saravanan, Varun, Gordon J Berman, and Samuel J Sober. “Application of the Hierarchical Bootstrap to Multi-Level Data in Neuroscience.” Neurons, Behavior, Data Analysis and Theory 3, no. 5 (2020): https://nbdt.scholasticahq.com/article/13927-application-of-the-hierarchical-bootstrap-to-multi-level-data-in-neuroscience.

    Parameters
    ----------
    sample1: array
        numpy array of values
    sample2: array
        numpy array of values

    Returns
    ---------
    pvalue1
        using joint probability distribution
    pvalue2
        using the number of sample2 being greater than sample1
    2d array
        joint probability matrix
    """

    # assert len(sample1) == len(sample2), "both inputs lengths should match"
    sample1 = np.array(sample1)
    sample2 = np.array(sample2)
    try:
        joint_low_val = min([min(sample1), min(sample2)])
        joint_high_val = max([max(sample1), max(sample2)])

        nbins = 100
        p_axis = np.linspace(joint_low_val, joint_high_val, num=nbins)
        edge_shift = (p_axis[2] - p_axis[1]) / 2
        p_axis_edges = p_axis - edge_shift
        p_axis_edges = np.append(p_axis_edges, (joint_high_val + edge_shift))

        # Calculate probabilities using histcounts for edges.

        p_sample1 = np.histogram(sample1, bins=p_axis_edges)[0] / np.size(sample1)
        p_sample2 = np.histogram(sample2, bins=p_axis_edges)[0] / np.size(sample2)

        # Now, calculate the joint probability matrix:
        # p_joint_matrix = np.zeros((nbins, nbins))
        # for i in np.arange(np.shape(p_joint_matrix)[0]):
        #     for j in np.arange(np.shape(p_joint_matrix)[1]):
        #         p_joint_matrix[i, j] = p_sample1[i] * p_sample2[j]

        p_joint_matrix = p_sample1[:, np.newaxis] * p_sample2[np.newaxis, :]

        # Normalize the joint probability matrix:
        p_joint_matrix = p_joint_matrix / p_joint_matrix.sum()

        # Get the volume of the joint probability matrix in the upper triangle:
        p_test = np.sum(np.triu(p_joint_matrix))

        p_test = 1 - p_test if p_test >= 0.5 else p_test

        statistic = np.abs(sample1.mean() - sample2.mean()) / np.sqrt(
            (sample1.std() ** 2 + sample2.std() ** 2) / 2
        )
        return statistic, p_test
    except ValueError:
        print('Missing data for one input - returning p=1.0 and stat=nan')
        return np.nan, 1.0


def get_bootstrap_prob_paired(arr1, arr2):
    l1, l2 = len(arr1), len(arr2)
    assert l1 == l2, f"len(arr1)={l1},len(arr2)={l2}: both inputs lengths should match"
    arr1 = np.array(arr1)
    arr2 = np.array(arr2)

    delta = arr1 - arr2
    p_test = np.sum(delta >= 0) / len(delta)
    p_test = 1 - p_test if p_test >= 0.5 else p_test

    return np.nanmean(delta), p_test
