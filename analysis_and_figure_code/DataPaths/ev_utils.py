import numpy as np
from scipy import stats
from neuropy import core
import warnings
import pandas as pd
import pingouin as pg
from scipy.optimize import curve_fit
import subjects

warnings.simplefilter(action="ignore", category=FutureWarning)


def sd_delta_ev(df: pd.DataFrame):
    grps = ["NSD", "SD"]
    columns = df.columns
    epochs = columns[~df.columns.isin(["PRE", "MAZE", "grp", "session"])]

    ev = np.zeros((2, len(epochs)))  # n_grps x n_epochs
    for g, grp in enumerate(grps):
        df_grp = df[df.grp == grp]
        for e, epoch in enumerate(epochs):
            r = pg.partial_corr(data=df_grp, x="MAZE", y=epoch, covar="PRE").r.values
            ev[g, e] = r**2

    nsd_ev, sd_ev = ev[0], ev[1]
    delta_ev = nsd_ev - sd_ev
    return delta_ev, nsd_ev, sd_ev


def get_ev(df: pd.DataFrame, as_df=True):
    """Calculates ev by using pooled pairwise correlation across sessions.

    Parameters
    ----------
    df : pd.DataFrame
        _description_
    as_df : bool
        return the output as dataframe, by default False

    Returns
    -------
    _type_
        _description_
    """
    df = df.drop(columns="session")
    df.columns = df.columns.astype("str")  # necessary otherwise gives error in pingouin
    good_df = df.dropna(axis=1, thresh=3)
    partial_corr = pg.pairwise_corr(
        data=good_df, columns=["MAZE"], covar=["PRE"], nan_policy="pairwise"
    ).rename(columns={"Y": "zt"})

    partial_corr["ev"] = partial_corr["r"] ** 2  # add explained variance column
    ev = partial_corr[["zt", "ev"]]  # keep only necessary coloumns

    return ev


def get_ev_rev(df: pd.DataFrame, as_df=True):
    """Calculates ev by using pooled pairwise correlation across sessions.

    Parameters
    ----------
    df : pd.DataFrame
        _description_
    as_df : bool
        return the output as dataframe, by default False

    Returns
    -------
    _type_
        _description_
    """
    df = df.drop(columns="session")
    df.columns = df.columns.astype("str")  # necessary otherwise gives error in pingouin
    good_df = df.dropna(axis=1, thresh=3)
    # print(f"\n{len(good_df.columns)}")
    ev = pg.pairwise_corr(
        data=good_df, columns=["MAZE"], covar=["PRE"], nan_policy="pairwise"
    ).rename(columns={"Y": "zt"})

    ev["ev"] = ev["r"] ** 2  # add explained variance column

    rev_cols = list(np.setdiff1d(good_df.columns, ["PRE", "MAZE", "grp"]))
    rev = []
    for col in rev_cols:
        col_rev = pg.pairwise_corr(
            data=good_df, columns=["MAZE", "PRE"], covar=col, nan_policy="pairwise"
        )
        col_rev["zt"] = int(col)
        rev.append(col_rev)

    rev = pd.concat(rev, ignore_index=True)

    rev["rev"] = rev["r"] ** 2  # add explained variance column
    rev = rev.sort_values(by="zt")

    ev = ev[["zt", "ev"]]  # keep only necessary coloumns
    ev["zt"] = ev["zt"].values.astype("int")
    rev = rev[["zt", "rev"]]  # keep only necessary coloumns

    return ev.merge(rev, on="zt")


def get_ev_mean(df: pd.DataFrame):
    """Calculates session-wise EV and takes mean across sessions.

    Parameters
    ----------
    df : pd.DataFrame
        _description_
    as_df : bool
        return the output as dataframe, by default False

    Returns
    -------
    _type_
        _description_
    """

    df.columns = df.columns.astype("str")
    sess_ids = df["session"].unique()
    ev_df = []
    for sess_id in sess_ids:
        sess_df = df[df["session"] == sess_id]
        good_sess_df = sess_df.dropna(axis=1, thresh=3)
        # assert all([c in good_sess_df for c in ["PRE", "MAZE"]])
        partial_corr = pg.pairwise_corr(
            data=good_sess_df, columns=["MAZE"], covar=["PRE"], nan_policy="pairwise"
        )
        ev_df.append(partial_corr)

    ev_df = pd.concat(ev_df, ignore_index=True).rename(columns={"Y": "zt"})
    ev_df["ev"] = ev_df["r"] ** 2  # add explained variance column
    ev_df = ev_df[["zt", "ev"]]  # keep only necessary coloumns
    mean_ev_df = ev_df.groupby("zt").mean(numeric_only=True).reset_index()

    try:
        mean_ev_df[["zt"]] = mean_ev_df["zt"].astype("float")
        return mean_ev_df
    except:
        return mean_ev_df


def exp_fit(x, a, b):
    return a * np.exp(-x / b)


def get_exp_time_constant(df: pd.DataFrame):
    """Calculates time constants using explained variance from pooled pairwise correlation.

    Parameters
    ----------
    df : pd.DataFrame
        _description_
    as_df : bool
        return the output as dataframe, by default False

    Returns
    -------
    _type_
        _description_
    """

    ev_df = get_ev(df)

    time = ev_df["zt"].values.astype("float")
    ev = ev_df["ev"].values.astype("float")

    sort_indx = np.argsort(time)
    time = time[sort_indx]
    ev = ev[sort_indx]

    popt, _ = curve_fit(exp_fit, time, ev)

    # ev_autocorr = np.correlate(ev, ev, "full")
    # ev_autocorr = ev_autocorr[len(ev_autocorr) // 2 :]  # right half of autocorr
    # max_val = ev_autocorr[0]
    # half_ind = np.where(ev_autocorr < (max_val / 2))[0][0] + np.argmax(ev)
    # tc = time[np.min([len(time) - 1, half_ind])]

    # linfit = stats.linregress(time, ev)
    # m = linfit.slope
    # c = linfit.intercept
    # yfit = m * time + c
    # half_max_indx = np.where(yfit < (yfit.max() / 2))[0]

    # # if half_max_indx.size==0:
    # #     tc = np.nan
    # # else:

    # tc = m

    return pd.DataFrame({"tc": popt[1]}, index=[0])


def get_linear_time_constant(df: pd.DataFrame):
    """Calculates time constants using explained variance from pooled pairwise correlation.

    Parameters
    ----------
    df : pd.DataFrame
        _description_
    as_df : bool
        return the output as dataframe, by default False

    Returns
    -------
    _type_
        _description_
    """

    ev_df = get_ev(df)

    time = ev_df["zt"].values.astype("float")
    ev = ev_df["ev"].values.astype("float")

    sort_indx = np.argsort(time)
    time = time[sort_indx]
    ev = ev[sort_indx]

    linfit = stats.linregress(time, ev)
    m = linfit.slope
    c = linfit.intercept
    yfit = m * time + c
    # half_max_indx = np.where(yfit < (ev.max() / 2))[0]

    half_max = ev.max() / 2
    half_max_time = (half_max - c) / m

    return pd.DataFrame({"tc": half_max_time}, index=[0])


def get_linear_time_constant_ev_rev(df: pd.DataFrame):
    """Calculates time constants using explained variance from pooled pairwise correlation.

    Parameters
    ----------
    df : pd.DataFrame
        _description_
    as_df : bool
        return the output as dataframe, by default False

    Returns
    -------
    _type_
        _description_
    """

    ev_df = get_ev_rev(df)

    time = ev_df["zt"].values.astype("float")
    ev = ev_df["ev"].values.astype("float")
    rev = ev_df["rev"].values.astype("float")

    #  Sort the time
    sort_indx = np.argsort(time)
    time = time[sort_indx]
    ev = ev[sort_indx]
    rev = rev[sort_indx]

    # keeping only EV values that are above REV
    good_indx = ev > rev
    time = time[good_indx]
    ev = ev[good_indx]
    rev = rev[good_indx]

    linfit = stats.linregress(time, ev)
    m = linfit.slope
    c = linfit.intercept
    # yfit = m * time + c
    # half_max_indx = np.where(yfit < (ev.max() / 2))[0]

    half_max = ev.max() / 2
    half_max_time = (half_max - c) / m

    return pd.DataFrame({"tc": half_max_time}, index=[0])


def get_slope(df: pd.DataFrame):
    """Calculates slope of explained variance from pooled pairwise correlation.

    Parameters
    ----------
    df : pd.DataFrame
        pairwise correlation dataframe

    Returns
    -------
    _type_
        _description_
    """

    ev_df = get_ev(df)

    time = ev_df["zt"].values.astype("float")
    ev = ev_df["ev"].values.astype("float")

    sort_indx = np.argsort(time)
    time = time[sort_indx]
    ev = ev[sort_indx]

    linfit = stats.linregress(time, ev)

    return pd.DataFrame({"slope": linfit.slope}, index=[0])


def get_decay_goodness_of_fit(df: pd.DataFrame):
    """Calculates time constants using explained variance from pooled pairwise correlation.

    Parameters
    ----------
    df : pd.DataFrame
        _description_
    as_df : bool
        return the output as dataframe, by default False

    Returns
    -------
    _type_
        _description_
    """

    ev_df = get_ev(df)

    time = ev_df["zt"].values.astype("float")
    ev = ev_df["ev"].values.astype("float")

    sort_indx = np.argsort(time)
    time = time[sort_indx]
    ev = ev[sort_indx]

    popt, _ = curve_fit(exp_fit, time, ev)
    exp_ev = exp_fit(ev, *popt)

    linfit = stats.linregress(time, ev)
    m = linfit.slope
    c = linfit.intercept
    lin_ev = m * time + c

    goodness_fit = lambda y, yfit: np.sum((y - yfit) ** 2 / yfit)
    lin_chi = goodness_fit(ev, lin_ev)
    exp_chi = goodness_fit(ev, exp_ev)

    return pd.DataFrame(
        {
            "goodness_fit": np.array([lin_chi, exp_chi]),
            "fit": np.array(["linear", "exp"]),
        }
    )


def get_ev_mean_time_constant(df: pd.DataFrame):
    """Calculates time constants using mean explained variances.

    Parameters
    ----------
    df : pd.DataFrame
        _description_
    as_df : bool
        return the output as dataframe, by default False

    Returns
    -------
    _type_
        _description_
    """

    df.columns = df.columns.astype("str")
    sess_ids = df["session"].unique()
    ev_df = []
    for sess_id in sess_ids:
        sess_df = df[df["session"] == sess_id]
        good_sess_df = sess_df.dropna(axis=1, thresh=3)
        # assert all([c in good_sess_df for c in ["PRE", "MAZE"]])
        partial_corr = pg.pairwise_corr(
            data=good_sess_df, columns=["MAZE"], covar=["PRE"], nan_policy="pairwise"
        )
        ev_df.append(partial_corr)

    ev_df = pd.concat(ev_df, ignore_index=True).rename(columns={"Y": "zt"})
    ev_df["ev"] = ev_df["r"] ** 2  # add explained variance column
    ev_df = ev_df[["zt", "ev"]]  # keep only necessary coloumns
    mean_ev_df = ev_df.groupby("zt").mean(numeric_only=True).reset_index()

    time = mean_ev_df["zt"].values.astype("float")
    ev = mean_ev_df["ev"].values.astype("float")

    sort_indx = np.argsort(time)
    time = time[sort_indx]
    ev = ev[sort_indx]

    # try:
    #     half_time = time[ev <= (ev.max() / 2)][0]
    # except:
    #     half_time = time[-1]

    # linfit = stats.linregress(time, ev)

    ev_autocorr = np.correlate(ev, ev, "same")
    ev_autocorr = ev_autocorr[len(ev_autocorr) // 2 :]  # half length
    max_val = ev_autocorr[0]
    half_ind = np.where(ev_autocorr < (max_val / 2))[0][0]

    return pd.DataFrame({"tc": time[half_ind]}, index=[0])


def get_pcorr(
    neurons: core.Neurons,
    epochs: core.Epoch,
    sub_epochs: core.Epoch = None,
    ignore_epochs=None,
    bin_size=0.25,
):
    """Get pairwise correlation within epochs

    Parameters
    ----------
    neurons : core.Neurons
        _description_
    epochs : core.Epoch
        _description_
    sub_epochs : core.Epoch
        _description_
    ignore_epochs : _type_, optional
        _description_, by default None
    bin_size : float, optional
        _description_, by default 0.25

    Returns
    -------
    _type_
        _description_
    """
    pairs_bool = neurons.get_waveform_similarity() < 0.8
    n_pairs = np.tril(pairs_bool, k=-1).sum()  # number of pairs

    _get_corr = lambda epoch: (
        neurons.time_slice(epoch[0], epoch[1])
        .get_binned_spiketrains(bin_size=bin_size, ignore_epochs=ignore_epochs)
        .get_pairwise_corr(pairs_bool=pairs_bool)
    )

    pcorr = []
    for e in epochs.itertuples():
        if sub_epochs is not None:
            e_sub_epochs = sub_epochs.time_slice(e.start, e.stop, strict=False)

            if e_sub_epochs.durations.sum() > 1:
                # In grosmark Buddy session sometimes there are 'zero' duration epoch, so taking care of that
                e_sub_epochs = e_sub_epochs.duration_slice(min_dur=0.5)
                e_spkcounts = np.hstack(
                    neurons.get_spikes_in_epochs(e_sub_epochs, bin_size=bin_size)[0]
                )
                e_binspk = core.BinnedSpiketrain(
                    e_spkcounts,
                    bin_size=bin_size,
                    neuron_ids=neurons.neuron_ids,
                    shank_ids=neurons.shank_ids,
                )
                e_corr = e_binspk.get_pairwise_corr(pairs_bool)
            else:
                e_corr = np.nan * np.ones(n_pairs)

        else:
            e_corr = _get_corr([e.start, e.stop])

        pcorr.append(e_corr)

    pcorr = np.array(pcorr).T

    assert pcorr.shape[0] == n_pairs, "Number of pairs mismatch"

    return pcorr, epochs.labels


def get_high_NREM_pcorr_df(thresh=0.5):
    nrem_duration_df = subjects.GroupData().nrem_duration_NREM
    nrem_duration_df["zt"] = nrem_duration_df["zt"].astype("float")
    mean_duration = (
        nrem_duration_df.groupby(["grp", "zt"]).mean(numeric_only=True).reset_index()
    )
    mean_duration["duration"] = mean_duration["duration"] / (900)
    mean_duration = mean_duration[mean_duration.duration >= thresh].reset_index(
        drop=True
    )

    nrem_pcorr_df = subjects.GroupData().pairwise_correlations_NREM

    new_df = []
    for g, grp in enumerate(["NSD"]):
        df = nrem_pcorr_df[nrem_pcorr_df.grp == grp]

        grp_dur = mean_duration[mean_duration.grp == grp]
        good_col = list(grp_dur.zt.values.astype("int")) + [
            "PRE",
            "MAZE",
            "session",
            "grp",
        ]

        new_df.append(df.loc[:, good_col].copy())

    new_df = pd.concat(new_df, ignore_index=True)
    return new_df
