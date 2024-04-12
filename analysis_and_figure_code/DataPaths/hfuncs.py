from base64 import b64encode
from io import BytesIO

import bokeh.models as bmodels
import bokeh.plotting as bplot
import numpy as np
import pandas as pd
from matplotlib import cm
from neuropy.utils import signal_process
from PIL import Image
from scipy import stats
from skimage import img_as_ubyte
from neuropy import core
from neuropy.utils.ccg import correlograms
from neuropy.io import OptitrackIO
from pathlib import Path
from datetime import datetime
from neuropy.core import Position, Signal
from typing import Union
from neuropy.utils.mathutil import min_max_scaler


def wavelet_gamma_theta_phase(signal, theta_phase, binsize=9, frgamma=None, fs=1250):
    if frgamma is None:
        frgamma = np.arange(25, 150)

    # ----- wavelet power for gamma oscillations----------
    wavdec = signal_process.wavelet_decomp(signal, freqs=frgamma, sampfreq=fs)
    wav = wavdec.colgin2009()
    wav = stats.zscore(wav, axis=1)

    # ----segmenting gamma wavelet at theta phases ----------
    bin_angle = np.linspace(0, 360, int(360 / binsize) + 1)
    phase_centers = bin_angle[:-1] + np.diff(bin_angle).mean() / 2

    bin_ind = np.digitize(theta_phase, bin_angle)

    gamma_at_theta = pd.DataFrame()
    for i in np.unique(bin_ind):
        find_where = np.where(bin_ind == i)[0]
        gamma_at_theta[phase_centers[i - 1]] = np.mean(wav[:, find_where], axis=1)
    gamma_at_theta.insert(0, column="freq", value=frgamma)
    gamma_at_theta.set_index("freq", inplace=True)

    return gamma_at_theta


def to_png(arr):
    out = BytesIO()
    im = Image.fromarray(arr)
    im.save(out, format="png")
    return out.getvalue()


def b64_image_files(images, colormap="hot", conv=5):
    cmap = cm.get_cmap(colormap)
    urls = []
    for im in images:
        im = np.apply_along_axis(
            np.convolve, axis=0, arr=im, v=np.ones(2 * conv + 1), mode="same"
        )
        im_ = np.flipud((im - np.nanmin(im)) / np.nanmax(im))
        png = to_png(img_as_ubyte(cmap(im_)))
        url = "data:image/png;base64," + b64encode(png).decode("utf-8")
        urls.append(url)
    return urls


def plot_in_bokeh(
    x,
    y,
    img_arr,
    annotate=None,
    color_by=None,
    palette="jet",
    size=5,
    width=1200,
    height=800,
    p=None,
):
    if annotate is not None:
        assert len(annotate) == 2
        annotate_keys = list(annotate.keys())

        tooltips = f"""
            <div>
                <div>
                    <img
                        src="@imgs" height="80" alt="@imgs" width="80"
                        style="float: left; margin: 0px 15px 15px 0px;image-rendering: pixelated"
                        border="2"
                    ></img>
                </div>
                <div>
                    <span style="font-size: 12px; color: #212121;">{annotate_keys[0]}: @{annotate_keys[0]}, </span>
                    <span style="font-size: 12px; color: #212121;">{annotate_keys[1]}: @{annotate_keys[1]} </span>
                </div>
            </div>
        """

    else:
        tooltips = f"""
            <div>
                <div>
                    <img
                        src="@imgs" height="100" alt="@imgs" width="100"
                        style="float: left; margin: 0px 15px 15px 0px;image-rendering: pixelated"
                        border="2"
                    ></img>
                </div>
            </div>
        """

    arr_images = b64_image_files(img_arr)

    if color_by is not None:
        cmap = cm.get_cmap(palette)
        color_by = color_by - np.min(color_by)
        color_by /= np.max(color_by)
        colors = cmap(color_by)
    else:
        colors = ["red"] * len(x)

    data_dict = dict(x=x, y=y, imgs=arr_images, colors=colors)
    data_dict = data_dict | annotate

    source = bplot.ColumnDataSource(data_dict)

    if p is None:
        p = bplot.figure(
            width=width,
            height=height,
            tooltips=tooltips,
            x_axis_label="Time",
            y_axis_label="Replay score",
            title="Replay score over time, Hover to see posterior",
        )

    p.circle("x", "y", size=size, source=source, color="colors")

    return p


def plot_replay_in_bokeh(
    data: core.Epoch, palette="jet_r", size=8, width=1200, height=800
):
    df = data.to_dataframe()
    x = data.starts
    y = df.score.values
    img_arr = data.metadata["posterior"]
    color_by = df["p_value"]
    velocity = np.round(df.velocity.values, 2)
    ind = df.index

    # info to show on images
    tooltips = """
        <div>
            <div>
                <img
                    src="@imgs" height="200" alt="@imgs" width="200"
                    style="float: left; margin: 0px 15px 15px 0px;image-rendering: pixelated"
                    border="2"
                ></img>
            </div>
            <div>
                <span style="font-size: 15px; color: #212121;">#@ind, </span>
                <span style="font-size: 15px; color: #212121;">Score: @y, </span>
                <span style="font-size: 15px; color: #212121;">V: @v</span>
            </div>

        </div>
    """

    arr_images = b64_image_files(img_arr)

    if color_by is not None:
        cmap = cm.get_cmap(palette)
        color_by = color_by - np.min(color_by)
        color_by /= np.max(color_by)
        colors = cmap(color_by)
    else:
        colors = ["red"] * len(x)

    source = bplot.ColumnDataSource(
        data=dict(
            ind=ind,
            x=x,
            y=y,
            v=velocity,
            imgs=arr_images,
            colors=colors,
        )
    )

    p = bplot.figure(
        width=width,
        height=height,
        tooltips=tooltips,
        x_axis_label="Time",
        y_axis_label="Replay score",
        title="Replay score over time, Hover to see posterior",
    )
    p.circle("x", "y", size=size, source=source, color="colors")
    return p


def linearize_using_shapely(position: core.Position):
    from matplotlib.backend_bases import MouseButton
    from shapely.geometry import LineString, Point
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(position.x, position.y)

    global coord
    coord = []
    global flag
    flag = 0

    def on_click(event):
        if (event.button is MouseButton.LEFT) and (event.inaxes):
            global coord
            coord.append((event.xdata, event.ydata))
            x = event.xdata
            y = event.ydata

            ax.plot(x, y, "o", color="r")
            fig.canvas.draw()  # redraw the figure

        if event.button is MouseButton.RIGHT:
            flag = 1
            fig.disconnent()

    fig.canvas.mpl_connect("button_press_event", on_click)

    while flag != 0:
        line = LineString(coord)
        lin_pos = []
        for x, y in zip(position.x, position.y):
            lin_pos.append(line.project(Point(x, y)))

        return lin_pos


def whiten_signal(signal: core.Signal):
    from statsmodels.tsa.ar_model import AutoReg

    trace = np.pad(signal.traces[0], (2, 0), "constant", constant_values=(0,))
    model = AutoReg(trace, 2, old_names=False)
    res = model.fit().predict(0, len(trace) - 1)[2:]

    return core.Signal(
        traces=(signal.traces[0] - res).reshape(1, -1),
        sampling_rate=signal.sampling_rate,
        t_start=signal.t_start,
    )


def get_ccg(arr1, arr2, bin_size=0.001, window_size=0.8, fs=1250):
    times = np.concatenate((arr1, arr2))
    ids = np.concatenate([np.ones(len(arr1)), 2 * np.ones(len(arr2))]).astype("int")
    sort_indx = np.argsort(times)

    ccgs = correlograms(
        times[sort_indx],
        ids[sort_indx],
        sample_rate=fs,
        bin_size=bin_size,
        window_size=window_size,
    )
    t = np.linspace(-window_size / 2, window_size / 2, ccgs.shape[-1])

    return t, ccgs[0, 1, :]


def position_alignment(
    opti_data: OptitrackIO, datetime_csv: Path, ttl_signal: Signal = None, fs=30000
):
    rec_datetime = pd.read_csv(datetime_csv)
    n_chunks = len(rec_datetime)
    nframes_chunk = rec_datetime.nFrames.values
    total_frames = np.sum(nframes_chunk)
    total_duration = total_frames / fs

    # --- starts,stops relative to start of .dat/.eeg file
    stops = np.cumsum(nframes_chunk) / fs
    starts = np.insert(stops[:-1], 0, 0)

    opti_datetime_starts = opti_data.datetime_starts
    opti_start = None
    if ttl_signal is not None:
        assert int(total_duration) == int(ttl_signal.duration)
        time = ttl_signal.time
        ttl = np.where((min_max_scaler(ttl_signal.traces[0])) > 0.5, 1, 0)
        ttl_diff = np.diff(np.insert(ttl, 0, 0))
        ttl_starts = time[ttl_diff == 1]
        ttl_stops = time[ttl_diff == -1]

    # ---- startimes of concatenated .dat files -------
    tracking_sRate = opti_data.sampling_rate
    data_time = []
    for i, file_time in enumerate(rec_datetime["StartTime"]):
        tbegin = datetime.strptime(file_time, "%Y-%m-%d_%H-%M-%S")
        datfile_start = starts[i]
        time_diff = ttl_starts - datfile_start
        closest_ttl_dist = np.where(time_diff > 0, time_diff, np.inf).min()

        datetime_diff = np.array(
            [(t - tbegin).total_seconds() for t in opti_datetime_starts]
        )
        closest_opti_dist = np.where(datetime_diff > 0, datetime_diff, np.inf).min()

        tbegin_error = closest_opti_dist - closest_ttl_dist

        assert tbegin_error > 0

        if tbegin_error < 1:
            tbegin = tbegin + pd.Timedelta(tbegin_error, unit="sec")

        nframes = nframes_chunk[i]
        duration = pd.Timedelta(nframes / fs, unit="sec")
        tend = tbegin + duration
        trange = pd.date_range(
            start=tbegin,
            end=tend,
            periods=int(duration.total_seconds() * tracking_sRate),
            inclusive="left",
        )
        data_time.extend(trange)
    data_time = pd.to_datetime(data_time)

    x, y, z = opti_data.get_position_at_datetimes(data_time)
    traces = np.vstack((z, x, y))

    return Position(traces=traces, t_start=0, sampling_rate=opti_data.sampling_rate)


def get_post_epochs(paradigm: core.Epoch, include_pre=True, include_maze=True):
    post = paradigm["post"].flatten()
    zts = np.array([0, 2.5, 5])
    post_starts = zts * 3600 + post[0]
    post_stops = post_starts + 2.5 * 3600

    labels = ["0-2.5", "2.5-5", "5-7.5"]

    if include_maze:
        maze = paradigm["maze"].flatten()
        post_starts = np.insert(post_starts, 0, maze[0])
        post_stops = np.insert(post_stops, 0, maze[1])
        labels = ["MAZE"] + labels

    if include_pre:
        pre = paradigm["pre"].flatten()
        pre = [np.max([pre[0], pre[1] - 2.5 * 3600]), pre[1]]

        post_starts = np.insert(post_starts, 0, pre[0])
        post_stops = np.insert(post_stops, 0, pre[1])
        labels = ["PRE"] + labels

    return core.Epoch.from_array(post_starts, post_stops, labels)


def merge_quiet_wake(states):
    slp_states = states.to_dataframe()
    slp_states.loc[slp_states.label.isin(["AW", "QW"]), "label"] = "WK"
    slp_states = core.Epoch(slp_states).merge_neighbors()
    return slp_states
