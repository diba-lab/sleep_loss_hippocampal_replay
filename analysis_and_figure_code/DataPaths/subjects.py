from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from neuropy import core
from neuropy.io import BinarysignalIO, NeuroscopeIO
from scipy.ndimage import gaussian_gradient_magnitude
from scipy import stats
import matplotlib as mpl
from neuropy import plotting
from collections import namedtuple
import zipfile


class SdFig:
    def __init__(self) -> None:
        self.common_settings = dict(
            fontsize=5, axis_lw=0.8, tick_size=2, constrained_layout=False
        )

    def fig1(self, nrows=9, ncols=12):
        fig = plotting.Fig(nrows, ncols, **self.common_settings)
        return fig

    def fig_supp(self, nrows=8, ncols=8, **kwargs):
        fig = plotting.Fig(nrows, ncols, **self.common_settings, **kwargs)
        return fig

    def fig1_supp(self, nrows=8, ncols=8):
        fig = plotting.Fig(nrows, ncols, **self.common_settings)
        return fig

    def fig2(self, nrows=8, ncols=10):
        fig = plotting.Fig(nrows, ncols, **self.common_settings)
        return fig

    def fig2_supp(self, nrows=8, ncols=8):
        fig = plotting.Fig(nrows, ncols, **self.common_settings)
        return fig

    def fig3(self, nrows=14, ncols=6):
        fig = plotting.Fig(nrows, ncols, **self.common_settings)
        return fig

    def fig4(self, nrows=6, ncols=5):
        fig = plotting.Fig(nrows, ncols, **self.common_settings)
        return fig


def get_statannot_ranksum():
    from statannotations.stats.StatTest import StatTest

    custom_long_name = "Wilcoxon_ranksum"
    custom_short_name = "Wilcoxon_ranksum"
    custom_func = stats.ranksums
    custom_test = StatTest(custom_func, custom_long_name, custom_short_name)
    return custom_test


def adjust_lightness(color, amount=0.5):
    import colorsys

    import matplotlib.colors as mc

    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    c = colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])
    return mc.to_hex(c)


def colors_sd(amount=1):
    return [
        adjust_lightness("#424242", amount=amount),
        adjust_lightness("#eb4034", amount=amount),
    ]


def colors_sd_light(amount=1):
    return [
        adjust_lightness("#707070", amount=amount),
        adjust_lightness("#f18179", amount=amount),
    ]


def colors_tn(amount=1):
    return [
        adjust_lightness("#e9cc2b", amount=amount),
        adjust_lightness("#12d399", amount=amount),
    ]


def colors_rs(amount=1):
    return [adjust_lightness("#5599ff", amount=amount)]


colors_sleep = {
    "AW": "k",
    "QW": "k",
    "REM": "k",
    "NREM": "k",
}


colors_sleep_old = {
    "nrem": "#a3a3a3",
    "rem": "#a3a3a3",
    "quiet": "#a3a3a3",
    "active": "#a3a3a3",
}

hypno_kw = dict(labels_order=["NREM", "REM", "QW", "AW"], colors=colors_sleep)


lineplot_kw = dict(
    marker="o",
    err_style="bars",
    linewidth=1,
    legend=None,
    mew=0.2,
    markersize=2,
    err_kws=dict(elinewidth=1, zorder=-1, capsize=1),
)

errorbar_kw = dict(
    marker="o",
    capsize=1,
    elinewidth=1,
    mec="w",
    markersize=2,
    linewidth=1,
    mew=0.2,
)


def boxplot_kw(color, lw=1):
    return dict(
        showfliers=False,
        linewidth=lw,
        boxprops=dict(facecolor="none", edgecolor=color),
        showcaps=True,
        capprops=dict(color=color),
        medianprops=dict(color=color, lw=lw),
        whiskerprops=dict(color=color),
    )


stat_kw = dict(
    text_format="star",
    loc="inside",
    # verbose=True,
    fontsize=mpl.rcParams["axes.labelsize"],
    line_width=0.5,
    line_height=0.01,
    text_offset=0.2,
    # line_offset=0.2,
    # line_offset_to_group=0.9,
    pvalue_thresholds=[[0.05, "*"], [1, "ns"]]
    # pvalue_format={'star':[[0.05, "*"],[1, "ns"]]},
    # pvalue_format= {'correction_format': '{star} ({suffix})',
    #                           'fontsize': 'small',
    #                           'pvalue_format_string': '{:.3e}',
    #                           'show_test_name': True,
    #                         #   'simple_format_string': '{:.2f}',
    #                           'text_format': 'star',
    #                           'pvalue_thresholds': [
    #                               [1e-4, "*"],
    #                               [1e-3, "*"],
    #                               [1e-2, "*"],
    #                               [0.05, "*"],
    #                               [1, "ns"]]
    #                           },
    # color= 'r',
    # line_offset_to_box=0.2,
    # use_fixed_offset=True,
)

sns_boxplot_kw = dict(
    linewidth=0.8,
    palette=colors_sd(1),
    saturation=1,
    showfliers=False,
    # linewidth=lw,
    boxprops=dict(edgecolor="k"),
    showcaps=True,
    capprops=dict(color="k"),
    medianprops=dict(color="k"),
    whiskerprops=dict(color="k"),
)

sns_violin_kw = dict(
    palette=colors_sd(1),
    saturation=1,
    linewidth=0.4,
    cut=True,
    split=False,
    inner="box",
    showextrema=False,
    # showmeans=True,
)


fig_folder = Path("/home/nkinsky/Documents/figures/")
fig_root = Path("/home/nkinsky/Documents/figures/")
figpath_sd = fig_root / "sleep_deprivation"
figpath_tn = fig_root / "two_novel"


class ProcessData:
    def __init__(self, basepath, tag=None):
        basepath = Path(basepath)
        try:
            xml_files = sorted(basepath.glob("*.xml"))
            assert len(xml_files) == 1, f"Found {len(xml_files)} .xml files"
            fp = xml_files[0].with_suffix("")
            self.recinfo = NeuroscopeIO(xml_files[0])
            if self.recinfo.eeg_filename.is_file():
                self.eegfile = BinarysignalIO(
                    self.recinfo.eeg_filename,
                    n_channels=self.recinfo.n_channels,
                    sampling_rate=self.recinfo.eeg_sampling_rate,
                )

            if self.recinfo.dat_filename.is_file():
                self.datfile = BinarysignalIO(
                    self.recinfo.dat_filename,
                    n_channels=self.recinfo.n_channels,
                    sampling_rate=self.recinfo.dat_sampling_rate,
                )

        except:
            fp = basepath / basepath.name

        self.filePrefix = fp
        self.sub_name = fp.name[:4]

        self.tag = tag

        if (f := self.filePrefix.with_suffix(".animal.npy")).is_file():
            d = np.load(f, allow_pickle=True).item()
            self.animal = core.Animal.from_dict(d)
            self.name = self.animal.name + self.animal.day

        self.probegroup = core.ProbeGroup.from_file(fp.with_suffix(".probegroup.npy"))

        # ----- epochs --------------
        # epoch_names = [
        #     "paradigm",
        #     "artifact",
        #     "brainstates",
        #     "spindle",
        #     "ripple",
        #     "theta",
        #     "pbe",
        # ]
        # for e in epoch_names:
        #     setattr(self, e, core.Epoch.from_file(fp.with_suffix(f".{e}.npy")))
        if (f := self.filePrefix.with_suffix(".best_channels.npy")).is_file():
            best_chans = namedtuple("BestChannels", ["theta", "slow_wave"])
            d = np.load(f, allow_pickle=True).item()
            self.best_channels = best_chans(d["theta"], d["slow_wave"])

        self.paradigm = core.Epoch.from_file(fp.with_suffix(".paradigm.npy"))
        self.artifact = core.Epoch.from_file(fp.with_suffix(".artifact.npy"))
        # self.brainstates = core.Epoch.from_file(fp.with_suffix(".brainstates.npy"))

        self.brainstates = core.Epoch.from_file(fp.with_suffix(".brainstates.finer.npy"))

        self.sw = core.Epoch.from_file(fp.with_suffix(".sw.npy"))
        self.spindle = core.Epoch.from_file(fp.with_suffix(".spindle.npy"))
        self.ripple = core.Epoch.from_file(fp.with_suffix(".ripple.npy"))
        self.theta = core.Epoch.from_file(fp.with_suffix(".theta.npy"))
        self.theta_epochs = core.Epoch.from_file(fp.with_suffix(".theta.epochs.npy"))
        self.pbe = core.Epoch.from_file(fp.with_suffix(".pbe.npy"))
        # self.off = core.Epoch.from_file(fp.with_suffix(".off.npy"))
        self.off_epochs = core.Epoch.from_file(fp.with_suffix(".off_epochs.npy"))
        self.micro_arousals = core.Epoch.from_file(fp.with_suffix(".micro_arousals.npy"))

        self.maze_run = core.Epoch.from_file(fp.with_suffix(".maze.running.npy"))
        self.maze1_run = core.Epoch.from_file(fp.with_suffix(".maze1.running.npy"))
        self.maze2_run = core.Epoch.from_file(fp.with_suffix(".maze2.running.npy"))
        self.remaze_run = core.Epoch.from_file(fp.with_suffix(".remaze.running.npy"))

        # Piezo epochs caputuring interruptions during sleep deprivations
        self.handling = core.Epoch.from_file(fp.with_suffix(".handling.npy"))

        # ---- neurons related ------------

        if (f := self.filePrefix.with_suffix(".neurons.iso.npy")).is_file():
            d = np.load(f, allow_pickle=True).item()
            self.neurons_iso = core.Neurons.from_dict(d)

        if (f := self.filePrefix.with_suffix(".position.npy")).is_file():
            d = np.load(f, allow_pickle=True).item()
            self.position = core.Position.from_dict(d)

        if (f := self.filePrefix.with_suffix(".maze.linear.npy")).is_file():
            d = np.load(f, allow_pickle=True).item()
            self.maze = core.Position.from_dict(d)

        if (f := self.filePrefix.with_suffix(".remaze.linear.npy")).is_file():
            d = np.load(f, allow_pickle=True).item()
            self.remaze = core.Position.from_dict(d)

        if (f := self.filePrefix.with_suffix(".maze1.linear.npy")).is_file():
            d = np.load(f, allow_pickle=True).item()
            self.maze1 = core.Position.from_dict(d)

        if (f := self.filePrefix.with_suffix(".maze2.linear.npy")).is_file():
            d = np.load(f, allow_pickle=True).item()
            self.maze2 = core.Position.from_dict(d)

    @property
    def delta_wave(self):
        if (f := self.filePrefix.with_suffix(".delta_wave.npy")).is_file():
            d = np.load(f, allow_pickle=True).item()
            return core.Epoch.from_dict(d)

    @property
    def emg(self):
        if (f := self.filePrefix.with_suffix(".emg.npy")).is_file():
            d = np.load(f, allow_pickle=True).item()
            return core.Signal.from_dict(d)
        else:
            return None

    @property
    def pbe_filters(self):
        """Code in 'sd_pbe_creation.ipynb'. This data has additional columns for PBEs depicting criteria such as:
        1) is_rpl
        2) is_5units
        3) is_80percetbins
        4) is_rest
        """
        if (f := self.filePrefix.with_suffix(".pbe.filters.npy")).is_file():
            d = np.load(f, allow_pickle=True).item()
            return core.Epoch.from_dict(d)

    @property
    def get_pbe_filters_bool(self, is_rpl=True):
        if (f := self.filePrefix.with_suffix(".pbe.filters.npy")).is_file():
            d = np.load(f, allow_pickle=True).item()
            pbe_filters = core.Epoch.from_dict(d)
            good_bool = (
                pbe_filters.is_rpl
                & pbe_filters.is_5neurons
                # & pbe_filter.is_lowtheta
                & pbe_filters.is_rest
            )

    @property
    def replay_filtered(self):
        """Contains events which satisfy the following criteria:
        1) has 1std ripple power
        2) has atleast 5 units firing
        3) ripple happens during rest
        4) contains continuous trajectory with jump distance < 40 cm

        Code cell in 'sd_replay_filters.ipynb' and was generated using '.pbe.replay.mua' file.
        """
        # if (f := self.filePrefix.with_suffix(".replay_pbe.npy")).is_file():
        if (f := self.filePrefix.with_suffix(f".pbe.replay.filtered.npy")).is_file():
            d = np.load(f, allow_pickle=True).item()
            return core.Epoch.from_dict(d)

    @property
    def replay_filtered20jd(self):
        """Contains events which satisfy the following criteria:
        1) has 1std ripple power
        2) has atleast 5 units firing
        3) ripple happens during rest
        4) contains continuous trajectory with jump distance < 20 cm

        Code cell in 'sd_replay_filters.ipynb' and was generated using '.pbe.replay.mua' file.
        """
        # if (f := self.filePrefix.with_suffix(".replay_pbe.npy")).is_file():
        if (f := self.filePrefix.with_suffix(f".pbe.replay.filtered.jumpthresh20.npy")).is_file():
            d = np.load(f, allow_pickle=True).item()
            return core.Epoch.from_dict(d)

    @property
    def replay_filtered1h(self):
        """Contains events which satisfy the following criteria:
        1) has 1std ripple power
        2) has atleast 5 units firing
        3) ripple happens during rest
        4) contains continuous trajectory

        Code cell in 'sd_replay_filters.ipynb' and was generated using '.pbe.replay.mua' file.
        """
        # if (f := self.filePrefix.with_suffix(".replay_pbe.npy")).is_file():
        if (f := self.filePrefix.with_suffix(".pbe.replay.filtered1h.npy")).is_file():
            d = np.load(f, allow_pickle=True).item()
            return core.Epoch.from_dict(d)

    @property
    def replay_pbe_mua(self):
        # if (f := self.filePrefix.with_suffix(".replay_pbe.npy")).is_file():
        if (f := self.filePrefix.with_suffix(".pbe.replay.mua.npy")).is_file():
            d = np.load(f, allow_pickle=True).item()
            return core.Epoch.from_dict(d)

    @property
    def replay_pbe_mua_column(self):
        # if (f := self.filePrefix.with_suffix(".replay_pbe.npy")).is_file():
        if (f := self.filePrefix.with_suffix(".pbe.replay.mua.column.npy")).is_file():
            d = np.load(f, allow_pickle=True).item()
            return core.Epoch.from_dict(d)

    # @property
    # def replay_pbe_mua_column_max(self):
    #     # if (f := self.filePrefix.with_suffix(".replay_pbe.npy")).is_file():
    #     if (
    #         f := self.filePrefix.with_suffix(".pbe.replay.mua.column_cycle.maxjd.npy")
    #     ).is_file():
    #         d = np.load(f, allow_pickle=True).item()
    #         return core.Epoch.from_dict(d)

    # @property
    # def replay_wcorr(self):
    #     # if (f := self.filePrefix.with_suffix(".replay_pbe.npy")).is_file():
    #     if (f := self.filePrefix.with_suffix(".pbe.wcorr.npy")).is_file():
    #         d = np.load(f, allow_pickle=True).item()
    #         return core.Epoch.from_dict(d)

    # @property
    # def replay_radon(self):
    #     # if (f := self.filePrefix.with_suffix(".replay_pbe.npy")).is_file():
    #     if (f := self.filePrefix.with_suffix(".pbe.radon.npy")).is_file():
    #         d = np.load(f, allow_pickle=True).item()
    #         return core.Epoch.from_dict(d)

    @property
    def replay_radon_mua(self):
        # if (f := self.filePrefix.with_suffix(".replay_pbe.npy")).is_file():
        if (f := self.filePrefix.with_suffix(".pbe.radon.mua.npy")).is_file():
            d = np.load(f, allow_pickle=True).item()
            return core.Epoch.from_dict(d)

    # @property
    # def replay_wcorr_mua(self):
    #     # if (f := self.filePrefix.with_suffix(".replay_pbe.npy")).is_file():
    #     if (f := self.filePrefix.with_suffix(".pbe.wcorr.mua.npy")).is_file():
    #         d = np.load(f, allow_pickle=True).item()
    #         return core.Epoch.from_dict(d)

    # @property
    # def replay_spearman(self):
    #     # if (f := self.filePrefix.with_suffix(".replay_pbe.npy")).is_file():
    #     if (f := self.filePrefix.with_suffix(".pbe.replay.spearman.npy")).is_file():
    #         d = np.load(f, allow_pickle=True).item()
    #         return core.Epoch.from_dict(d)

    @property
    def remaze_replay_pbe(self):
        if (f := self.filePrefix.with_suffix(".remaze_replay_pbe.npy")).is_file():
            d = np.load(f, allow_pickle=True).item()
            return core.Epoch.from_dict(d)

    @property
    def neurons(self):
        # it is relatively heavy on memory hence loaded only while required
        if (f := self.filePrefix.with_suffix(".neurons.npy")).is_file():
            d = np.load(f, allow_pickle=True).item()
            return core.Neurons.from_dict(d)

    @property
    def neurons_stable(self):
        # it is relatively heavy on memory hence loaded only while required
        if (f := self.filePrefix.with_suffix(".neurons.stable.npy")).is_file():
            d = np.load(f, allow_pickle=True).item()
            return core.Neurons.from_dict(d)

    @property
    def mua(self):
        if (f := self.filePrefix.with_suffix(".mua.npy")).is_file():
            d = np.load(f, allow_pickle=True).item()
            return core.Mua.from_dict(d)

    def get_zt_1h(self, include_pre=True, include_maze=True, pre_length=2.5):
        post = self.paradigm["post"].flatten()
        # post_starts = np.array([0, 4, 5]) * 3600 + post[0]
        post_starts = np.array([0, 1, 2, 3, 4, 5]) * 3600 + post[0]
        post_stops = post_starts + 3600

        # labels = ["0-1", "4-5", "5-6"]
        labels = ["0-1", "1-2", "2-3", "3-4", "4-5", "5-6"]

        if include_maze:
            maze = self.paradigm["maze"].flatten()
            post_starts = np.insert(post_starts, 0, maze[0])
            post_stops = np.insert(post_stops, 0, maze[1])
            labels = ["MAZE"] + labels

        if include_pre:
            pre = self.paradigm["pre"].flatten()
            pre = [np.max([pre[0], pre[1] - pre_length * 3600]), pre[1]]

            post_starts = np.insert(post_starts, 0, pre[0])
            post_stops = np.insert(post_stops, 0, pre[1])
            labels = ["PRE"] + labels

        return core.Epoch.from_array(post_starts, post_stops, labels)

    def get_zt_epochs(self, include_pre=True, include_maze=True):
        post = self.paradigm["post"].flatten()
        # zts = np.array([0, 2.5, 5])
        # post_starts = zts * 3600 + post[0]
        # post_stops = post_starts + 2.5 * 3600
        zts = np.arange(0, 8, 2.5) * 3600 + post[0]
        post_starts, post_stops = zts[:-1], zts[1:]

        labels = ["0-2.5", "2.5-5", "5-7.5"]

        if include_maze:
            maze = self.paradigm["maze"].flatten()
            post_starts = np.insert(post_starts, 0, maze[0])
            post_stops = np.insert(post_stops, 0, maze[1])
            labels = ["MAZE"] + labels

        if include_pre:
            pre = self.paradigm["pre"].flatten()
            pre = [np.max([pre[0], pre[1] - 2.5 * 3600]), pre[1]]

            post_starts = np.insert(post_starts, 0, pre[0])
            post_stops = np.insert(post_stops, 0, pre[1])
            labels = ["PRE"] + labels

        return core.Epoch.from_array(post_starts, post_stops, labels)

    def get_sliding_zt_epochs(
        self, window=900, slideby=None, include_pre=True, include_maze=True
    ):
        post = self.paradigm["post"].flatten()
        post_dur = (post[1] - post[0]) / 3600

        if slideby is None:
            slideby = window

        if self.tag == "NSD_HM":
            post_starts = np.arange(0, 2.7 * 3600 + window, slideby) + post[0]
        elif self.tag == "NSD_GM":
            post_starts = (
                np.arange(0, (post_dur - 0.3) * 3600 + window, slideby) + post[0]
            )
        else:
            post_starts = np.arange(0, 7.5 * 3600 + window, slideby) + post[0]

        post_stops = post_starts + window
        post_mids = (post_starts + post_stops) / 2
        labels = [np.round((t - post[0]) / 3600, 2) for t in post_mids]

        if include_maze:
            maze = self.paradigm["maze"].flatten()
            post_starts = np.insert(post_starts, 0, maze[0])
            post_stops = np.insert(post_stops, 0, maze[1])
            labels = ["MAZE"] + labels

        if include_pre:
            pre = self.paradigm["pre"].flatten()
            pre = [np.max([pre[0], pre[1] - 2.5 * 3600]), pre[1]]

            post_starts = np.insert(post_starts, 0, pre[0])
            post_stops = np.insert(post_stops, 0, pre[1])
            labels = ["PRE"] + labels

        return core.Epoch.from_array(post_starts, post_stops, labels)

    @property
    def data_table(self):
        files = [
            "paradigm",
            "artifact",
            "brainstates",
            "spindle",
            "ripple",
            "theta",
            "pbe",
            "neurons",
            "position",
            "maze.linear",
            "re-maze.linear",
            "maze1.linear",
            "maze2.linear",
        ]

        df = pd.DataFrame(columns=files)
        is_exist = []
        for file in files:
            if self.filePrefix.with_suffix(f".{file}.npy").is_file():
                is_exist.append(True)
            else:
                is_exist.append(False)

        df.loc[0] = is_exist
        df.insert(0, "session", self.filePrefix.name)

        return df

    def save_data(d, f):
        np.save(f, arr=d)

    def create_time_machine(self, suffix):
        files_id = [
            "animal",
            "paradigm",
            "artifact",
            "brainstates",
            "probegroup",
            "neurons",
            "neurons.stable",
            "ripple",
            "pbe",
            "position",
            "maze.linear",
            "maze.running",
            "remaze.linear",
            "remaze.running",
            "pbe.filters",
            "pbe.replay.filtered",
            "pbe.replay.mua",
            "pbe.replay.mua.column",
            "pbe.replay.mua.column_cycle.maxjd",
            "pbe.wcorr",
            "pbe.radon",
            "pbe.radon.mua",
            "pbe.wcorr.mua",
            "mua",
            "off_epochs",
        ]
        files = [self.filePrefix.with_suffix(f".{_}.npy") for _ in files_id]

        suffix = f".{suffix}.zip"
        zip_filename = self.filePrefix.with_suffix(suffix)
        with zipfile.ZipFile(zip_filename, "w") as zipF:
            for file in files:
                if file.is_file():
                    zipF.write(file, file.name, compress_type=zipfile.ZIP_DEFLATED)

            print("files has been compressed")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.recinfo.source_file.name})\n"


def data_table(sessions: list):
    df = []
    for sess in sessions:
        df.append(sess.data_table)

    return pd.concat(df, ignore_index=True)


class Group:
    tag = None
    basedir = Path("/data/Clustering/sessions/")

    def _process(self, rel_path):
        return [ProcessData(self.basedir / rel_path, self.tag)]

    def data_exist(self):
        self.allsess


class Of(Group):
    @property
    def ratJday4(self):
        return self._process("RatJ/Day4/")

    @property
    def ratKday4(self):
        return self._process("RatK/Day4/")

    @property
    def ratNday4(self):
        return self._process("RatN/Day4/")

    @property
    def ratUday5(self):
        return self._process("RatU/RatUDay5OpenfieldSD/")


class Sd(Group):
    tag = "SD"

    @property
    def allsess(self):
        pipelines: List[ProcessData]
        pipelines = (
            self.ratJday1
            + self.ratKday1
            + self.ratNday1
            + self.ratSday3
            + self.ratRday2
            + self.ratUday1
            + self.ratUday4
            + self.ratVday2
        )
        return pipelines

    @property
    def mua_sess(self):
        pipelines: List[ProcessData]
        pipelines = (
            self.ratJday1
            + self.ratKday1
            + self.ratNday1
            + self.ratSday3
            + self.ratRday2
            + self.ratUday1
            + self.ratUday4
            + self.ratVday2
        )
        return pipelines

    @property
    def ripple_sess(self):
        pipelines: List[ProcessData] = (
            self.ratJday1
            + self.ratKday1
            + self.ratNday1
            + self.ratSday3
            + self.ratRday2
            + self.ratUday1
            + self.ratUday4
            + self.ratVday2
        )
        return pipelines

    @property
    def brainstates_sess(self):
        pipelines: List[ProcessData] = (
            self.ratJday1
            + self.ratKday1
            + self.ratNday1
            + self.ratSday3
            + self.ratRday2
            + self.ratUday4
        )
        return pipelines

    @property
    def pf_sess(self):
        pipelines: List[ProcessData] = (
            self.ratKday1
            + self.ratNday1
            + self.ratSday3
            + self.ratUday1
            + self.ratUday4
            + self.ratVday2
            + self.ratRday2
        )
        return pipelines

    @property
    def bilateral(self):
        pipelines: List[ProcessData] = (
            self.ratSday3 + self.ratUday1 + self.ratUday4 + self.ratVday2
        )
        return pipelines

    @property
    def remaze(self):
        pipelines: List[ProcessData] = (
            self.ratSday3 + self.ratUday1 + self.ratUday4 + self.ratVday2 + self.ratRday2
        )
        return pipelines

    @property
    def handling_data_sess(self):
        pipelines: List[ProcessData] = self.ratUday1 + self.ratUday4 + self.ratVday2
        return pipelines

    @property
    def ratJday1(self):
        return self._process("RatJ/Day1/")

    @property
    def ratKday1(self):
        return self._process("RatK/Day1/")

    @property
    def ratNday1(self):
        return self._process("RatN/Day1/")

    @property
    def ratSday3(self):
        return self._process("RatS/Day3SD/")

    @property
    def ratRday2(self):
        return self._process("RatR/Day2SD")

    @property
    def ratUday1(self):
        return self._process("RatU/RatUDay1SD")

    @property
    def ratUday4(self):
        return self._process("RatU/RatUDay4SD")

    @property
    def ratVday2(self):
        return self._process("RatV/RatVDay2SD/")

    # @property
    # def ratUday5(self):
    #     path = "/data/Clustering/sessions/RatU/RatUDay5OpenfieldSD/"
    #     return [ProcessData(path)]

    @property
    def utkuAG_day1(self):
        path = "Utku/AG_2019-12-22_SD_day1/"
        return [ProcessData(path)]

    @property
    def utkuAG_day2(self):
        path = "Utku/AG_2019-12-26_SD_day2/"
        return [ProcessData(path)]

    def __add__(self, other):
        pipelines: List[ProcessData] = self.allsess + other.allsess
        return pipelines

    @staticmethod
    def color(amount=1):
        # return adjust_lightness("#df670c", amount=amount)
        # return adjust_lightness("#f06292", amount=amount)
        # return adjust_lightness("#ff0000", amount=amount)
        return adjust_lightness("#ff8080", amount=amount)

    @staticmethod
    def rs_color(amount=1):
        return adjust_lightness("#00B8D4", amount=amount)


class Nsd(Group):
    tag = "NSD"

    @property
    def allsess(self):
        pipelines: List[ProcessData]
        pipelines = (
            self.ratJday2
            + self.ratKday2
            + self.ratNday2
            + self.ratSday2
            + self.ratRday1
            + self.ratUday2
            + self.ratVday1
            + self.ratVday3
        )
        return pipelines

    @property
    def mua_sess(self):
        pipelines: List[ProcessData]
        pipelines = (
            self.ratJday2
            + self.ratKday2
            + self.ratNday2
            + self.ratSday2
            + self.ratRday1
            + self.ratUday2
            + self.ratVday1
            + self.ratVday3
        )
        return pipelines

    @property
    def ripple_sess(self):
        pipelines: List[ProcessData] = (
            self.ratJday2
            + self.ratKday2
            + self.ratNday2
            + self.ratSday2
            + self.ratRday1
            + self.ratUday2
            + self.ratVday1
            + self.ratVday3
        )
        return pipelines

    @property
    def brainstates_sess(self):
        pipelines: List[ProcessData] = (
            self.ratJday2 + self.ratKday2 + self.ratNday2 + self.ratSday2 + self.ratUday2
        )
        return pipelines

    @property
    def pf_sess(self):
        pipelines: List[ProcessData] = (
            self.ratKday2
            + self.ratNday2
            + self.ratSday2
            + self.ratUday2
            + self.ratVday1
            + self.ratVday3
        )
        return pipelines

    @property
    def bilateral(self):
        pipelines: List[ProcessData] = (
            self.ratSday2 + self.ratUday2 + self.ratVday1 + self.ratVday3
        )
        return pipelines

    @property
    def remaze(self):
        pipelines: List[ProcessData] = (
            self.ratSday2 + self.ratUday2 + self.ratVday1 + self.ratVday3
        )
        return pipelines

    @property
    def ratJday2(self):
        return self._process("RatJ/Day2/")

    @property
    def ratKday2(self):
        return self._process("RatK/Day2/")

    @property
    def ratNday2(self):
        return self._process("RatN/Day2/")

    @property
    def ratSday2(self):
        return self._process("RatS/Day2NSD/")

    @property
    def ratRday1(self):
        return self._process("RatR/Day1NSD/")

    @property
    def ratUday2(self):
        return self._process("RatU/RatUDay2NSD/")

    @property
    def ratVday1(self):
        return self._process("RatV/RatVDay1NSD/")

    @property
    def ratVday3(self):
        return self._process("RatV/RatVDay3NSD")

    def __add__(self, other):
        pipelines: List[ProcessData] = self.allsess + other.allsess
        return pipelines

    @staticmethod
    def color(amount=1):
        # return adjust_lightness("#815bcd", amount=amount)
        # return adjust_lightness("#424242", amount=amount)
        return adjust_lightness("#bdbdbd", amount=amount)


class Tn(Group):
    tag = "TN"
    paths = [
        "/data/Clustering/sessions/RatJ/Day3/",
        "/data/Clustering/sessions/RatK/Day3/",
        "/data/Clustering/sessions/RatN/Day3/",
    ]

    @property
    def allsess(self):
        pipelines: List[ProcessData]
        pipelines = self.ratKday3 + self.ratSday5 + self.ratUday3
        return pipelines

    # @property
    # def ratJday3(self):
    #     return self._process("RatJ/RatJDay3TwoNovel")

    @property
    def ratKday3(self):
        # path = "/data/Clustering/sessions/RatK/Day3/"
        return self._process("RatK/RatKDay3TwoNovel")

    @property
    def ratNday3(self):
        return self._process("RatN/Day3")

    @property
    def ratSday5(self):
        path = "/data/Clustering/sessions/RatS/Day5TwoNovel/"
        return [ProcessData(path)]

    @property
    def ratUday3(self):
        return self._process("RatU/RatUDay3TwoNovel")


class NsdHiro(Group):
    tag = "NSD_HM"

    @property
    def allsess(self):
        pipelines: List[ProcessData]
        pipelines = (
            self.ratRoyMaze1
            # + self.ratRoyMaze2
            # + self.ratRoyMaze3
            + self.ratTedMaze1
            + self.ratTedMaze2
            + self.ratTedMaze3
            + self.ratKevinMaze1
        )
        return pipelines

    @property
    def ratRoyMaze1(self):
        return self._process("Hiro/RoyMaze1")

    # @property
    # def ratRoyMaze2(self):
    #     return self._process("Hiro/RoyMaze2")

    # @property
    # def ratRoyMaze3(self):
    #     return self._process("Hiro/RoyMaze3")

    @property
    def ratTedMaze1(self):
        return self._process("Hiro/TedMaze1")

    @property
    def ratTedMaze2(self):
        return self._process("Hiro/TedMaze2")

    @property
    def ratTedMaze3(self):
        return self._process("Hiro/TedMaze3")

    @property
    def ratKevinMaze1(self):
        return self._process("Hiro/KevinMaze1")


class NsdGrosmark(Group):
    tag = "NSD_GM"

    @property
    def allsess(self):
        pipelines: List[ProcessData]
        pipelines = (
            self.ratAchilles_10252013
            + self.ratAchilles_11012013
            + self.ratBuddy_06272013
            + self.ratCicero_09172014
            + self.ratGatsby_08282013
        )
        return pipelines

    @property
    def ratAchilles_10252013(self):
        return self._process("GrosmarkReclusteredData/Achilles_10252013")

    @property
    def ratAchilles_11012013(self):
        return self._process("GrosmarkReclusteredData/Achilles_11012013")

    @property
    def ratBuddy_06272013(self):
        return self._process("GrosmarkReclusteredData/Buddy_06272013")

    @property
    def ratCicero_09172014(self):
        return self._process("GrosmarkReclusteredData/Cicero_09172014")

    @property
    def ratGatsby_08282013(self):
        return self._process("GrosmarkReclusteredData/Gatsby_08282013")


class NsdOld(Group):
    tag = "NSD_Old"

    @property
    def allsess(self):
        pipelines: List[ProcessData]
        pipelines = self.rat_2022_06_24
        return pipelines

    @property
    def ratB1_2022_06_24(self):
        return self._process("UtkuOldAnimals/RatB1/RatB1_2022-06-24_NSD_CA1_24Hrs")

    @property
    def ratB2_2022_05_28(self):
        return self._process("UtkuOldAnimals/RatB2/RatB2_2022-05-28_NSD_CA1_24hrs")


class SdOld(Group):
    tag = "SD_Old"

    @property
    def allsess(self):
        pipelines: List[ProcessData]
        pipelines = self.rat_2022_06_24
        return pipelines

    @property
    def ratB1_2022_06_27(self):
        return self._process("UtkuOldAnimals/RatB1/RatB1_2022-06-27_SD_CA1_24Hrs")


class SdRol(Group):
    tag = "SD_ROL"

    @property
    def allsess(self):
        pipelines: List[ProcessData]
        pipelines = (
            self.ratN_2019_10_19
            + self.ratA14_2020_02_26
            + self.ratU_2021_08_09
            + self.ratMR10_2021_08_23
        )
        return pipelines

    @property
    def ratN_2019_10_19(self):
        return self._process("rolipram/BGN_2019-10-19_SDROL")

    @property
    def ratA14_2020_02_26(self):
        return self._process("rolipram/A14_2020-02-26_SDROL")

    @property
    def ratU_2021_08_09(self):
        return self._process("rolipram/BGU_2021-08-09_SDROL")

    @property
    def ratMR10_2021_08_23(self):
        return self._process("rolipram/MR10_2021-08-23_SDROL")


class SdSal(Group):
    tag = "SD_SAL"

    @property
    def allsess(self):
        pipelines: List[ProcessData]
        pipelines = (
            self.ratN_2019_10_21
            + self.ratA14_2020_02_23
            + self.ratU_2021_08_11
            + self.ratMR10_2021_08_21
        )

        return pipelines

    @property
    def ratN_2019_10_21(self):
        return self._process("rolipram/BGN_2019-10-21_SDSAL")

    @property
    def ratA14_2020_02_23(self):
        return self._process("rolipram/A14_2020-02-23_SDPBS")

    @property
    def ratU_2021_08_11(self):
        return self._process("rolipram/BGU_2021-08-11_SDSAL")

    @property
    def ratMR10_2021_08_21(self):
        return self._process("rolipram/MR10_2021-08-21_SDPBS")


class SimData(Group):
    tag = "Sim"

    @property
    def allsess(self):
        pipelines: List[ProcessData]
        pipelines = self.ratSim1

        return pipelines

    @property
    def ratSim1(self):
        return self._process("Simulated/RatSim1")


class GroupData:
    __slots__ = (
        "path",
        "add_zt",
        "swa_examples",
        "brainstates_proportion",
        "ripple_psd",
        "ripple_examples",
        "ripple_rate",
        "ripple_rate_statewise_blocks",
        "ripple_total_duration",
        "ripple_features",
        "ripple_features_1h_blocks",
        "ripple_features_normalized",
        "ripple_autocorr",
        "ripple_bootstrap_session_ripples",
        "ripple_normalized_bootstrap_session_ripples",
        "ripple_1h_blocks_bootstrap_session_ripples",
        "ripple_rate_bootstrap_session",
        "ripple_rate_post5h_trend",
        "ripple_rate_post5h_trend_bootstrap",
        "ripple_features_brainstate",
        "ripple_NREM_bootstrap_session_ripples",
        "ripple_WK_bootstrap_session_ripples",
        "pbe_rate",
        "pbe_total_duration",
        "candidate_PBE_duration",
        "candidate_PBE_duration_bootstrap",
        "frate_zscore",
        "frate_post_chunks",
        "frate_post_chunks_statewise",
        "frate_1h_blocks",
        "frate_1h_blocks_bootstrap",
        # "frate_post_chunks_nrem_qw",
        "frate_post_chunks_zscore",
        "frate_ratio_nsd_vs_sd",
        "frate_interneuron_around_Zt5",
        "frate_change_1vs5",
        "frate_change_pre_to_post",
        "frate_pre_to_maze_quantiles_in_POST",
        "frate_pre_to_maze_quantiles_in_POST_shuffled",
        "frate_in_ripple",
        "frate_blocks_bootstrap_session_neurons",
        "frate_blocks_WK_bootstrap_session_neurons",
        "frate_blocks_NREM_bootstrap_session_neurons",
        "frate_IQR_blocks_bootstrap_session_neurons",
        "frate_ShapiroWilk_statistic_blocks_bootstrap",
        "frate_STD_blocks_bootstrap_session_neurons",
        "frate_Kurtosis_blocks_bootstrap_session_neurons",
        "frate_in_ripple_blocks_bootstrap_session_neurons",
        "pairwise_correlations_NREM",
        "pairwise_correlations_WAKE",
        "pairwise_correlations_aligned_by_NREM_onset",
        "pairwise_correlations_aligned_by_WAKE",
        "ei_ratio",
        "ev_pooled",
        "ev_in_chunks",
        "ev_brainstates",
        "ev_1h_blocks",
        "ev_1h_blocks_bootstrap",
        "ev_bootstrap_session",
        "ev_bootstrap_pairs",
        "ev_bootstrap_session_pairs",
        "ev_bootstrap_session_mean",
        "ev_bootstrap_pairs_mean",
        "ev_bootstrap_session_pairs_mean",
        "ev_NSD_WK_sliding_bootstrap_session_pairs",
        "ev_NSD_NREM_sliding_bootstrap_session_pairs",
        "ev_SD_WK_sliding_bootstrap_session_pairs",
        "ev_aligned_by_NREM_onset",
        "ev_NREM_bootstrap",
        "ev_mean_aligned_by_NREM_onset",
        "ev_aligned_by_WAKE",
        "ev_mean_aligned_by_WAKE",
        "pf_norm_tuning",
        "replay_examples",
        "replay_continuous_events",
        "replay_sig_frames",
        "replay_wcorr",
        "replay_wcorr_mua",
        "replay_radon",
        "replay_radon_mua",
        "replay_jumpdist",
        "replay_jumpdist_mua",
        "replay_re_maze_score",
        "replay_post_score",
        "replay_pos_distribution",
        "replay_re_maze_position_distribution",
        "continuous_replay_PBE_duration",
        "continuous_replay_PBE_duration_bootstrap",
        "continuous_replay_proportion_bootstrap",
        "continuous_replay_number",
        "continuous_replay_number_bootstrap",
        "continuous_replay_bias_blocks",
        "replay_continuous_events_1h_blocks",
        "continuous_replay_proportion_1h_blocks_bootstrap",
        "continuous_replay_number_1h_blocks",
        "continuous_replay_number_1h_blocks_bootstrap",
        "candidate_replay_number",
        "candidate_replay_number_bootstrap",
        "remaze_ev_example",
        "remaze_ev_on_POST_example",
        "remaze_ev",
        "remaze_temporal_bias",
        "remaze_maze_paircorr",
        "remaze_first5_paircorr",
        "remaze_first5_subsample",
        "remaze_first5_bootstrap",
        "remaze_last5_paircorr",
        "remaze_corr_across_session",
        "remaze_activation_of_maze",
        "remaze_temporal_bias_com_correlation_across_session",
        "remaze_ensemble_corr_across_sess",
        "remaze_ensemble_activation_across_sess",
        "remaze_ev_on_zt0to5",
        "remaze_ev_on_POST_pooled",
        "post_first5_last5_paircorr",
        "off_rate",
        "off_mean_duration",
        "off_rate_bootstrap_session",
        "nrem_duration_NREM",
        "nrem_duration_aligned_by_nrem_onset",
        "wake_duration_aligned_by_WAKE",
        "ev_tc_NREM",
        "ev_tc_WAKE",
        "ev_tc_linear_high_NREM",
        "ev_tc_linear_WAKE",
        "ev_slopes_high_NREM_bootstrap",
        "ev_slopes_WAKE_bootstrap",
        "ev_mean_tc_aligned_by_NREM_onset",
        "ev_mean_tc_aligned_by_WAKE",
        "delta_wave_rate",
        "delta_wave_amp_blocks",
        "delta_wave_rate_bootstrap",
        "ev_goodness_fit_NREM",
        "ev_goodness_fit_WAKE",
    )

    def __init__(self, add_zt: bool = True) -> None:
        self.path = Path("/home/nkinsky/Documents/sleep_deprivation/ProcessedData")
        self.add_zt = add_zt
        # for f in self.path.iterdir():
        #     setattr(self, f.name, self.load(f.stem))

    def save(self, d, fp):
        if isinstance(d, pd.DataFrame):
            d = d.to_dict()
        data = {"data": d}
        np.save(self.path / fp, data)
        print(f"{fp} saved")

    def load(self, fp):
        data = np.load(self.path / f"{fp}.npy", allow_pickle=True).item()
        try:
            data["data"] = pd.DataFrame(data["data"])
            if self.add_zt:
                data["data"] = add_zt_str(data["data"])
        except:
            pass
        return data

    def __getattr__(self, name: str):
        return self.load(name)["data"]


sd = Sd()
nsd = Nsd()
of = Of()
tn = Tn()
sdrol = SdRol()


def mua_sess():
    return nsd.mua_sess + sd.mua_sess


def pf_sess():
    sessions = nsd.pf_sess + sd.pf_sess
    print(f"#Sessions = {len(sessions)}")
    return sessions


def ripple_sess():
    return nsd.ripple_sess + sd.ripple_sess


def remaze_sess():
    return nsd.remaze + sd.remaze


def bilateral_sess():
    return nsd.bilateral + sd.bilateral


def add_zt_str(df: pd.DataFrame, zt_key="zt", epoch_str=("0-2.5", "2.5-5", "5-7.5")):
    """Fix zt strings to prepend ZT"""
    for epoch_name in epoch_str:
        df.loc[df[zt_key] == epoch_name, zt_key] = f"ZT {epoch_name}"

    return df


def sess_name_fix(str_to_fix):
    """Make capitalized names compatible with NSD and SD attribute names"""
    letter = str_to_fix[3]
    day = str_to_fix[7]
    return f"rat{letter}day{day}"

