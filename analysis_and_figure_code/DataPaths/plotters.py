import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statannotations
from statannotations.Annotator import Annotator
from statannotations.stats.StatTest import StatTest
from subjects import colors_sd, stat_kw
from typing import Unpack, TypedDict, Union
from dataclasses import dataclass
import pandas as pd
import itertools
from subjects import adjust_lightness
from copy import deepcopy

class Plotter:
    def __init__(
        self, data, x, y, hue=None, hue_order=None, xtick_rot=30, ax=None
    ) -> None:
        """Initiates data format for plotting

        Parameters
        ----------
        data : _type_
            _description_
        x : _type_
            _description_
        y : _type_
            _description_
        hue : _type_, optional
            _description_, by default None
        hue_order : _type_, optional
            _description_, by default None
        xtick_rot : int, optional
            _description_, by default 30
        ax : _type_, optional
            _description_, by default None
        """
        if ax is None:
            ax = plt.gca()

        self.plot_kw = dict(data=data, x=x, y=y, hue=hue, hue_order=hue_order, ax=ax)

        # Set up rotation - note that to ensure right justification of all text,
        # we will need to re-set some of these properties again for each plot
        self.xtick_rot = xtick_rot
        if xtick_rot is not None:
            ax.tick_params(axis="x", labelrotation=xtick_rot)

    def violinplot_sd(self, split=True, palette=None, scale="width", **kwargs):
        sns.violinplot(
            **self.plot_kw,
            split=split,
            inner="quartile",
            linewidth=0,
            palette=palette,
            saturation=1,
            cut=False,
            scale=scale,
            **kwargs,
        )
        for p in self.plot_kw["ax"].lines:
            p.set_linestyle("-")
            p.set_linewidth(0.5)  # Sets the thickness of the quartile lines
            p.set_color("white")  # Sets the color of the quartile lines
            p.set_alpha(1)

        self.plot_kw["ax"].legend("", frameon=False)

        # Right align xticklabels
        xlabels = self.plot_kw["ax"].get_xticklabels()
        self.plot_kw["ax"].set_xticklabels(xlabels,
                                           rotation=self.xtick_rot,
                                           ha="right")

        return self

    def boxplot_sd(self, palette=None, sep=0.9, inherit_whisker_color=False, **kwargs):
        ax = sns.boxplot(
            **self.plot_kw,
            linewidth=0,
            palette=palette,
            saturation=1,
            showfliers=False,
            medianprops=dict(color="white", linewidth=0.6, solid_capstyle="butt"),
            boxprops=dict(edgecolor="w", linewidth=0),
            whiskerprops=dict(color="k", linewidth=0.5, solid_capstyle="butt"),
            showcaps=True,
            capprops=dict(color="k"),
            # capwidths=0.2,
            **kwargs,
        )

        if inherit_whisker_color:
            box_patches = [
                patch for patch in ax.patches if type(patch) == mpl.patches.PathPatch
            ]

            n_boxes = len(box_patches)
            lines_per_box = len(ax.lines) // n_boxes

            for i, patch in enumerate(box_patches):
                box_color = patch.get_facecolor()

                # whisker_lines = [
                #     ax.lines[i * lines_per_box],
                #     ax.lines[i * lines_per_box+1],
                # ]

                whisker_lines = ax.lines[i * lines_per_box : (i + 1) * lines_per_box]
                whisker_lines = [whisker_lines[0], whisker_lines[1]]

                for line in whisker_lines:
                    line.set_color(box_color)

        for c in ax.get_children():
            # searching for PathPatches
            if isinstance(c, mpl.patches.PathPatch):
                # getting current width of box:
                p = c.get_path()
                verts = p.vertices
                verts_sub = verts[:-1]
                xmin = np.min(verts_sub[:, 0])
                xmax = np.max(verts_sub[:, 0])
                xmid = 0.5 * (xmin + xmax)
                xhalf = 0.5 * (xmax - xmin)

                # setting new width of box
                xmin_new = xmid - sep * xhalf
                xmax_new = xmid + sep * xhalf
                verts_sub[verts_sub[:, 0] == xmin, 0] = xmin_new
                verts_sub[verts_sub[:, 0] == xmax, 0] = xmax_new

                # setting new width of median line
                for l in ax.lines:
                    if np.all(l.get_xdata() == [xmin, xmax]):
                        l.set_xdata([xmin_new, xmax_new])

        yticks = ax.get_yticks()
        ax.set_yticks(yticks)

        self.plot_kw["ax"].legend("", frameon=False)

        # Right align xticklabels
        xlabels = self.plot_kw["ax"].get_xticklabels()
        self.plot_kw["ax"].set_xticklabels(xlabels,
                                           rotation=self.xtick_rot,
                                           ha="right")

        return self

    def boxplot_sd2(self, palette=None, sep=0.9, zorder=2, **kwargs):
        ax = sns.boxplot(
            **self.plot_kw,
            linewidth=0.4,
            palette=palette,
            saturation=1,
            showfliers=False,
            **kwargs,
        )

        box_patches = [
            patch for patch in ax.patches if type(patch) == mpl.patches.PathPatch
        ]

        n_boxes = len(box_patches)
        lines_per_box = len(ax.lines) // n_boxes

        for i, patch in enumerate(box_patches):
            box_color = patch.get_facecolor()
            patch.set_facecolor("None")
            patch.set_edgecolor(box_color)
            patch.set_zorder(zorder)

            # whisker_lines = [
            #     ax.lines[i * lines_per_box],
            #     ax.lines[i * lines_per_box+1],
            # ]

            whisker_lines = ax.lines[i * lines_per_box : (i + 1) * lines_per_box]
            # whisker_lines = [whisker_lines[0], whisker_lines[1]]

            for line in whisker_lines:
                line.set_color(box_color)
                line.set_zorder(zorder)

        for c in ax.get_children():
            # searching for PathPatches
            if isinstance(c, mpl.patches.PathPatch):
                # getting current width of box:
                p = c.get_path()
                verts = p.vertices
                verts_sub = verts[:-1]
                xmin = np.min(verts_sub[:, 0])
                xmax = np.max(verts_sub[:, 0])
                xmid = 0.5 * (xmin + xmax)
                xhalf = 0.5 * (xmax - xmin)

                # setting new width of box
                xmin_new = xmid - sep * xhalf
                xmax_new = xmid + sep * xhalf
                verts_sub[verts_sub[:, 0] == xmin, 0] = xmin_new
                verts_sub[verts_sub[:, 0] == xmax, 0] = xmax_new

                # setting new width of median line
                for l in ax.lines:
                    if np.all(l.get_xdata() == [xmin, xmax]):
                        l.set_xdata([xmin_new, xmax_new])

        self.plot_kw["ax"].legend("", frameon=False)

        # Right align xticklabels
        xlabels = self.plot_kw["ax"].get_xticklabels()
        self.plot_kw["ax"].set_xticklabels(xlabels,
                                           rotation=self.xtick_rot,
                                           ha="right")

        return self

    def stripbarplot_sd(self, palette=None, dodge=True, **kwargs):
        ax = sns.stripplot(
            **self.plot_kw,
            palette=palette,
            edgecolor="w",
            linewidth=0.3,
            size=3,
            dodge=dodge,
            **kwargs,
        )
        sns.barplot(
            **self.plot_kw,
            ci=None,
            facecolor="w",
            linewidth=0.5,
            edgecolor=".2",
            **kwargs,
        )
        ax.legend("", frameon=False)

        return self

    def stripbarlineplot_sd(self, palette=None, dodge=0.1, **kwargs):
        ax = sns.lineplot(
            **self.plot_kw,
            palette=palette,
            units="session",
            estimator=None,
            lw=0.25,
            alpha=0.25,
            # sort=True,
            # **kwargs,
        )
        sns.stripplot(
            **self.plot_kw,
            palette=palette,
            edgecolor="w",
            linewidth=0.2,
            size=2,
            dodge=dodge,
            jitter=True,
            # alpha=0.8,
            # **kwargs,
        )
        sns.barplot(
            **self.plot_kw,
            ci=None,
            facecolor="w",
            linewidth=0.5,
            edgecolor=".2",
            # **kwargs,
        )

        lines = [line for line in ax.lines if line.get_xdata().size > 0]
        n_lines_per_hue = len(lines) // 2

        # for line in lines[:8]:
        #     x = line.get_xdata()
        #     line.set_xdata(x - 0.2)

        # for line in lines[8:]:
        #     x = line.get_xdata()
        #     line.set_xdata(x + 0.2)

        if len(palette) == 2:
            for line in ax.lines:
                line_color = line.get_color()
                lw = line.get_linewidth()
                x = line.get_xdata()
                if lw == 0.25:
                    if line_color == palette[0]:
                        line.set_xdata(x - 0.2)
                        # line.set_zorder(zorder - 10)

                    if line_color == palette[1]:
                        line.set_xdata(x + 0.2)
                        # line.set_zorder(zorder - 10)

        self.plot_kw["ax"].legend("", frameon=False)

        # Right align xticklabels
        xlabels = self.plot_kw["ax"].get_xticklabels()
        self.plot_kw["ax"].set_xticklabels(xlabels,
                                           rotation=self.xtick_rot,
                                           ha="right")

        return self

    def striplineplot_sd(self, palette=None, dodge=0.1, zorder=2, **kwargs):
        ax = sns.lineplot(
            **self.plot_kw,
            palette=palette,
            units="session",
            estimator=None,
            lw=0.25,
            alpha=0.2,
            # sort=True,
            # **kwargs,
        )
        sns.stripplot(
            **self.plot_kw,
            palette=palette,
            edgecolor="w",
            linewidth=0.2,
            size=2,
            dodge=dodge,
            jitter=True,
            zorder=zorder,
            # alpha=0.5,
            **kwargs,
        )

        if palette is not None:
            if len(palette) == 1:
                color = palette[0]
                for line in ax.lines:
                    if line.get_linewidth() == 0.25:
                        line.set_color(color)
                        line.set_zorder(zorder - 10)
            if len(palette) == 2:
                for line in ax.lines:
                    line_color = line.get_color()
                    lw = line.get_linewidth()
                    x = line.get_xdata()
                    if lw == 0.25:
                        if line_color == palette[0]:
                            line.set_xdata(x - 0.2)
                            line.set_zorder(zorder - 10)

                        if line_color == palette[1]:
                            line.set_xdata(x + 0.2)
                            line.set_zorder(zorder - 10)

        self.plot_kw["ax"].legend("", frameon=False)

        # Right align xticklabels
        xlabels = self.plot_kw["ax"].get_xticklabels()
        self.plot_kw["ax"].set_xticklabels(xlabels,
                                           rotation=self.xtick_rot,
                                           ha="right")

        return self

    def stat_anot_sd(
        self,
        stat_within=None,
        stat_across=None,
        alpha_within: float or list = 0.05,
        alpha_across: float or list = 0.05,
        alpha_trend_within: float or None = 0.05,
        alpha_trend_across: float or None = 0.10,
        fontsize=8,
        verbose=False,
        **kwargs,
    ):
        ax = self.plot_kw["ax"]
        stat_kw["fontsize"] = fontsize

        results_list = []
        if stat_across is not None:
            # stat_kw["pvalue_thresholds"] = [[alpha_across, "*"], [1, "ns"]]
            alpha_across = [alpha_across] if isinstance(alpha_across, float) else alpha_across
            pval_thresh = [[a, '*' * (ida + 1)] for ida, a in enumerate(alpha_across)]
            if alpha_trend_across is not None:
                pval_thresh.append([alpha_trend_across, "#"])
            pval_thresh.append([1, "ns"])
            stat_kw["pvalue_thresholds"] = pval_thresh
            ax2 = ax.inset_axes([0, 0.9, 1, 0.15])
            self.plot_kw["ax"] = ax2
            ax2.set_axis_off()

            orders = self.plot_kw["data"].zt.unique()
            if callable(stat_across):
                custom_long_name = stat_across.__name__
                custom_short_name = stat_across.__name__
                stat_across = StatTest(stat_across, custom_long_name, custom_short_name)

            # Across groups
            pairs = [((_, "NSD"), (_, "SD")) for _ in orders]
            if ("0-2.5" in orders) & ("5-7.5" in orders):
                pairs = pairs + [(("0-2.5", "NSD"), ("5-7.5", "SD"))]
            elif ("ZT 0-2.5" in orders) & ("ZT 5-7.5" in orders):
                pairs = pairs + [(("ZT 0-2.5", "NSD"), ("ZT 5-7.5", "SD"))]
            elif ("0-1" in orders) & ("5-6" in orders):
                pairs = pairs + [(("0-1", "NSD"), ("5-6", "SD"))]
            elif ("ZT 0-1" in orders) & ("ZT 5-6" in orders):
                pairs = pairs + [(("ZT 0-1", "NSD"), ("ZT 5-6", "SD"))]
            annotator = Annotator(pairs=pairs, **self.plot_kw, order=orders)
            annotator.configure(test=stat_across, **stat_kw, color="k", verbose=verbose)
            annotator.apply_and_annotate()
            results_list.append(get_stats_from_annotation(annotator.annotations, **kwargs))
            annotator.reset_configuration()

        if stat_within is not None:
            # stat_kw["pvalue_thresholds"] = [[alpha_within, "*"], [1, "ns"]]
            alpha_within = [alpha_within] if isinstance(alpha_within, float) else alpha_within
            pval_thresh = [[a, '*' * (ida + 1)] for ida, a in enumerate(alpha_within)]
            if alpha_trend_within is not None:
                pval_thresh.append([alpha_trend_within, "#"])
            pval_thresh.append([1, "ns"])
            stat_kw["pvalue_thresholds"] = pval_thresh
            orders = self.plot_kw["data"].zt.unique()
            if callable(stat_within):
                custom_long_name = stat_within.__name__
                custom_short_name = stat_within.__name__
                stat_within = StatTest(stat_within, custom_long_name, custom_short_name)

            # Within groups
            yshift = 0
            for i, g in enumerate(["NSD", "SD"]):
                ax2 = ax.inset_axes([0.1, 1.1 + yshift, 0.9, 0.3])
                ax2.axis("off")
                self.plot_kw["ax"] = ax2

                # pairs = [
                #     (("0-2.5", g), ("2.5-5", g)),
                #     (("2.5-5", g), ("5-7.5", g)),
                #     (("0-2.5", g), ("5-7.5", g)),
                # ]
                order_pairs = list(itertools.combinations(orders, 2))
                pairs = [((p[0], g), (p[1], g)) for p in order_pairs]
                annotator = Annotator(pairs=pairs, **self.plot_kw, order=orders)
                annotator.configure(
                    test=stat_within, **stat_kw, color=colors_sd(1)[i], verbose=verbose
                )
                annotator.apply_and_annotate()
                results_list.append(get_stats_from_annotation(annotator.annotations, **kwargs))
                annotator.reset_configuration()
                yshift += 0.3

        return results_list

    def correct_order(self, order):
        try:
            if "0-1" in order:
                req_order = ["PRE", "MAZE", "0-1", "4-5", "5-6"]
                indx = np.array([req_order.index(i) for i in order])
                sort_indx = np.argsort(indx)
            else:
                req_order = ["PRE", "MAZE", "0-2.5", "2.5-5", "5-7.5"]
                indx = np.array([req_order.index(i) for i in order])
                sort_indx = np.argsort(indx)
        except ValueError:  # Fix if prepending "ZT" to each post epoch
            if "ZT 0-1" in order:
                req_order = ["PRE", "MAZE", "ZT 0-1", "ZT 4-5", "ZT 5-6"]
                indx = np.array([req_order.index(i) for i in order])
                sort_indx = np.argsort(indx)
            else:
                req_order = ["PRE", "MAZE", "ZT 0-2.5", "ZT 2.5-5", "ZT 5-7.5"]
                indx = np.array([req_order.index(i) for i in order])
                sort_indx = np.argsort(indx)

        return list(np.array(order)[sort_indx])

    def stat_anot(
        self,
        stat_within,
        alpha_within: float or list=0.05,
        alpha_trend: float or None = 0.1,
        stat_unequal=None,
        fontsize=8,
        verbose=False,
        **kwargs,  # to get_stats_from_annotations
    ):
        stat_kw["fontsize"] = fontsize
        ax = self.plot_kw["ax"]
        alpha_within = [alpha_within] if isinstance(alpha_within, float) else alpha_within
        pval_thresh = [[a, '*' * (ida + 1)] for ida, a in enumerate(alpha_within)]
        if alpha_trend is not None:
            pval_thresh.append([alpha_trend, "#"])

        pval_thresh.append([1, "ns"])
        stat_kw["pvalue_thresholds"] = pval_thresh
        orders = self.correct_order(self.plot_kw["data"].zt.unique())
        order_pairs = list(itertools.combinations(orders, 2))

        ax2 = ax.inset_axes([0, 0.9, 1, 0.6])
        ax2.axis("off")
        self.plot_kw["ax"] = ax2

        results_list = []
        if stat_unequal is not None:
            data = self.plot_kw["data"]
            unequal_bool = np.zeros(len(order_pairs)).astype("bool")
            for i, pair in enumerate(order_pairs):
                if len(data[data.zt == pair[0]]) != len(data[data.zt == pair[1]]):
                    unequal_bool[i] = True

            if callable(stat_unequal):
                custom_long_name = stat_unequal.__name__
                custom_short_name = stat_unequal.__name__
                stat_unequal = StatTest(stat_unequal, custom_long_name, custom_short_name)

            unequal_pairs = [
                order_pairs[i] for i, cond in enumerate(unequal_bool) if cond == True
            ]

            if len(unequal_pairs) > 0:
                annotator = Annotator(
                    pairs=unequal_pairs,
                    **self.plot_kw,
                    order=orders,
                )
                annotator.configure(
                    test=stat_unequal, **stat_kw, color="k", verbose=verbose
                )
                annotator.apply_and_annotate()
                results_list.append(get_stats_from_annotation(annotator.annotations, **kwargs))
                annotator.reset_configuration()

                order_pairs = [
                    order_pairs[i] for i, cond in enumerate(unequal_bool) if cond == False
                ]

        if callable(stat_within):
            custom_long_name = stat_within.__name__
            custom_short_name = stat_within.__name__
            stat_within = StatTest(stat_within, custom_long_name, custom_short_name)

        if len(order_pairs) > 0:
            # order_pairs = list(itertools.combinations(orders, 2))
            # pairs = [(p[0], p[1]) for p in order_pairs]
            annotator = Annotator(pairs=order_pairs, **self.plot_kw, order=orders)
            annotator.configure(test=stat_within, **stat_kw, color="k", verbose=verbose)
            annotator.apply_and_annotate()
            results_list.append(get_stats_from_annotation(annotator.annotations, **kwargs))
            annotator.reset_configuration()

        return results_list

    def _remove_legend(self):
        self.plot_kw["ax"].legend("", frameon=False)

    def areaplot(self, alpha=0.5, **kwargs):
        sns.lineplot(**self.plot_kw, **kwargs)
        ax = self.plot_kw["ax"]
        for line in ax.lines:
            x, y = line.get_xydata().T
            ax.fill_between(x, 0, y, color=line.get_color(), alpha=alpha, ec=None)

        self._remove_legend()


def get_nsd_vs_sd_df(data: pd.DataFrame, block_size: float or int in [1, 2.5] = 2.5):
    zt_prepend = ('ZT 0-2.5' in data.zt.unique()) or ('ZT 0-1' in data.zt.unique())
    if block_size == 2.5:
        if zt_prepend:
            df = data[data.zt.isin(["ZT 0-2.5", "ZT 2.5-5", "ZT 5-7.5"])].copy()
            df.drop(df[(df.zt == "ZT 0-2.5") & (df.grp == "SD")].index, inplace=True)
            df.drop(df[(df.zt == "ZT 5-7.5") & (df.grp == "NSD")].index, inplace=True)
            df.loc[(df.zt == "ZT 0-2.5") & (df.grp == "NSD"), "zt"] = "ZT 0-2.5 vs 5-7.5"
            df.loc[(df.zt == "ZT 5-7.5") & (df.grp == "SD"), "zt"] = "ZT 0-2.5 vs 5-7.5"
        else:
            df = data[data.zt.isin(["0-2.5", "2.5-5", "5-7.5"])].copy()
            df.drop(df[(df.zt == "0-2.5") & (df.grp == "SD")].index, inplace=True)
            df.drop(df[(df.zt == "5-7.5") & (df.grp == "NSD")].index, inplace=True)
            df.loc[(df.zt == "0-2.5") & (df.grp == "NSD"), "zt"] = "0-2.5 vs 5-7.5"
            df.loc[(df.zt == "5-7.5") & (df.grp == "SD"), "zt"] = "0-2.5 vs 5-7.5"
    else:  # For 1h blocks
        if zt_prepend:
            df = data[data.zt.isin(["ZT 0-1", "ZT 4-5", "ZT 5-6"])].copy()
            df.drop(df[(df.zt == "ZT 0-1") & (df.grp == "SD")].index, inplace=True)
            df.drop(df[(df.zt == "ZT 5-6") & (df.grp == "NSD")].index, inplace=True)
            df.loc[(df.zt == "ZT 0-1") & (df.grp == "NSD"), "zt"] = "ZT 0-1 vs 5-6"
            df.loc[(df.zt == "ZT 5-6") & (df.grp == "SD"), "zt"] = "ZT 0-1 vs 5-6"
        else:
            df = data[data.zt.isin(["0-1", "4-5", "5-6"])].copy()
            df.drop(df[(df.zt == "0-1") & (df.grp == "SD")].index, inplace=True)
            df.drop(df[(df.zt == "5-6") & (df.grp == "NSD")].index, inplace=True)
            df.loc[(df.zt == "0-1") & (df.grp == "NSD"), "zt"] = "0-1 vs 5-6"
            df.loc[(df.zt == "5-6") & (df.grp == "SD"), "zt"] = "0-1 vs 5-6"

    return df


def get_stats_from_annotation(annotation_in, prepend=()):
    """Grabs stats from a list of annotator objects and puts them into csv from annotator
    set prepend to add a list of items to the beginning of results"""
    results = []
    if isinstance(annotation_in, statannotations.Annotation.Annotation):
        cmp1 = annotation_in.structs[0]["label"]
        cmp2 = annotation_in.structs[1]["label"]
        stats_str = [aa.split(" ") for aa in annotation_in.formatted_output.split(":")]
        test_used = stats_str[0][0]
        pval_str = stats_str[1][0]
        test_stat = stats_str[1][1].split("=")[1]

        results.append([*prepend, cmp1, cmp2, test_used, pval_str, test_stat])
    elif isinstance(annotation_in, list):
        for a in annotation_in:
            results.append(get_stats_from_annotation(a, prepend=prepend)[0])

    return results


def get_nsd_vs_sd_df_by_state(data: pd.DataFrame):
    """"Parses dataframe for comparison between NSD 0-2.5h vs SD 5-7.5h NREM states
    and 2-5.5h NSD vs. SD WAKE states"""

    # Grab appropriate keys to use
    state_key = "brainstate" if "brainstate" in data.keys() else "state"
    wake_key = "WK" if "WK" in data[state_key].unique() else "WAKE"

    # Grab appropriately formatted ZT string
    nrem_drop_str, wake_drop_str = "", ""
    if "2.5-5" in data.zt.unique():
        nrem_drop_str, wake_drop_str = "2.5-5", "0-2.5 vs 5-7.5"
    elif "ZT 2.5-5" in data.zt.unique():
        nrem_drop_str, wake_drop_str = "ZT 2.5-5", "ZT 0-2.5 vs 5-7.5"
    else:
        ValueError("data input zt field is improperly formatted.")

    # Grab NREM and drop 2.5-5 hour session - leaves only 0-2.5 NSD and 5-7.5 SD
    df_nrem = get_nsd_vs_sd_df(data[data[state_key] == "NREM"])
    df_nrem.drop(df_nrem[(df_nrem.zt == nrem_drop_str)].index, inplace=True)

    # Grab WAKE and drop the 0-2.5 vs 5-7.5 designation
    df_wake = get_nsd_vs_sd_df(data[data[state_key] == wake_key])
    df_wake.drop(df_wake[df_wake.zt == wake_drop_str].index, inplace=True)

    return pd.concat((df_nrem, df_wake))


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

colors = ["#999897", "#f07067"] + ["#424242", "#eb4034"] * 2 + ["#424242", "#48bdf7"]


def violinplotx(
    data,
    x,
    y,
    ax=None,
    stat_anot=False,
    stat_test=None,
    color="k",
    xtick_rot=30,
    **kwargs,
):
    if ax is None:
        ax = plt.gca()

    plot_kw = dict(data=data, x=x, y=y, ax=ax)
    sns.violinplot(
        **plot_kw,
        inner="quartile",
        linewidth=0,
        color=color,
        # colors=colors,
        # scale="width",
        saturation=1,
        cut=True,
        **kwargs,
    )
    for p in ax.lines:
        p.set_linestyle("-")
        p.set_linewidth(0.5)  # Sets the thickness of the quartile lines
        p.set_color("white")  # Sets the color of the quartile lines
        p.set_alpha(1)

    ax.legend("", frameon=False)
    if xtick_rot is not None:
        ax.tick_params(axis="x", labelrotation=xtick_rot)
    return ax


def tn_violinplot(
    data,
    x,
    y,
    ax=None,
    stat_anot=False,
    stat_test=None,
    xtick_rot=30,
    split=True,
    **kwargs,
):
    if ax is None:
        ax = plt.gca()

    plot_kw = dict(data=data, x=x, y=y, ax=ax)
    sns.violinplot(
        **plot_kw,
        split=split,
        inner="quartile",
        linewidth=0,
        # palette=colors_sd(1),
        # colors=colors,
        # scale="width",
        saturation=1,
        cut=True,
        **kwargs,
    )
    for p in ax.lines:
        p.set_linestyle("-")
        p.set_linewidth(0.8)  # Sets the thickness of the quartile lines
        p.set_color("white")  # Sets the color of the quartile lines
        p.set_alpha(1)

    if stat_anot:
        orders = data[x].unique()
        if stat_test is None:
            stat_test = "t-test_welch"

        # Across groups
        pairs = [
            ("pre", "post1"),
            ("pre", "post2"),
            ("maze1", "maze2"),
            ("post1", "post2"),
        ]
        annotator = Annotator(pairs=pairs, **plot_kw, order=orders)
        annotator.configure(test=stat_test, **stat_kw, color="k")
        annotator.apply_and_annotate()
        annotator.reset_configuration()

    ax.legend("", frameon=False)
    if xtick_rot is not None:
        ax.tick_params(axis="x", labelrotation=xtick_rot)
    return ax


def stripplot(
    data,
    x,
    y,
    hue="grp",
    hue_order=["NSD", "SD"],
    ax=None,
    dodge=True,
    size=3,
    stat_anot=False,
    stat_test=None,
    xtick_rot=30,
    **kwargs,
):
    if ax is None:
        ax = plt.gca()

    plot_kw = dict(data=data, x=x, y=y, hue=hue, hue_order=hue_order, ax=ax)
    sns.stripplot(
        **plot_kw,
        palette=colors_sd(1),
        edgecolor="w",
        linewidth=0.3,
        size=size,
        dodge=dodge,
        **kwargs,
    )

    if stat_anot:
        orders = data.zt.unique()
        if stat_test is None:
            stat_test = "t-test_welch"

        # Across groups
        pairs = [((_, "NSD"), (_, "SD")) for _ in orders]
        annotator = Annotator(pairs=pairs, **plot_kw, order=orders)
        annotator.configure(test=stat_test, **stat_kw, color="#4AB33E")
        annotator.apply_and_annotate()
        annotator.reset_configuration()

    ax.legend("", frameon=False)
    if xtick_rot is not None:
        ax.tick_params(axis="x", labelrotation=xtick_rot)
        ax.set_xticklabels(ax.get_xticklabels, rotation=xtick_rot, ha="right")
    return ax

def add_zt_str(df: pd.DataFrame, zt_key="zt", epoch_str=("0-2.5", "2.5-5", "5-7.5")):
    """Fix zt strings to prepend ZT"""
    for epoch_name in epoch_str:
        df.loc[df[zt_key] == epoch_name, zt_key] = f"ZT {epoch_name}"

    return df
