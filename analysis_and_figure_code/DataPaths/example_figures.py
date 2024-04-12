import numpy as np
from neuropy.utils.signal_process import filter_sig
from neuropy import core
from neuropy.analyses import Pf1D
from neuropy import plotting


def place_fields(
    neurons: core.Neurons,
    signal: core.Signal,
    position: core.Position,
    run_epochs: core.Epoch,
    run_period,
    replay_period,
):
    pf = Pf1D(
        neurons=neurons,
        position=position,
        sigma=4,
        grid_bin=2,
        epochs=run_epochs,
        frate_thresh=0.5,
    )

    neurons_sorted = neurons[pf.get_sort_order("neuron_id")].time_slice(*run_period)
    pos_epoch = position.time_slice(*run_period)

    fig = plotting.Fig(grid=(5, 6))

    sig_ax = fig.subplot(fig.gs[0, 1:6])
    sig_ax.plot(signal.time, signal.traces[0], "k", lw=0.2)
    sig_ax.set_xlim(run_period)
    sig_ax.axis("off")

    rpl_signal = filter_sig.bandpass(signal, lf=150, hf=250)
    rpl_ax = fig.subplot(fig.gs[1, 1:6], sharex=sig_ax)
    rpl_ax.plot(signal.time, rpl_signal.traces[0], "#4a77bae8", lw=0.1)
    rpl_ax.set_xlim(run_period)
    rpl_ax.axis("off")

    ax = fig.subplot(fig.gs[2:6, 1:6], sharex=sig_ax)
    plotting.plot_raster(neurons_sorted, color="cool_r", ax=ax, markersize=2)

    ax2 = ax.twinx()

    ax2.plot(pos_epoch.time, pos_epoch.x, color="#a3a3a3", alpha=0.7, lw=3)
    ax2.set_yticks([])

    xticks = np.arange(run_period[0], run_period[1], 5)
    ax.set_xticks(xticks, xticks - xticks[0])
    ax.tick_params(rotation=0)
    ax.set_yticks([0, len(neurons_sorted)])

    replay_ax = fig.subplot(fig.gs[2:6, 0])
    plotting.plot_raster(
        neurons_sorted.time_slice(11420.22, 11420.5), ax=replay_ax, color="cool_r"
    )
    replay_ax.axis("off")

    sig_ax = fig.subplot(fig.gs[0, 0], sharex=replay_ax)
    signal = signal.time_slice(None, *replay_period)
    sig_ax.plot(signal.time, signal.traces[0], "k", lw=0.5)
    sig_ax.axis("off")

    rpl_ax = fig.subplot(fig.gs[1, 0], sharex=replay_ax)
    replay_sig = rpl_signal.time_slice(None, *replay_period)
    rpl_ax.plot(
        replay_sig.time,
        rpl_signal.time_slice(None, *replay_period).traces[0],
        "#4a77bae8",
        lw=0.5,
    )
    # rpl_ax.set_xlim(run_epoch)
    rpl_ax.axis("off")
