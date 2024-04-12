import numpy as np
from bokeh import plotting as bplotting
from neuropy.analyses.reactivation import NeuronEnsembles
from bokeh.core.properties import AngleSpec
from bokeh.layouts import gridplot, column
from bokeh.transform import linear_cmap
from bokeh.palettes import tol, plasma
from bokeh.models import ColumnarDataSource, CustomJS, Slider


def plot_neuron_ensembles(ensembles: NeuronEnsembles, act_t=None):

    p1 = bplotting.figure(width=1000, height=500)
    p1.xgrid.grid_line_color = None
    p1.ygrid.grid_line_color = None

    t_start, t_stop = ensembles.t_start, ensembles.t_stop
    neurons = ensembles.neurons.time_slice(t_start, t_stop)
    for i, spktrn in enumerate(neurons.spiketrains):
        p1.dash(spktrn, (i + 1) * np.ones(len(spktrn)), angle=np.pi / 2, color="gray")

    if act_t is not None:
        act_source = {str(k): act_t[k] for k in range(act_t.shape[0])}
        callback = CustomJS(
            args=dict(s1=act_source, s2=act_source),
            code="""
            const f = cb_obj.value
            const x = source.data.x
            // let extractColumn = (arr, column) => arr.map(x=>x[column])
            const y = s2.data[f]
            source.data = { x, y }
        """,
        )

        slider = Slider(start=0, end=9, value=1, step=1, title="power")
        slider.js_on_change("value", callback)

        activation, t = ensembles.get_activation(*act_t)
        p2 = bplotting.figure(width=1000, height=200, x_range=p1.x_range)
        # cmap = linear_cmap(field_name='y', palette="Spectral6", low=min(y), high=max(y))
        colors = plasma(activation.shape[0])
        for i, y in enumerate(activation):
            p2.line(x=t, y=y, line_color=colors[i])

        # gridplot(p1, p2)
        bplotting.show(column(p1, p2))
    # bplotting.show(p)
