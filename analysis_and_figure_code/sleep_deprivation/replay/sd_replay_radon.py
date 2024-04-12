import numpy as np
import pandas as pd
import subjects
from neuropy.analyses import Decode1d, Pf1D
from neuropy.core import Epoch

sessions = subjects.sd.pf_sess
# sessions = subjects.nsd.ratNday2
# sessions = subjects.sd.pf_sess[-3:-1]

for sub, sess in enumerate(sessions):
    print(sess.animal.name)

    neurons = sess.neurons_stable.get_neuron_type(["pyr", "mua"])
    neurons = neurons[neurons.firing_rate <= 10]

    position = sess.maze
    pbe_epochs = sess.pbe
    print(f"#Epochs: {len(pbe_epochs)}")
    metadata = {"score_method": ["wcorr"]}

    maze_run = sess.maze_run

    score_df = []
    for direction in ["up", "down"]:
        pf = Pf1D(
            neurons=neurons,
            position=position,
            sigma=4,
            grid_bin=2,
            epochs=maze_run[direction],
            frate_thresh=0.3,
        )
        pf_neurons = neurons.get_by_id(pf.neuron_ids)
        decode = Decode1d(
            neurons=pf_neurons, ratemap=pf, epochs=pbe_epochs, bin_size=0.02, n_jobs=6
        )
        wcorr, jd = decode.get_wcorr(jump_stat="max")
        # radon_score, velocity, intercept = decode.get_radon_transform(
        #     nlines=10000, margin=16
        # )
        shuffled_measures = decode.get_shuffled_wcorr(
            n_iter=1000, method="column_cycle", jump_stat="max"
        )

        df = pd.DataFrame(
            dict(
                wcorr=wcorr,
                jd=jd,
                # radon=radon_score,
                # vel=velocity,
                # intercept=intercept,
            )
        )

        score_df.append(df.add_prefix(direction + "_"))
        metadata[direction + "_posterior"] = decode.posterior
        metadata[direction + "_shuffle_measures"] = shuffled_measures

    score_df = pd.concat(score_df, axis=1)
    new_epochs = pbe_epochs.add_dataframe(score_df)
    new_epochs.metadata = metadata
    new_epochs.save(sess.filePrefix.with_suffix(".pbe.replay.mua.column_cycle.maxjd"))


# import numpy as np
# import pandas as pd
# import subjects
# from neuropy.analyses import Decode1d, Pf1D
# from neuropy.core import Epoch

# sessions = subjects.pf_sess()

# for sub, sess in enumerate(sessions):
#     print(sess.animal.name)

#     neurons = sess.neurons_stable.get_neuron_type(["pyr", "mua"])
#     neurons = neurons[neurons.firing_rate <= 10]

#     position = sess.maze
#     pbe_epochs = sess.pbe
#     print(f"#Epochs: {len(pbe_epochs)}")
#     metadata = {"score_method": "radon_transform", "nlines": 10000, "decode_margin": 16}

#     maze_run = sess.maze_run

#     score_df = []
#     for direction in ["up", "down"]:
#         pf = Pf1D(
#             neurons=neurons,
#             position=position,
#             sigma=4,
#             grid_bin=2,
#             epochs=maze_run[direction],
#             frate_thresh=0.3,
#         )
#         pf_neurons = neurons.get_by_id(pf.neuron_ids)
#         decode = Decode1d(
#             neurons=pf_neurons,
#             ratemap=pf,
#             epochs=pbe_epochs,
#             bin_size=0.02,
#             score_method="radon_transform",
#             radon_kw=dict(nlines=10000, decode_margin=16),
#             n_jobs=4,
#         )
#         # decode.calculate_shuffle_score(method="neuron_id", n_iter=1000)
#         df = pd.DataFrame(
#             dict(
#                 score=decode.score,
#                 velocity=decode.velocity,
#                 intercept=decode.intercept,
#             )
#         )

#         score_df.append(df.add_prefix(direction + "_"))
#         metadata[direction + "_posterior"] = decode.posterior
#         # metadata[direction + "_shuffle_score"] = decode.shuffle_score

#     score_df = pd.concat(score_df, axis=1)
#     new_epochs = pbe_epochs.add_dataframe(score_df)
#     new_epochs.metadata = metadata
#     new_epochs.save(sess.filePrefix.with_suffix(".pbe.radon.mua"))
