import numpy as np
import pandas as pd
import subjects
from neuropy.analyses import Pf1D
from neuropy.analyses import Decode1d
from neuropy.core import Epoch

sessions = subjects.remaze_sess()


for sub, sess in enumerate(sessions):
    print(sess.animal.name)
    maze = sess.paradigm["maze"].flatten()
    post = sess.paradigm["post"].flatten()
    remaze = sess.paradigm["re-maze"].flatten()
    neurons = sess.neurons.get_neuron_type(neuron_type="pyr")
    position = sess.maze
    pbe_epochs = sess.pbe.time_slice(remaze[0], remaze[1])
    print(f"#Epochs: {len(pbe_epochs)}")
    metadata = {"score_method": "wcorr"}

    score_df = []
    for direction in ["up", "down"]:
        pf = Pf1D(
            neurons=neurons,
            position=position,
            sigma=4,
            grid_bin=2,
            epochs=sess.maze_run[direction],
            frate_thresh=0.5,
        )
        pf_neurons = neurons.get_by_id(pf.neuron_ids)
        decode = Decode1d(
            neurons=pf_neurons,
            ratemap=pf,
            epochs=pbe_epochs,
            bin_size=0.02,
            score_method="wcorr",
            n_jobs=6,
        )
        decode.calculate_shuffle_score(method="neuron_id", n_iter=1000)
        df = pd.DataFrame(
            dict(
                score=decode.score,
                percentile_score=decode.percentile_score,
                sequence_score=decode.sequence_score,
                replay_order=np.where(decode.score >= 0, "f", "r"),
            )
        )

        score_df.append(df.add_prefix(direction + "_"))
        metadata[direction + "_posterior"] = decode.posterior
        metadata[direction + "_shuffle_score"] = decode.shuffle_score

    score_df = pd.concat(score_df, axis=1)
    new_epochs = pbe_epochs.add_dataframe(score_df)
    new_epochs.metadata = metadata
    new_epochs.save(sess.filePrefix.with_suffix(".remaze_replay_pbe"))
