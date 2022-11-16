import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU') 
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True) # limit gpu memory
    
from psychrnn.tasks.task import Task
from psychrnn.backend.models.basic import Basic
from psychrnn.backend.models.lstm import LSTM
import pandas as pd
import numpy as np
from typing import *
import os
import json
from uuid import UUID
from pathlib import Path
import click


teena_data = pd.read_pickle("./teena_df.pickle")
teena_data = teena_data[teena_data["sf"] == True]
teena_data_array = np.dstack(teena_data["traj"])
traj_data = teena_data_array[:2500, :, :100]


class Reach2Grab(Task):
    def __init__(self, dt, tau, T, N_batch):
        super(Reach2Grab, self).__init__(1, 2, dt, tau, T, N_batch)

    def generate_trial_params(self, batch, trial):
        params = dict()
        params["trial_ix"] = np.random.randint(0, traj_data.shape[2])

        return params

    def trial_function(self, time, params):
        x_t = 1 # just a "go" cue
        y_t = traj_data[time, :, params["trial_ix"]]
        
        mask_t = np.ones(shape=y_t.shape, dtype=bool)
        # y_t returns a trajectory

        return x_t, y_t, mask_t

def save_model(m: Union[Basic, LSTM], u: Union[str, UUID]):
    d = Path(f"/home/kushalk/repos/rnn-go-brr/rnngobrr/models/{u}")
    weights_path = d.joinpath("weights")
    params_path = d.joinpath("params.json")
    os.mkdir(d)
    
    m.save(weights_path)
    json.dump(m.params, open(params_path, "w"))

    
@click.command()
@click.option("--uuid", type=str)
def main(uuid):
    df = pd.read_pickle("./models_dataframe.pickle")

    row = df[df["uuid"] == uuid].squeeze()

    u = row["uuid"]

    task = Reach2Grab(
        dt=row["dt"],
        tau=row["tau"],
        T=row["T"],
        N_batch=128
    )

    network_params = row["task_params"]
    network_params['name'] = row["uuid"]
    network_params['N_rec'] = row["N_rec"]

    model_iter = Basic(network_params)

    train_params = {
        "training_iters": row["training_iters"],
        "learning_rate": row["learning_rate"],
    }

    model_iter.train(task, train_params)
    save_model(model_iter, u)

if __name__ == "__main__":
    main()
