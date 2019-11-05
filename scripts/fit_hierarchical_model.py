import os
import pickle
import random
import sys

sys.path.append("../")

import exoplanet as xo
import matplotlib as mpl
import numpy as np
import pandas as pd
import pymc3 as pm
import theano.tensor as T
from matplotlib import pyplot as plt

import caustic as ca
from models import HierarchicalModel

random.seed(42)


def load_samples(data_path, params):
    samples_list = []
    samples_logp_list = []

    for directory in os.listdir(data_path):
        path = data_path + directory

        # Load trace and MAP parameters
        if os.path.exists(path + "/model_albrow/trace.csv"):
            trace = pd.read_csv(path + "/model_albrow/trace.csv")
            for param in params:
                if param not in trace.columns.tolist():
                    raise ValueError("Parameter not found in the trace file.")

            samples_list.append(trace[params].values)
            samples_logp_list.append(
                trace[["logp_" + param for param in params]].values
            )

        elif os.path.exists(path + "/model_albrow/trace_diverging.csv"):
            print("Divergent samples in directory ", directory)

        else:
            print("Trace not found in directory", directory)

    return np.stack(samples_list, axis=0), np.stack(samples_logp_list, axis=0)


def fit_model(samples, samples_logp, output_dir):
    # Convert samples to theano.tensor
    samples_tensor = T.as_tensor_variable(samples)  # for performance reasons
    samples_logp_tensor = T.as_tensor_variable(samples_logp)

    model = HierarchicalModel(samples_tensor, samples_logp_tensor)

    with model:
        # Print initial logps
        print(model.test_point)

        # Run sampling
        trace = pm.sample(tune=100, draws=1000, cores=4, step=xo.get_dense_nuts_step())

    # Save the samples to disk
    samples_hyper = np.stack(
        (
            trace["lam_ln_A0"],
            trace["mu_ln_delta_t0"],
            trace["sig_ln_delta_t0"],
            trace["mu_ln_tE"],
            trace["sig_ln_tE"],
            trace["alpha_f"],
            trace["beta_f"],
        )
    ).T

    np.save(output_dir + "samples_hyper.npy", samples_hyper)


# Load posterior samples for individual events
samples, samples_logp = load_samples(
    "../../../../scratch/astro/fb90/output/2002/",
    ["ln_A0", "ln_delta_t0", "ln_tE", "f"],
)

#  Fit model
fit_model(samples[:, ::5], samples_logp[:, ::5], "")  #  for performance reasons
