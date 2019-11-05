import multiprocessing
import os
import pickle
import sys

sys.path.append("../")

import exoplanet as xo
import numpy as np
import pymc3 as pm
import theano.tensor as T
from joblib import Parallel, delayed
from matplotlib import pyplot as plt

import caustic as ca
from utils import find_alert_time, run_sampling, run_optimizer
from models import DefaultModel, DefaultModelUniformPriors

num_cores = multiprocessing.cpu_count()

np.random.seed(42)

print("Number of cores", num_cores)


def fit_maximum_likelihood_model(event, output_dir, model_name):
    n_bands = len(event.light_curves)

    # Define output directories
    output_dir = output_dir + event.event_name + f"/{model_name}/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save data object
    with open(output_dir + "/data.pkl", "wb") as output:
        pickle.dump(event, output)

    # Save data plot
    fig, ax = plt.subplots(figsize=(15, 5))
    event.plot(ax)
    plt.savefig(output_dir + "/data.png")

    # Initialize model
    alert_time = find_alert_time(event)
    model = DefaultModelUniformPriors(event)

    # Fit MAP model
    map_params, ll_final = run_optimizer(model, return_logl=True)

    if not np.isnan(ll_final):
        # Save MAP parameters to file
        with open(output_dir + "/MAP_params.pkl", "wb") as output:
            pickle.dump(map_params, output)

    with model:
        # Plot model
        t_dense = np.tile(np.linspace(model.t_min, model.t_max, 1500), (n_bands, 1))
        t_dense_tensor = T.as_tensor_variable(t_dense)

        ## Compute the trajectory of the lens
        trajectory = ca.trajectory.Trajectory(
            event, alert_time + T.exp(model.ln_delta_t0), model.u0, model.tE
        )
        u_dense = trajectory.compute_trajectory(t_dense_tensor)

        ## Compute the magnification
        mag_dense = (u_dense ** 2 + 2) / (u_dense * T.sqrt(u_dense ** 2 + 4))

        ## Compute the mean model
        F_base = 10 ** (-(model.m_b - 22.0) / 2.5)
        mean_dense = model.f * F_base * mag_dense + (1 - model.f) * F_base

        fig, ax = plt.subplots(
            2, 1, gridspec_kw={"height_ratios": [3, 1]}, figsize=(10, 8), sharex=True
        )

        # Plot MAP model
        ca.plot_map_model_and_residuals(
            ax, event, model, map_params, t_dense_tensor, mean_dense
        )

        plt.savefig(output_dir + "/map_model.png")


def fit_default_model(event, output_dir, model_name):
    n_bands = len(event.light_curves)

    # Define output directories
    output_dir = output_dir + event.event_name + f"/{model_name}/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save data object
    with open(output_dir + "/data.pkl", "wb") as output:
        pickle.dump(event, output)

    # Save data plot
    fig, ax = plt.subplots(figsize=(15, 5))
    event.plot(ax)
    plt.savefig(output_dir + "/data.png")

    # Initialize model
    alert_time = find_alert_time(event)
    model = DefaultModel(event)

    # Sample the model
    trace = run_sampling(model, output_dir, ncores=4, nchains=4)

    if trace is None:
        return None

    with model:
        # Plot model
        t_dense = np.tile(np.linspace(model.t_min, model.t_max, 1500), (n_bands, 1))
        t_dense_tensor = T.as_tensor_variable(t_dense)

        ## Compute the trajectory of the lens
        trajectory = ca.trajectory.Trajectory(
            event, alert_time + T.exp(model.ln_delta_t0), model.u0, model.tE
        )
        u_dense = trajectory.compute_trajectory(t_dense_tensor)

        ## Compute the magnification
        mag_dense = (u_dense ** 2 + 2) / (u_dense * T.sqrt(u_dense ** 2 + 4))

        ## Compute the mean model
        F_base = 10 ** (-(model.m_b - 22.0) / 2.5)
        mean_dense = model.f * F_base * mag_dense + (1 - model.f) * F_base

        fig, ax = plt.subplots(
            2, 1, gridspec_kw={"height_ratios": [3, 1]}, figsize=(10, 8), sharex=True
        )

        # Plot posterior model
        ca.plot_model_and_residuals(
            ax, event, model, trace, t_dense_tensor, mean_dense, n_samples=50
        )

        plt.savefig(output_dir + "/model.png")

        # Save posterior plot to file
        pm.plot_posterior(trace, figsize=(12, 12))
        plt.savefig(output_dir + "/posterior.png")


exclude_2002 = [
    "018",
    "023",
    "040",
    "051",
    "068",
    "069",
    "077",
    "080",
    "081",
    "099",
    "113",
    "119",
    "126",
    "127",
    "128",
    "129",
    "131",
    "135",
    "143",
    "149",
    "159",
    "175",
    "194",
    "202",
    "203",
    "205",
    "215",
    "228",
    "229",
    "232",
    "238",
    "254",
    "255",
    "256",
    "266",
    "273",
    "307",
    "315",
    "339",
    "348",
    "360",
]


exclude_2003 = [
    "145",
    "160",
    "168",
    "170",
    "176",
    "192",
    "200",
    "230",
    "236",
    "252",
    "260",
    "266",
    "267",
    "271",
    "282",
    "286",
    "293",
    "303",
    "306",
    "311",
    "359",
    "380",
    "419",
    "188",
    "197",
    "245",
    "263",
    "274",
    "297",
    "387",
    "399",
    "407",
    "412",
    "413",
    "417",
    "420",
    "422",
    "429",
    "430",
    "432",
    "433",
    "435",
    "437",
    "440",
    "441",
    "442",
    "443",
    "444",
    "449",
    "450",
    "452",
    "453",
    "454",
    "455",
    "457",
    "459",
    "461",
    "462",
]
dirs_all = []

## Load training sample light curves from Albrow (2002)
# data_path = "../../../../../scratch/astro/fb90/OGLE_ews/2003/"
#
# for directory in os.listdir(data_path):
#    path = data_path + directory
#
#    if directory[-3:] not in exclude_2003:
#        dirs_all.append(data_path + directory)

# Split directories into 4 different parts, for parallelization
# split_dirs = np.array_split(dirs_all, 4)
#
# dirs = split_dirs[int(sys.argv[1])]
dirs = dirs_all

# n_jobs = 32
# result = Parallel(n_jobs)(
#    delayed(fit_default_model)(
#        ca.data.OGLEData(directory),
#        "../../output_data.nosync/output/ogle/2002",
#        "model_default",
#    )
#    for directory in dirs
# )


fit_maximum_likelihood_model(
    ca.data.OGLEData("../../../data.nosync/OGLE_ews/2003/blg-208"),
    "../../output_data.nosync/output/ogle/2003/",
    "model_maxl",
)
