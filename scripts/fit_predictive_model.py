import logging
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
import pandas as pd

import caustic as ca

# from model_predictive import initialize_model
from utils import *
from models import PredictiveModel, PredictiveModelEmpiricalPriors

np.random.seed(42)


def fit_predictive_model(pm_model, event, output_dir, masking_time, sample=True):
    """
    Fits hierarchical predictive model to partially masked lightcurve. The
    data is fitted for 4 equal intervals in time between alert_time and t0.
    Return trace.
    """
    model_name = "blend"

    # Compute alert time
    alert_time = find_alert_time(event)

    # Update mask
    mask = event.light_curves[0]["HJD"] - 2450000 < masking_time
    event.light_curves[0]["mask"] = mask

    # Save mask to file
    np.save(output_dir + "/mask.npy", mask)

    #  Either sample or optimize the model
    if sample == True:
        trace = run_sampling(
            pm_model, output_dir, ncores=1, nchains=2, filename=f"trace"
        )

        if trace is None:
            print("Sampling failed, check that the model is not ill defined.")
            return None

    else:
        # Fit predictive model for MAP values
        map_params_pred, ll_final = run_optimizer(pm_model, return_logl=True)

        if np.isnan(ll_final):
            print("Optimization failed, final log likelihood is NaN.")
            return None

    # Create dense grid
    ti = event.light_curves[0]["HJD"][0] - 2450000
    tf = event.light_curves[0]["HJD"][-1] - 2450000
    t_dense = np.linspace(ti, tf, 4000)
    t_dense_tensor = T.as_tensor_variable(t_dense)

    t_test = np.array(event.light_curves[0]["HJD"][~mask] - 2450000)
    t_test_tensor = T.as_tensor_variable(t_test)

    if sample == True:
        # Evaluate the median hierarchical model, the MAP hierarchical model and the
        # Albrow model on a dense grid
        prediction = evaluate_prediction_quantiles(
            event, pm_model, trace, t_dense_tensor, alert_time
        )

        prediction_test = evaluate_prediction_quantiles(
            event, pm_model, trace, t_test_tensor, alert_time
        )

    else:
        # Evaluate the MAP model on a dense grid
        prediction = evaluate_map_prediction(
            event, pm_model, map_params_pred, t_dense_tensor, alert_time
        )

        # Evaluate MAP prediction on test set
        prediction_test = evaluate_map_prediction(
            event, pm_model, map_params_pred, t_test_tensor, alert_time
        )

    # Evaluate MSE
    event.units = "fluxes"
    fluxes_test = np.array(event.light_curves[0]["flux"][~mask])
    event.units = "magnitudes"

    # Compute residuals
    residuals_hierarchical = fluxes_test - prediction_test[1]

    # Save residuals to file
    np.save(output_dir + f"/residuals.npy", residuals_hierarchical)

    # Save predictions to file
    np.save(output_dir + f"/prediction.npy", prediction)


def run_analysis_hierarchical(
    event, output_directory, model_name, samples_tensor, sample=False
):
    model = PredictiveModel(event, samples_tensor)

    # Define output directories
    output_dir = output_directory + event.event_name + f"/{model_name}"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Don't continue if sampling the full event failed
    path_to_full_trace = (
        output_directory + event.event_name + "/model_default/trace.csv"
    )
    if not os.path.exists(path_to_full_trace):
        print("No trace containing samples for full light curve.")
        return None

    # Compute alert time
    alert_time = find_alert_time(event)

    # Load trace from fitting a full light curve
    trace = pd.read_csv(path_to_full_trace)
    t0 = np.exp(np.median(trace["ln_delta_t0"])) + alert_time

    masking_times = np.linspace(alert_time, t0, 4)

    # Fit predictive model for different masking times
    for i, mask_time in enumerate(masking_times):
        output = output_dir + f"/prediction_{i}"

        if not os.path.exists(output):
            os.makedirs(output)

        # Fit predictive model
        fit_predictive_model(model, event, output, masking_times[i], sample=sample)


def run_analysis_empirical(event, output_directory, model_name, sample=False):
    # Initialize model
    model = PredictiveModelEmpiricalPriors(event)

    # Define output directories
    output_dir = output_directory + event.event_name + f"/{model_name}"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Don't continue if sampling the full event failed
    path_to_full_trace = (
        output_directory + event.event_name + "/model_default/trace.csv"
    )
    if not os.path.exists(path_to_full_trace):
        print("No trace containing samples for full light curve.")
        return None

    # Compute alert time
    alert_time = find_alert_time(event)

    # Load trace from fitting a full light curve
    trace = pd.read_csv(path_to_full_trace)
    t0 = np.exp(np.median(trace["ln_delta_t0"])) + alert_time

    masking_times = np.linspace(alert_time, t0, 4)

    # Fit predictive model for different masking times
    for i, mask_time in enumerate(masking_times):
        output = output_dir + f"/prediction_{i}"

        if not os.path.exists(output):
            os.makedirs(output)

        # Fit predictive model
        fit_predictive_model(model, event, output, masking_times[i], sample=sample)


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
events_2003 = []

dirs_all = []


# Iterate over all directories and run the fits
data_path = "../../../../../scratch/astro/fb90/OGLE_ews/2003/"

for directory in os.listdir(data_path):
    path = data_path + directory

    if directory[-3:] not in exclude_2003:
        dirs_all.append(data_path + directory)

# Split directories into 4 different parts, for parallelization
# split_dirs = np.array_split(dirs_all, 4)
#
# dirs = split_dirs[int(sys.argv[1])]

# Load hyperparameter samples
samples_tensor = T.as_tensor_variable(np.load("../data/samples_hyper.npy"))

n_jobs = 8
result = Parallel(n_jobs)(
    delayed(run_analysis_hierarchical)(
        ca.data.OGLEData(directory),
        "../../../../../scratch/astro/fb90/output/2003/",
        "model_hierarchical_predictive",
        sample=True,
    )
    for directory in dirs_all
)

