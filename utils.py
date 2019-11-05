import logging
import os
import pickle
import random
import sys

import exoplanet as xo
import matplotlib as mpl
import numpy as np
import pandas as pd
import pymc3 as pm
import theano.tensor as T
from matplotlib import pyplot as plt

import caustic as ca

random.seed(42)


def run_sampling(
    pm_model, output_dir, ncores=1, nchains=2, max_attempts=2, filename="trace"
):
    # Log file output
    logging.basicConfig(
        filename=output_dir + "/sampling.log",
        filemode="w",
        format="%(name)s - %(levelname)s - %(message)s",
    )

    # Sample the model
    divperc = 20

    with pm_model:
        # Run initial chain
        try:
            trace = pm.sample(
                tune=1000,
                draws=4000,
                cores=ncores,
                chains=nchains,
                step=xo.get_dense_nuts_step(),
            )

        except pm.exceptions.SamplingError:
            logging.error("Sampling failed, model misspecified")
            return None

        # Check for divergences, restart sampling if necessary
        divergent = trace["diverging"]
        divperc = divergent.nonzero()[0].size / len(trace) * 100

        n_attempts = 1
        while divperc > 15.0 and n_attempts <= max_attempts:
            # Run sampling
            trace = pm.sample(
                tune=2000,
                draws=n_attempts * 10000,
                cores=ncores,
                chains=nchains,
                step=xo.get_dense_nuts_step(target_accept=0.9),
            )

            n_attempts += 1

        if divperc > 15:
            logging.warning(f"{divperc} of samples are diverging.")
            df = pm.trace_to_dataframe(trace, include_transformed=True)
            df.to_csv(output_dir + f"/{filename}.csv")
            return None

        else:
            df = pm.trace_to_dataframe(trace, include_transformed=True)
            df.to_csv(output_dir + f"/{filename}.csv")

    return trace


def find_alert_time(event):
    """
    Finds the alert time in the data. 

    This is defined in the paper as the time where there are 3 data points
    1 standard deviation away from baseline.
    """
    tmp = event.units

    event.units = "magnitudes"
    times = np.array(event.light_curves[0]["HJD"] - 2450000)
    mags = np.array(event.light_curves[0]["mag"])
    event.units = tmp

    # Also not clear from the paper how to doe this,
    # use the first 10 data points in the light curve to determine the magnitude
    # baseline

    mean_mag = np.mean(mags[:10])
    std_mag = np.std(mags[:10])

    num_above = 0
    i = 9

    while num_above < 3 and i < len(times) - 1:

        i += 1

        if mags[i] < mean_mag - std_mag:
            num_above += 1
        else:
            num_above = 0.0

    if len(times) - 1 == i:
        print(
            "Give me more training data, not alerted yet,\
                this is probably going to fail"
        )

    return times[i - 1]


def run_optimizer(pm_model, return_logl=False):
    with pm_model:
        map_params = xo.optimize(start=pm_model.test_point, vars=pm_model.m_b)
        map_params = xo.optimize(start=map_params, vars=[pm_model.ln_delta_t0])
        map_params = xo.optimize(
            start=map_params,
            vars=[pm_model.ln_delta_t0, pm_model.ln_A0, pm_model.ln_tE],
        )
        map_params, info = xo.optimize(start=map_params, return_info=True)

    if return_logl == True:
        return map_params, info.fun
    else:
        return map_params


def evaluate_prediction_quantiles(event, pm_model, trace, t_grid, alert_time):
    """
    Evaluates median model prediciton in data space.
    """
    n_samples = 500

    samples = xo.get_samples_from_trace(trace, size=n_samples)

    with pm_model:
        # Compute the trajectory of the lens
        trajectory = ca.trajectory.Trajectory(
            event, alert_time + T.exp(pm_model.ln_delta_t0), pm_model.u0, pm_model.tE
        )
        u_dense = trajectory.compute_trajectory(t_grid)

        # Compute the magnification
        mag_dense = (u_dense ** 2 + 2) / (u_dense * T.sqrt(u_dense ** 2 + 4))

        F_base = 10 ** (-(pm_model.m_b - 22.0) / 2.5)

        # Compute the mean model
        prediction = pm_model.f * F_base * mag_dense + (1 - pm_model.f) * F_base

    # Evaluate model for each sample on a fine grid
    n_pts_dense = T.shape(t_grid)[0].eval()
    n_bands = len(event.light_curves)

    prediction_eval = np.zeros((n_samples, n_bands, n_pts_dense))

    # Evaluate predictions in model context
    with pm_model:
        for i, sample in enumerate(samples):
            prediction_eval[i] = xo.eval_in_model(prediction, sample)

    for i in range(n_bands):
        q = np.percentile(prediction_eval[:, i, :], [16, 50, 84], axis=0)

    return q


def evaluate_map_prediction(event, pm_model, map_params, t_grid, alert_time):
    """
    Evaluates MAP model prediciton in data space.
    """
    with pm_model:
        # Compute the trajectory of the lens
        trajectory = ca.trajectory.Trajectory(
            event, alert_time + T.exp(pm_model.ln_delta_t0), pm_model.u0, pm_model.tE
        )
        u_dense = trajectory.compute_trajectory(t_grid)

        # Compute the magnification
        mag_dense = (u_dense ** 2 + 2) / (u_dense * T.sqrt(u_dense ** 2 + 4))

        F_base = 10 ** (-(pm_model.m_b - 22.0) / 2.5)

        # Compute the mean model
        prediction = pm_model.f * F_base * mag_dense + (1 - pm_model.f) * F_base

    # Evaluate model for each sample on a fine grid
    n_pts_dense = T.shape(t_grid)[0].eval()
    n_bands = len(event.light_curves)

    prediction_eval = np.zeros(n_pts_dense)

    # Evaluate predictions in model context
    with pm_model:
        prediction_eval = xo.eval_in_model(prediction, map_params)

    return prediction_eval

