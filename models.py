import sys

import numpy as np
import pymc3 as pm
import theano.tensor as T

import caustic as ca

sys.path.append("../")

from utils import find_alert_time


class DefaultModel(ca.models.SingleLensModel):
    """
    Default model.
    """

    def __init__(self, data):
        super(DefaultModel, self).__init__(data, standardize=False)

        # Compute alert time
        alert_time = find_alert_time(data)

        n_bands = len(data.light_curves)
        BoundedNormal = pm.Bound(pm.Normal, lower=0.0)
        BoundedNormal_1 = pm.Bound(pm.Normal, lower=1.0)

        # Initialize linear parameters
        f = pm.Uniform("f", 0.0, 1.0, testval=0.9)

        m_b = pm.Normal("m_b", mu=15.0, sd=10.0, testval=15.0)

        F_base = 10 ** (-(m_b - 22.0) / 2.5)

        # Initialize non-linear parameters
        ## Posterior is multi-modal in t0 and it's critical that the it is
        ## initialized near the true value
        ln_t0_testval = T.log(ca.utils.estimate_t0(data) - alert_time)
        ln_delta_t0 = pm.Normal("ln_delta_t0", 4.0, 5.0, testval=ln_t0_testval)
        delta_t_0 = T.exp(ln_delta_t0)

        ln_A0 = pm.Exponential("ln_A0", 0.1, testval=np.log(3.0))

        ln_tE = pm.Normal("ln_tE", mu=4.0, sd=5.0, testval=3.0)

        # Deterministic transformations
        tE = pm.Deterministic("tE", T.exp(ln_tE))
        u0 = pm.Deterministic(
            "u0", T.sqrt(2 * T.exp(ln_A0) / T.sqrt(T.exp(ln_A0) ** 2 - 1) - 2)
        )

        # Compute the trajectory of the lens
        trajectory = ca.trajectory.Trajectory(data, alert_time + delta_t_0, u0, tE)
        u = trajectory.compute_trajectory(self.t)

        # Compute the magnification
        mag = (u ** 2 + 2) / (u * T.sqrt(u ** 2 + 4))

        # Compute the mean model
        mean = f * F_base * mag + (1 - f) * F_base

        # We allow for rescaling of the error bars by a constant factor
        c = BoundedNormal_1(
            "c",
            mu=T.ones(n_bands),
            sd=2.0 * T.ones(n_bands),
            testval=1.5 * T.ones(n_bands),
            shape=(n_bands),
        )

        # Diagonal terms of the covariance matrix
        var_F = (c * self.sig_F) ** 2

        # Compute the Gaussian log_likelihood, add it as a potential term to the model
        ll = self.compute_log_likelihood(self.F - mean, var_F)
        pm.Potential("log_likelihood", ll)

        # Save logp-s for each variable
        pm.Deterministic("logp_f", f.distribution.logp(f))
        pm.Deterministic("logp_ln_delta_t0", ln_delta_t0.distribution.logp(ln_delta_t0))
        pm.Deterministic("logp_ln_A0", ln_A0.distribution.logp(ln_A0))
        pm.Deterministic("logp_ln_tE", ln_tE.distribution.logp(ln_tE))


class DefaultModelUniformPriors(ca.models.SingleLensModel):
    """
    This is just for the purpose of computing the maximum likelihood solution.
    """

    def __init__(self, data):
        super(DefaultModelUniformPriors, self).__init__(data, standardize=False)
        # Compute alert time
        alert_time = find_alert_time(data)

        n_bands = len(data.light_curves)
        BoundedNormal = pm.Bound(pm.Normal, lower=0.0)
        BoundedNormal_1 = pm.Bound(pm.Normal, lower=1.0)

        # Initialize linear parameters
        f = pm.Uniform("f", 0.0001, 0.999, testval=0.9)

        m_b = pm.Uniform("m_b", 8.0, 25.0, testval=15.0)

        F_base = 10 ** (-(m_b - 22.0) / 2.5)

        # Initialize non-linear parameters
        ## Posterior is multi-modal in t0 and it's critical that the it is
        ## initialized near the true value
        t0_testval = T.log(ca.utils.estimate_t0(data) - alert_time)
        ln_delta_t0 = pm.Uniform("ln_delta_t0", -1.0, 10.0, testval=t0_testval)
        delta_t_0 = T.exp(ln_delta_t0)

        ln_A0 = pm.Uniform("ln_A0", 0.1, 100, testval=np.log(3.0))

        ln_tE = pm.Uniform("ln_tE", -1.0, 10, testval=3.0)

        # Deterministic transformations
        tE = pm.Deterministic("tE", T.exp(ln_tE))
        u0 = pm.Deterministic(
            "u0", T.sqrt(2 * T.exp(ln_A0) / T.sqrt(T.exp(ln_A0) ** 2 - 1) - 2)
        )

        # Compute the trajectory of the lens
        trajectory = ca.trajectory.Trajectory(data, alert_time + delta_t_0, u0, tE)
        u = trajectory.compute_trajectory(self.t)

        # Compute the magnification
        mag = (u ** 2 + 2) / (u * T.sqrt(u ** 2 + 4))

        # Compute the mean model
        mean = f * F_base * mag + (1 - f) * F_base

        # We allow for rescaling of the error bars by a constant factor
        c = BoundedNormal_1(
            "c",
            mu=T.ones(n_bands),
            sd=5.0 * T.ones(n_bands),
            testval=1.5 * T.ones(n_bands),
            shape=(n_bands),
        )

        # Diagonal terms of the covariance matrix
        var_F = (c * self.sig_F) ** 2

        # Compute the Gaussian log_likelihood, add it as a potential term to the model
        ll = self.compute_log_likelihood(self.F - mean, var_F)
        pm.Potential("log_likelihood", ll)


class HierarchicalModel(pm.Model):
    """
    Hierchical model using the importance resampling trick.
    """

    def __init__(self, samples_tensor, samples_logp_tensor):
        super(HierarchicalModel, self).__init__()

        # This will take a while, loading the massive arrays into memory is costly
        BoundedNormal = pm.Bound(pm.Normal, lower=0.0)

        #  Parameters of the prior for blend fraction f
        alpha_f = BoundedNormal("alpha_f", 0, 10.0, testval=5.0)
        beta_f = BoundedNormal("beta_f", 0, 10.0, testval=1.0)

        # Parameters of the prior for delta_t0
        mu_ln_delta_t0 = pm.Normal("mu_ln_delta_t0", 4, 5.0, testval=3.0)
        sig_ln_delta_t0 = BoundedNormal("sig_ln_delta_t0", 1.0, 5.0, testval=1.0)

        # Parameters of the prior for ln_A0
        lam_ln_A0 = BoundedNormal("lam_ln_A0", 0.0, 1.0, testval=0.1)

        # Parameters of the prior for ln_tE
        mu_ln_tE = pm.Normal("mu_ln_tE", 3.0, 10.0, testval=3.0)
        sig_ln_tE = BoundedNormal("sig_ln_tE", 1.0, 20.0, testval=1.0)

        def compute_ll():
            n_bands = T.shape(samples_tensor).eval()[0]

            result = 0.0

            # Iterate over members of the population
            for i in range(n_bands):
                # Compute new prior

                # ln_A0
                log_new_prior = pm.Exponential.dist(lam_ln_A0).logp(
                    samples_tensor[i, :, 0]
                )

                # delta_t0
                log_new_prior += pm.Normal.dist(
                    mu=mu_ln_delta_t0, sd=sig_ln_delta_t0
                ).logp(samples_tensor[i, :, 1])

                # ln_tE
                log_new_prior += pm.Normal.dist(mu=mu_ln_tE, sd=sig_ln_tE).logp(
                    samples_tensor[i, :, 2]
                )

                # f
                log_new_prior += pm.Beta.dist(alpha_f, beta_f).logp(
                    samples_tensor[i, :, 3]
                )

                # Compute old prior
                log_old_prior = T.sum(samples_logp_tensor[i], axis=1)

                # Compute importance resampling fraction
                log_frac = log_new_prior - log_old_prior

                result += T.log(T.sum(T.exp(log_frac)) / T.shape(samples_tensor)[1])

            return result

        pm.Potential("log_likelihood", compute_ll())


class PredictiveModelEmpiricalPriors(ca.models.SingleLensModel):
    """
    Model optimized for prediction of ongoing events, based on Albrow (2004).
    """

    def __init__(
        self,
        data,
        lam_ln_A0=0.62569998,
        mu_ln_delta_t0=3.14397,
        sig_ln_delta_t0=1.198987,
        mu_ln_tE=3.746819,
        sig_ln_tE=1.26364,
    ):
        super(PredictiveModelEmpiricalPriors, self).__init__(data, standardize=False)

        # Compute alert time
        alert_time = find_alert_time(data)

        #  Compute empirically determined parameters of prior distributions
        n_bands = len(data.light_curves)
        BoundedNormal = pm.Bound(pm.Normal, lower=0.0)
        BoundedNormal_1 = pm.Bound(pm.Normal, lower=1.0)

        # Initialize linear parameters
        f = pm.Uniform("f", 0.0, 1.0, testval=0.5)

        m_b = pm.Normal("m_b", mu=15.0, sd=10.0, testval=15.0)

        F_base = 10 ** (-(m_b - 22.0) / 2.5)

        # Initialize non-linear parameters
        ln_delta_t0 = pm.Normal("ln_delta_t0", 3.14397, 1.198987, testval=3.0)
        delta_t_0 = T.exp(ln_delta_t0)

        ln_A0 = pm.Exponential("ln_A0", 0.62569998, testval=2.0)
        ln_tE = pm.Normal("ln_tE", mu=3.746819, sd=1.26364, testval=3.0)

        # Deterministic transformations
        tE = pm.Deterministic("tE", T.exp(ln_tE))
        u0 = pm.Deterministic(
            "u0", T.sqrt(2 * T.exp(ln_A0) / T.sqrt(T.exp(ln_A0) ** 2 - 1) - 2)
        )

        # Compute the trajectory of the lens
        trajectory = ca.trajectory.Trajectory(data, alert_time + delta_t_0, u0, tE)
        u = trajectory.compute_trajectory(self.t)

        # Compute the magnification
        mag = (u ** 2 + 2) / (u * T.sqrt(u ** 2 + 4))

        # Compute the mean model
        mean = f * F_base * mag + (1 - f) * F_base

        # We allow for rescaling of the error bars by a constant factor
        c = BoundedNormal_1(
            "c",
            mu=T.ones(n_bands),
            sd=2.0 * T.ones(n_bands),
            testval=1.5 * T.ones(n_bands),
            shape=(n_bands),
        )

        # Diagonal terms of the covariance matrix
        var_F = (c * self.sig_F) ** 2

        # Compute the Gaussian log_likelihood, add it as a potential term to the model
        ll = self.compute_log_likelihood(self.F - mean, var_F)
        pm.Potential("log_likelihood", ll)


class PredictiveModel(ca.models.SingleLensModel):
    """
    Hierarchical predictive model. Uses posterior samples over the
    hyperaparameters describing a population of events to reweight the priors
    on key parameters. 
    """

    def __init__(self, data, samples_tensor, fit_blending=False):
        super(PredictiveModel, self).__init__(data, standardize=False)

        # Compute alert time
        alert_time = find_alert_time(data)

        n_bands = len(data.light_curves)
        BoundedNormal = pm.Bound(pm.Normal, lower=0.0)
        BoundedNormal_1 = pm.Bound(pm.Normal, lower=1.0)

        # mock prior if fit_blending=True
        f = pm.Uniform("f", 0.0, 1.0, testval=0.5, shape=(n_bands))

        m_b = pm.Normal(
            "m_b",
            mu=15.0 * T.ones(n_bands),
            sd=10.0 * T.ones(n_bands),
            testval=22.0
            - 2.5
            * T.log10(T.as_tensor_variable(ca.utils.estimate_baseline_flux(data))),
            shape=(n_bands),
        )

        F_base = 10 ** (-(m_b - 22.0) / 2.5)

        # The following are mock priors, they don't do anything, it's just so
        # PyMC3 initializes the RVs
        ln_delta_t0 = pm.Uniform("ln_delta_t0", -1, 8, testval=3.0)  # mock prior
        ln_A0 = pm.Uniform("ln_A0", 0.0, 100, testval=2.0)  # mock prior
        ln_tE = pm.Uniform("ln_tE", -1, 8, testval=3.0)  # mock prior

        delta_t0 = T.exp(ln_delta_t0)

        u0 = pm.Deterministic(
            "u0", T.sqrt(2 * T.exp(ln_A0) / T.sqrt(T.exp(ln_A0) ** 2 - 1) - 2)
        )

        # Deterministic transformations
        tE = pm.Deterministic("tE", T.exp(ln_tE))

        # Compute the trajectory of the lens
        trajectory = ca.trajectory.Trajectory(data, alert_time + delta_t0, u0, tE)
        u = trajectory.compute_trajectory(self.t)

        # Compute the magnification
        mag = (u ** 2 + 2) / (u * T.sqrt(u ** 2 + 4))

        # Compute the mean model
        mean = f * F_base * mag + (1 - f) * F_base

        # We allow for rescaling of the error bars by a constant factor
        c = BoundedNormal_1(
            "c",
            mu=T.ones(n_bands),
            sd=2.0 * T.ones(n_bands),
            testval=1.5,
            shape=(n_bands),
        )

        # Diagonal terms of the covariance matrix
        var_F = (c * self.sig_F) ** 2

        # Compute the Gaussian log_likelihood, add it as a potential term to the model
        ll_single = self.compute_log_likelihood(self.F - mean, var_F)

        # Compute additional term for the likelihood
        lam_ln_A0 = samples_tensor[:, 0]
        mu_ln_delta_t0 = samples_tensor[:, 1]
        sig_ln_delta_t0 = samples_tensor[:, 2]
        mu_ln_tE = samples_tensor[:, 3]
        sig_ln_tE = samples_tensor[:, 4]
        alpha_f = samples_tensor[:, 5]
        beta_f = samples_tensor[:, 6]

        # Iterate over samples from hyperparameters
        prior_ln_A0 = pm.Exponential.dist(lam_ln_A0).logp(ln_A0)
        prior_ln_delta_t0 = pm.Normal.dist(mu=mu_ln_delta_t0, sd=sig_ln_delta_t0).logp(
            ln_delta_t0
        )
        prior_ln_tE = pm.Normal.dist(mu=mu_ln_tE, sd=sig_ln_tE).logp(ln_tE)

        if fit_blending == True:
            prior_f = pm.Beta.dist(alpha_f, beta_f).logp(f)

            ll_hyper = T.log(
                T.sum(T.exp(prior_ln_A0 + prior_ln_tE + prior_ln_delta_t0 + prior_f))
            )
        else:

            ll_hyper = T.log(
                T.sum(T.exp(prior_ln_A0 + prior_ln_tE + prior_ln_delta_t0))
            )
