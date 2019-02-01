import numpy as np
import sys

import pymc3 as pm
import theano
import theano.tensor as T
from theano.ifelse import ifelse

import exoplanet as xo
from exoplanet.gp import terms, GP
from exoplanet.utils import get_samples_from_trace
from exoplanet.utils import eval_in_model

from scipy.special import gamma
from scipy.stats import invgamma
from scipy.optimize import fsolve


class PointSourcePointLens(pm.Model):
    #  override __init__ function from pymc3 Model class
    def __init__(self, data, use_joint_prior=True, name='', model=None):
        # call super's init first, passing model and name
        # to it name will be prefix for all variables here if
        # no name specified for model there will be no prefix
        super(PointSourcePointLens, self).__init__(name, model)
        # now you are in the context of instance,
        # `modelcontext` will return self you can define
        # variables in several ways note, that all variables
        # will get model's name prefix

        # Pre process the data
        data.convert_data_to_fluxes()
        df = data.get_standardized_data()

        # Data
        self.t = df['HJD - 2450000'].values
        self.F = df['I_flux'].values
        self.sigF = df['I_flux_err'].values

        # Microlensing model parameters
        self.ln_DeltaF = pm.DensityDist('DeltaF', self.prior_ln_DeltaF, testval=np.log(5))
        self.Fb = pm.DensityDist('Fb', self.prior_Fb, testval=0.1)
        # Posterior is multi-modal in t0 and it's critical that the it is 
        # initialized near the true value
        t0_guess_idx = (np.abs(self.F - np.max(self.F))).argmin() 
        self.ln_t0 = pm.DensityDist('ln_t0', self.prior_ln_t0,
            testval=T.log(self.t[t0_guess_idx]))
        self.ln_u0 = pm.DensityDist('ln_u0', self.prior_ln_u0, testval=np.log(2))
        self.ln_tE = pm.DensityDist('ln_tE', self.prior_ln_tE, testval=np.log(100))
        
        # Save source flux and blend parameters
        #m_source, m_blend = self.revert_flux_params_to_nonstandardized_format(
        #    data)
        #self.mag_source = pm.Deterministic("m_source", m_source)
        #self.mag_blend = pm.Deterministic("m_blend", m_blend)

        # Noise model parameters
        self.ln_K = pm.DensityDist('ln_K', self.prior_ln_K, testval=np.log(1.2))

        # Save the logp for all of the priors
        self.logp_DeltaF = pm.Deterministic("logp_ln_DeltaF", 
            self.prior_ln_DeltaF(self.DeltaF))
        self.logp_Fb = pm.Deterministic("logp_Fb", self.prior_Fb(self.Fb))
        self.logp_ln_t0 = pm.Deterministic("logp_ln_t0", 
            self.prior_ln_t0(self.ln_t0))
        self.logp_ln_u0 = pm.Deterministic("logp_ln_u0", 
            self.prior_ln_u0(self.ln_u0))
        self.logp_ln_tE = pm.Deterministic("logp_ln_tE", 
            self.prior_ln_tE(self.ln_tE))
        self.logp_ln_K = pm.Deterministic("logp_ln_K", 
            self.prior_ln_K(self.ln_K))

        # Save log posterior value
        self.log_posterior = pm.Deterministic("log_posterior", self.logpt)

        #test = T.printing.Print('sd')(t0)

        Y_obs = pm.Normal('Y_obs', mu=self.mean_function(), 
            sd=((T.exp(self.ln_K) + 1))*self.sigF, 
            observed=self.F, shape=len(self.F))

    def revert_flux_params_to_nonstandardized_format(self, data):
        # Revert Fb and DeltaF to non-standardized units
        median_F = np.median(data.df['I_flux'].values)
        std_F = np.std(data.df['I_flux'].values)

        DeltaF_ = std_F*self.DeltaF + median_F
        Fb_ = std_F*self.DeltaF + median_F

        # Calculate source flux and blend flux
        FS = DeltaF_/(self.peak_mag() - 1)
        FB = (Fb_ - FS)/FS

        # Convert fluxes to magnitudes
        mu_m, sig_m = data.fluxes_to_magnitudes(np.array([FS, FB]), 
            np.array([0., 0.]))
        mag_source, mag_blend = mu_m

        return mag_source, mag_blend
    
    def log_likelihood(self):
        """Gaussian log likelihood."""
        # Gaussian prior
        mu = self.mean_function()
        sig = (T.exp(self.ln_K) + 1.)*self.sigF

        res = T.sum(-0.5*T.log(2*np.pi*sig**2) - 0.5*(self.F - mu)**2/sig**2, axis=0)
        return res

    def mean_function(self):
        """PSPL model"""
        u0 = T.exp(self.ln_u0)
        t0 = T.exp(self.ln_t0)
        tE = T.exp(self.ln_tE)
        DeltaF = T.exp(self.ln_DeltaF)
        u = T.sqrt(u0**2 + ((self.t - t0)/tE)**2)
        A = lambda u: (u**2 + 2)/(u*T.sqrt(u**2 + 4))

        return DeltaF*(A(u) - 1)/(A(u0) - 1) + self.Fb

    def peak_mag(self):
        """PSPL model"""
        u = T.sqrt(self.u0**2 + ((self.t - self.t0)/self.tE)**2)
        A = lambda u: (u**2 + 2)/(u*T.sqrt(u**2 + 4))

        return A(self.u0)

    def prior_ln_DeltaF(self, value):
        ln_DeltaF = T.cast(value, 'float64')

        # Gaussian prior
        mu = 0.
        sig = np.log(10)

        return -0.5*T.log(2*np.pi*sig**2) - 0.5*(ln_DeltaF - mu)**2/sig**2

    def prior_Fb(self, value):
        Fb = T.cast(value, 'float64')

        # Gaussian prior
        mu = 0.
        sig = np.log(50)

        return -0.5*T.log(2*np.pi*sig**2) - 0.5*(Fb - mu)**2/sig**2

    def prior_ln_t0(self, value):
        ln_t0 = T.cast(value, 'float64')

        # Gaussian prior
        mu = self.t[0] + (self.t[-1] - self.t[0])/2
        sig = (self.t[-1] - self.t[0])/2

        return -0.5*T.log(2*np.pi*sig**2) - 0.5*(ln_t0 - mu)**2/sig**2


    def prior_ln_tE(self, value):
        ln_tE = T.cast(value, 'float64')

        # Gaussian prior
        mu = T.log(20)
        sig = T.log(300)

        return -0.5*T.log(2*np.pi*sig**2) - 0.5*(ln_tE - mu)**2/sig**2

    def prior_ln_u0(self, value):
        ln_u0 = T.cast(value, 'float64')

        # Gaussian prior
        mu = 0.
        sig = np.log(1.2)

        return -0.5*T.log(2*np.pi*sig**2) - 0.5*(ln_u0 - mu)**2/sig**2

    def prior_ln_K(self, value):
        ln_K = T.cast(value, 'float64')

        # Prior
        mu = 0.
        sig = np.log(2)

        return -0.5*T.log(2*np.pi*sig**2) - 0.5*(ln_K - mu)**2/sig**2