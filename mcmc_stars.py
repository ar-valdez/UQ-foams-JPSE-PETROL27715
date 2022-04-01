# ****************************************************************************
# Name           : mcmc_stars.py
# Author         : Andres Valdez - Bernardo Rocha
# Version        : 1.0
# Description    : performs mcmc sampling using pymc3.
#                  valid only for STARS Newtonian and non-Newtonian
# Data           : 03-24-2022
# ****************************************************************************

from __future__ import unicode_literals
import os, sys
import numpy as np
import scipy as scp
import matplotlib as mpl
import time

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

#For 0.49\textwidth figs works ok
mpl.rcParams['axes.labelsize']  = 14
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['legend.fontsize'] = 12
mpl.rcParams['text.usetex']     = True
mpl.rcParams['font.family']     = 'serif'

from scipy import optimize

import pymc3 as pmc
import arviz as arz
import theano.tensor as tt
from theano.compile.ops import as_op

np.random.seed(1234)

########################################################################
# foam model definitions
########################################################################
class fmf_models(object):
    """
    This class contains all the methods to analyze different foam models.
    """
    def __init__(self,core_params):
        self.swc, self.sgr      = core_params[0], core_params[1]
        self.nw,  self.ng       = core_params[2], core_params[3]
        self.kuw, self.kug      = core_params[4], core_params[5]
        self.muw, self.mug      = core_params[6], core_params[7]
        self.u,   self.sigma_wg = core_params[8], core_params[9]
    
    def frac_flow(self,sw):
        """
        This function returns the fractional flow theory functions
        """
        if(isinstance(sw, float)):
            sw = np.array([sw])
        
        se = np.zeros(len(sw))
        for j in range(len(sw)):
            if(sw[j] < self.swc):
                se[j] = 0.0
            elif(sw[j] > 1.0 - self.sgr):
                se[j] = 1.0
            else:
                se[j] = (sw[j] - self.swc) / (1.0 - self.swc - self.sgr)

        krw = self.kuw * np.power(se,self.nw)
        krg = self.kug * np.power(1.0-se,self.ng)
        mob_w = krw / self.muw
        mob_g = krg / self.mug
        
        return krw,krg,mob_w,mob_g
    
    def stars_Newtonian_model(self,sw,fmmob,SF,sfbet,simple=True):
        """
        This function returns the functions for Newtonian CMG-STARS model
        """
        if(isinstance(sw, float)):
            sw = np.array([sw])
        
        krw,krg,mob_w,mob_g = self.frac_flow(sw)
        
        F2 = 0.5 + (1.0/np.pi) * np.arctan( sfbet * (sw - SF))
        
        mrf    = 1.0 + fmmob * F2
        lt     = mob_w + (mob_g/mrf)
        fg     = (mob_g / mrf) / lt

        mu_app = 1.0 / lt
        
        if(simple):
            return fg, mu_app, mrf, lt
        else:
            return fg, mu_app, mrf, lt, F2

    def root_stars(self,mu_foam,sw,fmmob,SF,sfbet,epcap,fmcap):
        """
        This function is used to solve  mu_foam - mu_app, for CMG-STARS non-Newtonian
        """
        # Compute F2
        F2 = self.stars_Newtonian_model(sw,fmmob,SF,sfbet,simple=False)[4]
        
        # Compute F5
        Nca = (mu_foam * self.u) / self.sigma_wg
        F5  = np.zeros((len(Nca)))
        for i in range(len(Nca)):
            if(Nca[i] < fmcap):
                F5[i] = 1.0
            else:
                F5[i] = np.power(fmcap/Nca[i],epcap)
        
        # Compute mrf
        mrf = 1.0 + fmmob * F2 * F5
        
        # Compute apparent viscosity
        krw,krg,mob_w,mob_g = self.frac_flow(sw)
        lt     = mob_w + (mob_g/mrf)
        mu_app = 1.0 / lt
        
        return mu_foam - mu_app

    def stars_non_Newtonian_model(self,sw,fmmob,SF,sfbet,epcap,fmcap,simple=True):
        """
        This function returns the functions for non-Newtonian CMG-STARS model
        """
        if(isinstance(sw, float)):
            sw = np.array([sw])
        
        krw,krg,mob_w,mob_g = self.frac_flow(sw)
        
        # Compute F2
        F2 = self.stars_Newtonian_model(sw,fmmob,SF,sfbet,simple=False)[4]
        
        # Make an initial guess for mu_apparent function
        F5   = 1.0
        mrf  = 1.0 + fmmob * F2 * F5
        lt   = mob_w + (mob_g/mrf)
        mu_0 = 1.0 / lt
        
        mu_app = optimize.fsolve(self.root_stars, mu_0, args=(sw,fmmob,SF,sfbet,epcap,fmcap))

        # Compute F5, with mu_app
        Nca = (mu_app * self.u) / self.sigma_wg
        F5  = np.zeros((len(Nca)))
        for i in range(len(Nca)):
            if(Nca[i] < fmcap):
                F5[i] = 1.0
            else:
                F5[i] = np.power(fmcap/Nca[i],epcap)
        
        # Compute mrf and results of fg, mu_app, and lt
        mrf    = 1.0 + fmmob * F2 * F5
        lt     = mob_w + (mob_g/mrf)
        fg     = (mob_g / mrf) / lt
        mu_app = 1.0 / lt
        
        if(simple):
            return fg, mu_app, mrf, lt
        else:
            return fg, mu_app, mrf, lt, F2, F5


########################################################################
# Auxiliary functions
########################################################################
def load_parameters(filename):
    """
    :param filename: parameters file name
    :return: dictionary with variables names and values
    """
    dpar = {}
    with open(filename) as f:
        for line in f:
            (key, val) = line.split()
            dpar[str(key)] = float(val)
    return dpar

def load_exp_data(filename):
    """    
    File format for data at particular coordinate' of reservoir
    1st column: Foam quality (f_g)
    2nd column: Apparent viscosity (mu_app)
    """
    data   = np.loadtxt(filename, comments="#")
    fg     = data[:,0]
    mu_app = data[:,1]
    return fg,mu_app

def water_saturation(fg,mu_app,mprop):
    """
    This function returns the experimental water saturation
    """
    swc = mprop[0]
    sgr = mprop[1]
    nw  = mprop[2]
    kuw = mprop[4]
    muw = mprop[6]
    
    x_sw = np.zeros(len(fg))
    for k in range(len(fg)):
        if(mu_app[k] > 1.0e-08):
            x_sw[k] = swc + (1.0 - swc - sgr) * np.power( (muw * (1.0 - fg[k])) / (kuw * mu_app[k]) ,1.0/nw)
    return x_sw

def mrf_exp(mob_g,fg,mu_app):
    """
    This function returns the experimental MRF
    """
    mrf = np.ones(len(fg))
    for k in range(len(fg)):
        if(fg[k] > 1.0e-06):
            mrf[k] = mob_g[k] * mu_app[k] / fg[k]
    return mrf

def model_labels(foam_model):
    """
    This function returns the labels for the foam models
    """
    if(foam_model == 'CMG-STARS'):
        model_name   = 'CMG-STARS'
        param_labels = ['fmmob','SF','sfbet']
    elif(foam_model == 'STARS-FULL'):
        model_name   = 'STARS-FULL'
        param_labels = ['fmmob','SF','sfbet','epcap','fmcap']
    return model_name,param_labels

def files_dataset(test_case):
    """
    Points to the file names for each dataset
    Returns labels
    """
    if(test_case == 'Synthetic'):
        param_file = 'input_par_Alvarez2001.dat'
        data_file  = 'Synthetic.dat'
        nmax       = 0.0
        b          = 1.0 / 3.0
        Ndata      = 1
    elif(test_case == 'Smooth'):
        param_file = 'input_par_Alvarez2001.dat'
        data_file  = 'Smooth.dat'
        nmax       = 0.0
        b          = 1.0 / 3.0
        Ndata      = 1
    else:
        print(test_case,'Not studied')
        sys.exit()
    
    return param_file,data_file,nmax,b,Ndata

def ranges_lmfit(foam_model,test_case,core_params):
    """
    Define bounds and initial shoots for the foam models
    p0, p1, , ..., pn :: Are the upper-lower bounds for each parameter
    i0, i1, , ..., in :: Are the initial values (shoot) for each parameter
    """

    if(foam_model == 'STARS-FULL'):
        if(test_case == 'Synthetic'):
            i0 , i1 , i2 , i3 , i4 = 1.6e+05,0.31,500,0.5,2.46e-05
            p0 , p1 , p2 , p3 , p4 = [0.10*i0,1.90*i0] , [core_params[0],1.0-core_params[1]] , [0.10*i2,1.90*i2] , [0.10*i3,1.90*i3] , [0.10*i4,1.90*i4]
            p0 , p1 , p2 , p3 , p4 = [0.10*i0,1.90*i0] , [core_params[0],1.0-core_params[1]] , [1.0e+01,1.0e+04] , [0,2] , [0,1.0e-04]
            penalty                = 1.0e+00
        elif(test_case == 'Smooth'):
            i0 , i1 , i2 , i3 , i4 = 1.6e+05,0.31,50,0.5,2.46e-05
            p0 , p1 , p2 , p3 , p4 = [0.10*i0,1.90*i0] , [core_params[0],1.0-core_params[1]] , [0.10*i2,1.90*i2] , [0.10*i3,1.90*i3] , [0.10*i4,1.90*i4]
            p0 , p1 , p2 , p3 , p4 = [0.10*i0,1.90*i0] , [core_params[0],1.0-core_params[1]] , [1.0e+01,1.0e+04] , [0,2] , [0,1.0e-04]
            p0 , p1 , p2 , p3 , p4 = [0.10*i0,1.90*i0] , [core_params[0],1.0-core_params[1]] , [1.00,100] , [0,2] , [0,1.0e-04] # Just for pfl
            penalty                = 1.0e+00
        return p0,p1,p2,p3,p4,i0,i1,i2,i3,i4,penalty

def save_trace(trace,foam_model,type_fit,sample_file):
    """
    Routine to save samples
    """
    fid = open(sample_file, "w")
    
    if(type_fit in ['mrf_fit','mu_fg_fit']):
        if(foam_model == 'STARS-FULL' and test_type == 'Complete'):
            p1 , p2 , p3 , p4 , p5 , sigma = trace[param_labels[0]] , trace[param_labels[1]] , trace[param_labels[2]] , trace[param_labels[3]] , trace[param_labels[4]] , trace['sigma']
            for k in range(len(p1)):
                fid.write("%d \t %0.8e \t %0.8e \t %0.8e \t %0.8e \t %0.8e \t %0.8e\n" % (k+1,p1[k],p2[k],p3[k],p4[k],p5[k],sigma[k]))
        elif(foam_model == 'STARS-FULL' and test_type == 'Reduced'):
            p1 , p2 , p3 , p4 , sigma = trace[param_labels[0]] , trace[param_labels[1]] , trace[param_labels[2]] , trace[param_labels[3]] , trace['sigma']
            for k in range(len(p1)):
                fid.write("%d \t %0.8e \t %0.8e \t %0.8e \t %0.8e \t %0.8e \t %0.8e\n" % (k+1,p1[k],p2[k],p3[k],p4[k],min_Nca,sigma[k]))
        elif(foam_model == 'CMG-STARS'):
            p1 , p2 , p3 , sigma = trace[param_labels[0]] , trace[param_labels[1]] , trace[param_labels[2]] , trace['sigma']
            for k in range(len(p1)):
                fid.write("%d \t %0.8e \t %0.8e \t %0.8e \t %0.8e\n" % (k+1,p1[k],p2[k],p3[k],sigma[k]))

    
    elif(type_fit == 'full_fit'):
        if(foam_model == 'STARS-FULL' and test_type == 'Complete'):
            p1 , p2 , p3 , p4 , p5 , sigma = trace[param_labels[0]] , trace[param_labels[1]] , trace[param_labels[2]] , trace[param_labels[3]] , trace[param_labels[4]] , trace['sigma']
            for k in range(len(p1)):
                fid.write("%d \t %0.8e \t %0.8e \t %0.8e \t %0.8e \t %0.8e \t %0.8e \t %0.8e\n" % (k+1,p1[k],p2[k],p3[k],p4[k],p5[k],sigma[k,0],sigma[k,1]))
        elif(foam_model == 'STARS-FULL' and test_type == 'Reduced'):
            p1 , p2 , p3 , p4 , sigma = trace[param_labels[0]] , trace[param_labels[1]] , trace[param_labels[2]] , trace[param_labels[3]] , trace['sigma']
            for k in range(len(p1)):
                fid.write("%d \t %0.8e \t %0.8e \t %0.8e \t %0.8e \t %0.8e \t %0.8e \t %0.8e\n" % (k+1,p1[k],p2[k],p3[k],p4[k],min_Nca,sigma[k,0],sigma[k,1]))
        elif(foam_model == 'CMG-STARS'):
            p1 , p2 , p3 , sigma = trace[param_labels[0]] , trace[param_labels[1]] , trace[param_labels[2]] , trace['sigma']
            for k in range(len(p1)):
                fid.write("%d \t %0.8e \t %0.8e \t %0.8e \t %0.8e \t %0.8e\n" % (k+1,p1[k],p2[k],p3[k],sigma[k,0],sigma[k,1]))
    
    fid.close()

########################################################################
# The execution
########################################################################

# Define here a path to save the mcmc samples
path = '/media/valdez/data/local_res/uq_res/'

if __name__ == "__main__":
    
    t0 = time.process_time() # Here start count time
    
    # Define the foam model to study
    # CMG-STARS :: Newtonian just Dry out
    # STARS-FULL :: non-Newtonian Dry out + Shear thinning
    foam_model = 'STARS-FULL'
    
    # Define the data to upload (Synthetic or Smooth)
    test_case  = "Synthetic"
    
    # If the parameter fmcap is fixed then set test_type as "Reduced", otherwise
    # set test_type as "Complete". For simplicity the code works for "Reduced" fit
    test_type  = "Reduced"
    
    # Define the objective function
    # OF1 :: mu_fg_fit
    # OF2 :: mrf_fit
    # OF3 :: full_fit
    type_fit   = "full_fit"
    
    # Get labels and pointers for files
    model_name , param_labels = model_labels(foam_model)
    param_file , data_file , nmax , b , dummy = files_dataset(test_case)

    print('Solving',model_name,'running on PyMC3 v{}'.format(pmc.__version__))
    print('Analyzed dataset',test_case)
    print('Test type',test_type)
    
    if(test_type == 'Reduced'):
        N_var = len(param_labels) - 1
    elif(test_type == 'Complete'):
        N_var = len(param_labels) 
    
    # Load parameters from file
    mprop = load_parameters(param_file)
    core_params = np.array([mprop['swc'],mprop['sgr'],mprop['nw'],mprop['ng'],
                            mprop['kuw'],mprop['kug'],mprop['muw'],mprop['mug'],
                            mprop['u'],mprop['sigma'],mprop['phi'],mprop['kappa'],nmax,b])
    # Load experimental data
    fg, mu_app = load_exp_data(data_file)
    mu_app = mu_app  * 1.0e-3 # convert from cP to Pa.s
    x_tgt, y_tgt = fg, mu_app
    
    # Evaluate the experimental water saturation
    x_sw  = water_saturation(x_tgt,y_tgt,core_params)
    # Instantiate class
    model = fmf_models(core_params)
    # Evaluate the experimental MRF
    m_tgt = mrf_exp(model.frac_flow(x_sw)[3],x_tgt,y_tgt)
    
    # Start the fitting section
    penalty = 1.0
    min_Nca = None
    
    # Get ranges for uniform priors DEFAULT
    if(foam_model == 'CMG-STARS'):
        p0 , p1 , p2 , dummy , dummy , dummy , dummy , dummy , dummy , dummy , penalty = ranges_lmfit('STARS-FULL',test_case,core_params)
        
    elif(foam_model == 'STARS-FULL'):
        p0 , p1 , p2 , p3 , p4 , dummy , dummy , dummy , dummy , dummy , penalty = ranges_lmfit(foam_model,test_case,core_params)

    omega = np.ones(len(y_tgt))
    omega[np.argmax(y_tgt)] = penalty
    
    if(type_fit == 'mrf_fit'):
        fit_name = 'mrf_'
        px , py , pm = 0 , 0 , 1
    elif(type_fit == 'mu_fg_fit'):
        fit_name = 'mu_'
        px , py , pm = 0 , 1 , 0
    elif(type_fit == 'full_fit'):
        fit_name = 'all_'
        px , py , pm = 0 , 1 , 1

    # Redefine the Observed data
    if(type_fit in ['mrf_fit','mu_fg_fit']):
        data_obs = np.array([py * y_tgt* omega + pm * m_tgt* omega])
    elif(type_fit == 'full_fit'):
        data_obs = np.array([py * y_tgt* omega,pm * m_tgt* omega]).T
        
    min_mu_app = y_tgt[0]
    min_Nca    = (min_mu_app * core_params[8]) / core_params[9]
    
    min_Nca = 2.46e-05
    print('fmcap fixed in: %0.3e' % min_Nca)
    
    # Evaluate each foam model
    if(foam_model == 'STARS-FULL'):
        if(test_type == 'Complete'):
            
            if(type_fit == 'full_fit'):
                # In order to execute PyMC3 you must use this Theano function style.
                @as_op(itypes=[tt.dscalar,tt.dscalar,tt.dscalar,tt.dscalar,tt.dscalar], otypes=[tt.dmatrix])
                def th_forward_model(param1,param2,param3,param4,param5):
                    fg,mu_app,mrf,dummy = model.stars_non_Newtonian_model(x_sw,param1,param2,param3,param4,param5)
                    res = np.array([py * mu_app * omega,pm * mrf * omega]).T
                    return res

            elif(type_fit in ['mrf_fit','mu_fg_fit']):
                # In order to execute PyMC3 you must use this Theano function style.
                @as_op(itypes=[tt.dscalar,tt.dscalar,tt.dscalar,tt.dscalar,tt.dscalar], otypes=[tt.dmatrix])
                def th_forward_model(param1,param2,param3,param4,param5):
                    fg,mu_app,mrf,dummy = model.stars_non_Newtonian_model(x_sw,param1,param2,param3,param4,param5)
                    res = np.array([py * mu_app * omega + pm * mrf * omega])
                    return res

        elif(test_type == 'Reduced'):

            if(type_fit == 'full_fit'):
                # In order to execute PyMC3 you must use this Theano function style.
                @as_op(itypes=[tt.dscalar,tt.dscalar,tt.dscalar,tt.dscalar], otypes=[tt.dmatrix])
                def th_forward_model(param1,param2,param3,param4):
                    fg,mu_app,mrf,dummy = model.stars_non_Newtonian_model(x_sw,param1,param2,param3,param4,min_Nca)
                    res = np.array([py * mu_app * omega,pm * mrf * omega]).T
                    return res

            elif(type_fit in ['mrf_fit','mu_fg_fit']):
                # In order to execute PyMC3 you must use this Theano function style.
                @as_op(itypes=[tt.dscalar,tt.dscalar,tt.dscalar,tt.dscalar], otypes=[tt.dmatrix])
                def th_forward_model(param1,param2,param3,param4):
                    fg,mu_app,mrf,dummy = model.stars_non_Newtonian_model(x_sw,param1,param2,param3,param4,min_Nca)
                    res = np.array([py * mu_app * omega + pm * mrf * omega])
                    return res
        
    elif(foam_model == 'CMG-STARS'):
        
        if(type_fit == 'full_fit'):
            # In order to execute PyMC3 you must use this Theano function style.
            @as_op(itypes=[tt.dscalar,tt.dscalar,tt.dscalar], otypes=[tt.dmatrix])
            def th_forward_model(param1,param2,param3):
                fg,mu_app,mrf,dummy = model.stars_Newtonian_model(x_sw,param1,param2,param3)
                res = np.array([py * mu_app * omega,pm * mrf * omega]).T
                return res

        elif(type_fit in ['mrf_fit','mu_fg_fit']):
            # In order to execute PyMC3 you must use this Theano function style.
            @as_op(itypes=[tt.dscalar,tt.dscalar,tt.dscalar], otypes=[tt.dmatrix])
            def th_forward_model(param1,param2,param3):
                fg,mu_app,mrf,dummy = model.stars_Newtonian_model(x_sw,param1,param2,param3)
                res = np.array([py * mu_app * omega + pm * mrf * omega])
                return res

    # here comes mcmc stuff
    basic_model = pmc.Model()
    
    with basic_model:
        # define priors

        if(foam_model == 'CMG-STARS'):
            param0   = pmc.Uniform(param_labels[0], lower = p0[0], upper = p0[1])
            param1   = pmc.Uniform(param_labels[1], lower = p1[0], upper = p1[1])
            param2   = pmc.Uniform(param_labels[2], lower = p2[0], upper = p2[1])
        
        elif(foam_model == 'STARS-FULL'):
            param0   = pmc.Uniform(param_labels[0], lower = p0[0], upper = p0[1])
            param1   = pmc.Uniform(param_labels[1], lower = p1[0], upper = p1[1])
            param2   = pmc.Uniform(param_labels[2], lower = p2[0], upper = p2[1])
            param3   = pmc.Uniform(param_labels[3], lower = p3[0], upper = p3[1])
            if(test_type == 'Complete'):
                param4   = pmc.Uniform(param_labels[4], lower = p4[0], upper = p4[1])
        
        if(foam_model == 'STARS-FULL' and test_type == 'Complete'):
            model_eval = th_forward_model(param0,param1,param2,param3,param4)
        elif(foam_model == 'STARS-FULL' and test_type == 'Reduced'):
            model_eval = th_forward_model(param0,param1,param2,param3)
        elif(foam_model == 'CMG-STARS'):
            model_eval = th_forward_model(param0,param1,param2)
        
        # define the error in the data and likelihoods
        if(type_fit in ['mrf_fit','mu_fg_fit']):
            if(test_case in ['Synthetic','Smooth']):
                error_val = np.amax(data_obs) * 0.05
            else:
                error_val = np.amax(data_obs) * 0.001
            
            sigma_val = pmc.HalfNormal('sigma', sigma=error_val) # here adapt sigma to experiment.
            Y_obs = pmc.Normal('Y_obs', mu=model_eval, sigma=sigma_val, observed=data_obs)
        
        elif(type_fit == 'full_fit'):
            if(test_case in ['Synthetic','Smooth']):
                error_val_mu , error_val_mrf = np.amax(data_obs[:,0]) * 0.05 , np.amax(data_obs[:,1]) * 0.05
            else:
                error_val_mu , error_val_mrf = np.amax(data_obs[:,0]) * 0.001 , np.amax(data_obs[:,1]) * 0.001
            
            sigma = pmc.HalfNormal('sigma', sigma=np.array([error_val_mu , error_val_mrf]),shape=2)
            
            # define the Covariance matrix (if many data observed)
            Cov = np.cov(data_obs[:,0],data_obs[:,1]) * np.power(sigma,2)
            Y_obs = pmc.MvNormal('Y_obs', mu=model_eval, cov=Cov, observed=data_obs)
        
        # instantiate sampler
        step = pmc.Slice()
        
        # draw posterior samples
        trace = pmc.sample(25000, step=step, tune=1000, cores=4)
    
    if(foam_model == 'STARS-FULL'):
        if(test_type == 'Reduced'):
            sample_file  = path + fit_name + test_case + 'starsfull_mcmc_samples.arv'
        elif(test_type == 'Complete'):
            sample_file  = path + fit_name + test_case + 'Complete_starsfull_mcmc_samples.arv'
    elif(foam_model == 'CMG-STARS'):
        sample_file  = path + fit_name + test_case + 'stars_mcmc_samples.arv'

    # Save samples in data files
    save_trace(trace,foam_model,type_fit,sample_file)
    
    # text-based summary of the posteriors
    s = pmc.summary(trace).round(2)
    print(s)

    t1 = time.process_time() # Here end counting time
    
    print("Elapsed time to solve: ",(t1-t0)/60,'minutes')

