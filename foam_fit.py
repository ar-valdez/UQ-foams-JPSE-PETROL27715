# ****************************************************************************
# Name           : foam_fit.py
# Author         : Andres Valdez - Bernardo Rocha
# Version        : 1.0
# Description    : (deterministic) Fitting of different foam model to experimental data (fg, muapp)
# Data           : 03-24-2022
# ****************************************************************************

from pre_set import *
import lmfit as lmf
from fmf_models import *

import warnings
warnings.filterwarnings("ignore")

np.random.seed(1234)

path = '/media/valdez/data/local_res/mcmc_res/'


########################################################################
# Auxiliary functions
########################################################################
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

def sigma_mcmc(foam_model,test_case,type_fit):
    """
    This function returns (worst) sigmas for weighting objective functions
    """
    if(test_case == 'Synthetic'):
        if(foam_model == 'STARS-FULL' and type_fit in ['mrf_fit','mu_fg_fit']):
            mu_app = 3.940e+01
            mrf    = 4.883e+03
        elif(foam_model == 'STARS-FULL' and type_fit == 'full_fit'):
            mu_app = 1.493e+02
            mrf    = 3.596e-01
    elif(test_case == 'Smooth'):
        if(foam_model == 'STARS-FULL' and type_fit in ['mrf_fit','mu_fg_fit']):
            mu_app = 3.642e+01
            mrf    = 5.387e+03
        elif(foam_model == 'STARS-FULL' and type_fit == 'full_fit'):
            mu_app = 1.608e+02
            mrf    = 4.535e-01
    else:
        mu_app = 1
        mrf    = 1
    
    return mu_app , mrf

def foo_residual(params, sw=None, data=None, foam_model=None, penalty=None, fmcap=None):
    """
    :param params: model parameters
    :param sw: water saturation array (from data)
    :param prop: core sample parameters
    :param foam_model: Name of the model to fit
    """

    # define weights. Eq. 4.18 ThesisMa2012.pdf
    
    x_tgt , y_tgt , m_tgt = data[0] , data[1] , data[2]
    px    , py    , pm    = data[3] , data[4] , data[5]
    sy    , sm            = data[6] , data[7]
    
    omega = np.ones(len(y_tgt))
    omega[np.argmax(y_tgt)] = penalty
    
    
    # Evaluate the model
    if(foam_model == 'CMG-STARS'):
        param_labels = ['fmmob','SF','sfbet']
        fg,mu_app,mrf,dummy = model.stars_Newtonian_model(sw,params[param_labels[0]],params[param_labels[1]],params[param_labels[2]])
    elif(foam_model == 'STARS-FULL'):
        param_labels = ['fmmob','SF','sfbet','epcap','fmcap']
        if(fmcap == None):
            fg,mu_app,mrf,dummy = model.stars_non_Newtonian_model(sw,params[param_labels[0]],params[param_labels[1]],params[param_labels[2]],params[param_labels[3]],params[param_labels[4]])
        else:
            fg,mu_app,mrf,dummy = model.stars_non_Newtonian_model(sw,params[param_labels[0]],params[param_labels[1]],params[param_labels[2]],params[param_labels[3]],fmcap)
    
    #res_x = px * ((fg - x_tgt) / np.amax(x_tgt))
    
    res_y = py * ((mu_app - y_tgt) / (sy * np.amax(y_tgt)))
    res_m = pm * ((mrf - m_tgt) / (sm * np.amax(m_tgt)))
    
    res = res_y + res_m
    return  res * np.sqrt(omega)

########################################################################
# The execution
########################################################################

if __name__ == "__main__":
    
    model_list = ['CMG-STARS','STARS-FULL']
    
    if len(sys.argv) < 5:
        print('Usage: [foam_fit.py] [foam_model] [test_case] [test_type] [type_fit]')
        print('foam_model: CMG-STARS or STARS-FULL' )
        print('test_case: Synthetic or Smooth')
        print('test_type: Complete or Reduced (Only for STARS-FULL)')
        print('type_fit :', ['mu_fg_fit','mrf_fit','full_fit'])
        print(' ')
        sys.exit()
    
    t0 = time.process_time() # Here start count time
    
    foam_model = sys.argv[1]
    test_case  = sys.argv[2]
    test_type  = sys.argv[3]
    type_fit   = sys.argv[4]
    
    if(foam_model not in model_list):
        print('Foam model not implemented')
        print('Available foam models:',model_list)
        sys.exit()

    model_name , param_labels = model_labels(foam_model)
    param_file , data_file , nmax , b , dummy = files_dataset(test_case)
    
    print('Fitting',foam_model,'running on lmfit v{}'.format(lmf.__version__))
    print('Analyzed dataset',test_case)
    
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
    
    # Obtain "errors" from mcmc
    mu_sigma , mrf_sigma = sigma_mcmc(foam_model,test_case,type_fit)

    # Compact all the experimental data and CF parameters
    if(type_fit == 'mrf_fit'):
        px , py , pm = 0 , 0 , 1
    elif(type_fit == 'full_fit'):
        px , py , pm = 0 , 1 , 1
    elif(type_fit == 'mu_fg_fit'):
        px , py , pm = 0 , 1 , 0
    exp_data = [x_tgt,y_tgt,m_tgt,px,py,pm,mu_sigma,mrf_sigma]
    
    # Start the fitting section
    penalty = 1.0
    
    if(foam_model == 'CMG-STARS'):
        p0 , p1 , p2 , i0 , i1 , i2 , penalty = ranges_lmfit(foam_model,test_case,core_params)
        init_val , pval = [i0,i1,i2] , [p0,p1,p2]
    elif(foam_model == 'STARS-FULL'):
        p0 , p1 , p2 , p3 , p4 , i0 , i1 , i2 , i3 , i4 , penalty = ranges_lmfit(foam_model,test_case,core_params)
        init_val , pval  = [i0,i1,i2,i3,i4] , [p0,p1,p2,p3,p4]
    
    min_mu_app = y_tgt[0]
    min_Nca    = (min_mu_app * core_params[8]) / core_params[9]
    
    min_Nca = 2.46e-05
    print('fmcap fixed in: %0.3e' % min_Nca)
 
    # fitting
    params = lmf.Parameters()
    
    if(test_type == 'Complete'):
        for k in range(len(param_labels)):
            params.add(param_labels[k], min=pval[k][0], max=pval[k][1])
        min_Nca = None
    elif(test_type == 'Reduced'):
        for k in range(len(param_labels)-1):
            params.add(param_labels[k], min=pval[k][0], max=pval[k][1])

    # lmfit minimizer 
    foo = lmf.Minimizer(foo_residual, params,
                        fcn_kws={'sw': x_sw, 'data':exp_data, 'foam_model': foam_model, 'penalty': penalty, 'fmcap':min_Nca})
    
    result = foo.minimize(method='differential_evolution')
    
    lmf.report_fit(result)
    
    if(foam_model in ['CMG-STARS','Linear_Kinetic','Grassia2020A']):
        theta = result.params[param_labels[0]].value , result.params[param_labels[1]].value , result.params[param_labels[2]].value
    elif(foam_model in ['STARS-FULL']):
        if(test_type == 'Complete'):
            theta = result.params[param_labels[0]].value , result.params[param_labels[1]].value , result.params[param_labels[2]].value , result.params[param_labels[3]].value , result.params[param_labels[4]].value
        elif(test_type == 'Reduced'):
            theta = result.params[param_labels[0]].value , result.params[param_labels[1]].value , result.params[param_labels[2]].value , result.params[param_labels[3]].value , min_Nca

    # Make the plot comparing fit vs data
    xx_sw = np.linspace(core_params[0],1.0-core_params[1], 1000)
    
    if(foam_model == 'CMG-STARS'):
        param_labels = ['fmmob','SF','sfbet']
        fg,mu_app,mrf,dummy = model.stars_Newtonian_model(xx_sw,theta[0],theta[1],theta[2])
    elif(foam_model == 'STARS-FULL'):
        param_labels = ['fmmob','SF','sfbet','epcap','fmcap']
        fg,mu_app,mrf,dummy = model.stars_non_Newtonian_model(xx_sw,theta[0],theta[1],theta[2],theta[3],theta[4])

    # show fitted
    fig , axs = plt.subplots(1,2,figsize=(10,5))
    
    axs[0].plot(x_tgt,y_tgt*1000,'bo',label='data: ' + data_file[:-4])
    axs[0].plot(fg, mu_app*1000, 'r-',lw=2, label='model: ' + model_name)
    axs[0].set_ylabel("apparent viscosity (cP)")


    axs[1].plot(x_tgt,m_tgt,'bo',label='data: ' + data_file[:-4])
    axs[1].plot(fg, mrf, 'r-',lw=2, label='model: ' + model_name)
    axs[1].set_ylabel("mobility reduction factor")

    axs[0].set_xlabel("foam quality")
    axs[0].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    axs[1].set_xlabel("foam quality")
    axs[1].ticklabel_format(axis="y", style="sci", scilimits=(0,0))

    axs[1].legend(loc='best')
    plt.tight_layout()
    plt.show()
    plt.clf()

    t1 = time.process_time() # Here end counting time
    
    print("Elapsed time to solve deterministic fitting: ",t1-t0)
    
    sys.exit()
    
    for k in range(len(x_tgt)):
        print('{:10.2f}'.format(x_tgt[k]),'{:10.2f}'.format(1000*y_tgt[k]))
    
