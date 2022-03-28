# ****************************************************************************
# Name           : fmf_models.py
# Author         : Andres Valdez
# Version        : 1.0
# Description    : Object oriented implementation to study the Newtonian and non-Newtonian Stars foam models
# Data           : 03-24-2022
# ****************************************************************************

from pre_set import *
from scipy import optimize

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
            x_sw[k] = swc + (1.0 - swc - sgr) * pow( (muw * (1.0 - fg[k])) / (kuw * mu_app[k]) ,1.0/nw)
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
# The execution
########################################################################

if __name__ == "__main__":
    
    t0 = time.process_time() # Here start count time
    print('Testing the classes')
    
    thesis_figs = True
    
    # Benchmark Alvarez2001.pdf
    par_cs  = [50000.0000,0.31009193,245.356108]
    ref_stf = [1.6e+05,0.31,500,0.5,2.46e-05]
    nmax    = 1.91e+13  # From Gassara2017A.pdf
    b       = 1.0 / 3.0 # I fixed in cubic. later will "release it"


    # load parameters from file
    mprop = load_parameters('input_par_Alvarez2001.dat')
    core_params = np.array([mprop['swc'],mprop['sgr'],mprop['nw'],mprop['ng'],
                            mprop['kuw'],mprop['kug'],mprop['muw'],mprop['mug'],
                            mprop['u'],mprop['sigma'],mprop['phi'],mprop['kappa'],nmax,b])
    
    # load experimental data
    fg, mu_app = load_exp_data('Synthetic.dat')
    mu_app = mu_app * 1.0e-3 # convert from cP to Pa.s
    x_tgt, y_tgt = fg, mu_app
    
    # find water saturation
    x_sw = water_saturation(fg, mu_app,core_params)
    
    # Instantiate foam model class
    model = fmf_models(core_params)

    # Evaluate the experimental MRF
    mob_g = model.frac_flow(x_sw)[3]
    mrf_exp = mrf_exp(mob_g,x_tgt,y_tgt)

    # Evaluate each foam models
    sw                            = np.linspace(core_params[0],1.0-core_params[1],1000)
    fg_cs,mu_app_cs,mrf_cs,lt_cs  = model.stars_Newtonian_model(sw,par_cs[0],par_cs[1],par_cs[2])
    fg_cf,mu_app_cf,mrf_cf,lt_cf  = model.stars_non_Newtonian_model(sw,ref_stf[0],ref_stf[1],ref_stf[2],ref_stf[3],ref_stf[4])
    
    # Make the big figures
    fig , axs = plt.subplots(1,2,figsize=(8,4))
    
    axs[0].plot(fg_cs, mu_app_cs*1.0e+03, 'g-',lw=2, label='Newtonian')
    axs[0].plot(fg_cf, mu_app_cf*1.0e+03, 'r-',lw=2, label='non-Newtonian')
    axs[0].plot(x_tgt,y_tgt*1.0e+03,'ko',label='experimental')

    axs[0].set_ylabel("apparent viscosity (cP)")
    axs[0].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    axs[0].set_xlabel("foam quality")
    
    axs[1].plot(fg_cs, mrf_cs, 'g-',lw=2, label='Newtonian')
    axs[1].plot(fg_cf, mrf_cf, 'r-',lw=2, label='non-Newtonian')
    axs[1].plot(x_tgt,mrf_exp,'ko',label='experimental')

    axs[1].legend(loc='best')
    axs[1].set_ylabel("mobility reduction factor")
    axs[1].set_xlabel("foam quality")
    axs[1].ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    axs[1].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    
    plt.tight_layout()
    plt.show()
    plt.clf()

    t1 = time.process_time() # Here end counting time
    
    print("Elapsed time to test the class: ",t1-t0)

