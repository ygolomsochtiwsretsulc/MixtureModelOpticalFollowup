__author__='sebastiangrandis'

import numpy as np
from astropy.io import fits
import scipy.stats as st
from astropy.cosmology import FlatLambdaCDM
import time as tm

class MixtureModel(object):
    
    def __init__(self, cata_name, zmin=0.05, zmax=0.8, lam_min=5, cr_min=1e-2, CR_KEY='ML_RATE_0',
                  optical_randoms='Pz_Plambda_random.npy', numz=10, zpiv=0.3, cr_med = 2e-1,
                 constants={}, mapping={}):
        
        self.constants = constants
        self.mapping = mapping
        
        hdulist_erassX = fits.open(cata_name)
        scidata_erassX = hdulist_erassX[1].data
        
        self.z = scidata_erassX['best_z']
        self.rate = scidata_erassX[CR_KEY]
        self.rich = scidata_erassX['lambda_norm']
        self.texp = scidata_erassX['ML_EXP_1']
        
        self.flag =  ( self.z<zmax ) & ( zmin<self.z ) & (self.rate > cr_min) & (self.rich > lam_min) \
                     & (scidata_erassX['IN_XGOOD']==True) & (scidata_erassX['SPLIT_CLEANED']==True) \
                       
        self.flag_fit = ( self.z<1. ) & ( 0.01<self.z ) \
                         & (scidata_erassX['IN_XGOOD']==True) & (scidata_erassX['SPLIT_CLEANED']==True)
        
        self.pretty_cont = (self.z>0.4) & (self.rich<10) & self.flag
        
        with open(optical_randoms, 'rb') as f:
            z_cen = np.load(f)
            PDF_z = np.load(f)
            CFD_z = np.load(f)
            lambda_cen = np.load(f)
            PDF_lambda = np.load(f)
            CDF_lambda = np.load(f)
            
        ill = np.argmin(np.abs(lambda_cen-lam_min))
        print(ill)
            
        self.P_lnl_cont_arr = (PDF_lambda*lambda_cen)[ill:]
        self.lnl_cont_arr = np.log(lambda_cen)[ill:]

        self.P_lnl_cont_arr /= 0.5*np.sum((self.lnl_cont_arr[1:]-self.lnl_cont_arr[:-1])* \
                                          (self.P_lnl_cont_arr[1:]+self.P_lnl_cont_arr[:-1]) )
        
        self.zbins = np.percentile(self.z[self.flag], np.linspace(0, 100, num=numz))
        
        self.numz = numz
        self.zpiv = zpiv
        self.cr_med = cr_med
        self.lam_min = lam_min
        
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        self.dl = cosmo.luminosity_distance(self.z).value /  cosmo.luminosity_distance(zpiv).value
        

    def setup(self):
        
        Xobs = self.rate*(self.texp/1e-1)**0.62
        
        start = tm.time()
        
        kernel = st.gaussian_kde( np.vstack( (self.z[self.flag_fit], np.log10(Xobs)[self.flag_fit]) ) )
        kernel_z = st.gaussian_kde(self.z[self.flag_fit])
        kernel_cont = st.gaussian_kde(np.log10(Xobs)[self.pretty_cont & (Xobs<9e1)])
        
        start = tm.time()
        
        dNtot = kernel( np.vstack( (self.z, np.log10(Xobs)) ) ) / kernel_z( self.z )
        
        dNcont = kernel_cont(np.log10(Xobs))
        
        self.frac = dNcont/dNtot 
        
        pass
    
    def computeLikelihood(self, ctx):
        
        p1 = np.array(ctx.getParams())
        
        return self.getLike(p1)
        
        
    def getLike(self, p1):
        
        L, _, _ = self.get_single_likes(p1)
        
        if (L[self.flag]>0).all() and np.isfinite(L[self.flag]).all():
            return np.sum(np.log(L[self.flag]))
        else:
            return -np.inf
        
    def _vec2params(self, p1):
        
        params = self.constants.copy()
        
        for k, v in self.mapping.items():
            params[k] = p1[v]
            
        return params
    
    
    def get_single_likes(self, p1):
        
        params = self._vec2params(p1)
            
        pred_lam, var = self.scaling_rel(params)
        
        prior = self.cont_prior(params)
             
        scaled_lam = params['alpha_rnd']*np.log(self.rich/self.lam_min) + np.log(self.lam_min)
        P_l_cont = params['alpha_rnd']*np.exp( np.interp( scaled_lam, self.lnl_cont_arr, np.log(self.P_lnl_cont_arr) ) )
        
        if (prior<0).any() or (prior>1).any():
            return -np.inf*np.ones(len(self.rich)), -np.inf*np.ones(len(self.rich)), -np.inf*np.ones(len(self.rich))
        else:
            pp = P_l_cont*prior

            malmquist = st.norm.sf(np.log(self.lam_min), loc=np.log(pred_lam), scale=np.sqrt(var))
            P_sr = 1/np.sqrt(2*np.pi*var)*np.exp( -0.5*np.log(self.rich/pred_lam)**2/var )

            pcl = (1-prior)*P_sr/malmquist

            L_tot = pp + pcl

            return L_tot, pp/L_tot, pcl/L_tot
     
    
    def scaling_rel(self, params):
        
        amp_z = np.zeros(self.numz)
        
        for i in range(self.numz):
            amp_z[i] = params['amp_z_%i'%i]
            
        pred_lam = np.exp(params['lam_med']) * (self.rate/self.cr_med*self.dl**2)**params['alpha_cr'] * \
                       np.exp(np.interp(self.z, self.zbins, amp_z))
        
        poisson_rich = np.where(pred_lam<1, 0, (pred_lam-1)/pred_lam**2)

        var = np.exp(2*params['lns_intr']) + poisson_rich + \
                  (np.exp(params['fudge'])/self.rate/self.texp)**params['alpha_cr']
        
        return pred_lam, var
    
    
    def cont_prior(self, params):
        
        fracs = np.zeros(self.numz)
        
        for i in range(self.numz):
            fracs[i] = params['frac_%i'%i]
            
        A = np.interp(self.z, self.zbins, fracs)
        
        return A*self.frac
    
    
    def draw_mock(self, params, seed=0):
        
        rng = np.random.default_rng(seed)
        
        pred_lam, var = self.scaling_rel(params)
        
        prior = self.cont_prior(params)
        
        lam_rl = st.truncnorm.rvs(np.log(self.lam_min), np.inf, loc=np.log(pred_lam), scale=np.sqrt(var))
            

        
        
        
        
        