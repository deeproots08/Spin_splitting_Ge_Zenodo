import math
import cmath
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix,  bmat, diags, identity, issparse, vstack, hstack
from scipy.sparse.linalg import eigsh,eigs


###defining universal constants##

#hbar = 1.055*10**(-34);   #in J-s
#ee = 1.602*10**(-19);     #in C
#hbar_mev = hbar/ee*1000   # in meV-s
#me = 9.11*10**(-31);      #in kg
#hbm = hbar**2*1000/2/me/ee/(1e-18) #in meV (nm)**2

from config import *

### defining germanium constants - Winkler SOC book ###

gamma1 = 13.38
gamma2 = 4.24
gamma3 = 5.69
#kappa = 3.41
#q = 0.06

'''
Note: The effective heavy hole model has been derived assuming quantum well potential i z direction and  treating the Ez term in z direction 
as  perturbative. Electric field values and z direction terms, if any, should be taken in accordance to the above.
'''
class Eff_hh_model_cubic_param:

    def __init__(self, g1 = gamma1, g2 = gamma2, g3 = gamma3, m=1, Lz=10, E_strain=0):

        """
        Initializes a new instance of the Eff_hh_model_cubic_param class.
        
        Args:

            g1, g2, g3 : material constants, defaults to Germanium constants
            m : z subband quantum no. , defaults to 1. 
            m refers to z subband subspace on which Hamiltonian is projected, for effective Heavy hole model, we are projecting on one z subband only
            Lz : z confinement length in nm
            E_strain : strain energy in meV. Defaults to 0

        """

        self.g1 = g1 
        self.g2 = g2 
        self.g3 = g3 
        self.m = m 
        self.Lz = Lz 
        self.Es = E_strain        

    def Em(self):

        '''
        returns Energy of mth heavy hole state on which we are projecting
        '''

        return (self.g1-2*self.g2)*hbm*math.pi**2*self.m**2/self.Lz**2 - self.Es

    def Ep_hh(self,p):

        '''
        returns Energy of p_th heavy hole state

        '''

        return (self.g1-2*self.g2)*hbm*math.pi**2*p**2/self.Lz**2 - self.Es

    def Ep_lh(self,p):

        '''
        returns Energy of p_th light hole state

        '''

        return (self.g1+2*self.g2)*hbm*math.pi**2*p**2/self.Lz**2 + self.Es

    def Del_E_phh(self,p):

        return self.Em() - self.Ep_hh(p) 
    
    def Del_E_plh(self,p):

        return self.Em() - self.Ep_lh(p)

    def Del_E_olh(self):

        return self.Em() - self.Ep_lh(1)

    def Eff_mass(self, pmax = 10):

        M = self.g1 + self.g2

        for p in range(pmax):

            p2 = 2*p
            M = M + 12* self.g3**2 * 16*p2**2/(p2**2 -1)**2/self.Lz**2/self.Del_E_plh(p2)

        return M**(-1)

    def cubicSO_coeff(self, Ef, pmax ):

        """
        Returns cubic spin orbit coupling factor
        
        Args:

            Ef (float) : Electric field in meV/nm
            pmax : maximum index till which you want to sum 
        """

        M = 0
        N = hbm**2*(3*self.g3)*(self.g2 + self.g3)*Ef*64/math.pi**2

        for p in range(pmax):

            p2 = 2*p
            A = 1/self.Del_E_phh(p2)/self.Del_E_olh() - 1/self.Del_E_phh(p2)/self.Del_E_plh(p2) -1/self.Del_E_olh()/self.Del_E_plh(p2)
            M = M + N*p2**2/(p2**2-1)**3*A

        return M





