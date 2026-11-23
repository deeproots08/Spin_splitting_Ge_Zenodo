from config import *

import Operators2D as Op

class Model:

    ###THIS IS CODE FOR EFFECTIVE  ONE SUBBAND MODEL FOR LINEAR SPIN ORBIT COUPLING(LSOC) ###
    ###EFFECTIVE ONE SUBBAND MODEL FOR CUBIC SPIN ORBIT COUPLING(CSOC) IS NOT REQUIRED AS MINIMAL MODEL 1D EXPLPAINS THAT WELL. THIS IS BECAUSE INTRA SUBBAND EFFECTES ARE NOT IMPORTANT FOR CUBIC CASE ###

    def __init__(self, n, nsm, nsc, ax, omega, alphat, delta=0, meff=1, mu = 0, px=False, phi=0, kx0=0):

        """
        Initializes a new instance of the Model class.
        
        Args:
            n (int): The number of horizontal lattice points of the system.
            nsm (int): The number of horizontal lattice points of the semiconductor weak link.
            nsc (int): The number of horizontal lattice points of the superconductor on both sides.
            ax (float): The horizontal (x) lattice spacing.
            omega (float): y subband confinement energy 
            alphat (float): spin orbit coupling constant
            delta (float): superconducting gap (in meV)
            meff (float): Effective mass
            mu (float): chemical potential of the system (in meV)
            px (bool): toggles periodicity on/off
            phi (float): superconduncting phase difference across the Josephson junction
            kx0 (float): value of kx, used only if there is periodicity in x direction( i.e. px==True)
            
        """    

        self.n = n
        self.nsm = nsm
        self.nsc = nsc
        self.ax = ax
        self.omega = omega
        self.alphat = alphat
        self.delta = delta
        self.meff = meff
        self.mu = mu
        self.px = px
        self.phi = phi
        self.kx0 = kx0

        self.iden = Op.Operators_2d(self.n, 1, self.ax, 1, self.px, False).iden()
        self.kx = Op.Operators_2d(self.n, 1, self.ax, 1, self.px, False).Kx(self.kx0)
        self.kxsq = Op.Operators_2d(self.n, 1, self.ax, 1, self.px, False).KxSq(self.kx0)
        self.kxcu = Op.Operators_2d(self.n, 1, self.ax, 1, self.px, False).KxCu(self.kx0)

        self.omega_minus = self.omega/2
        self.omega_plus = self.omega
        self.kappa = math.sqrt(self.omega*self.meff/4/hbm)
        self.total = math.sqrt(self.omega_minus**2+(self.alphat*self.kappa)**2)
        self.cos = self.omega_minus/self.total
        self.sin = self.alphat*self.kappa/self.total
        self.mu_lattice = self.mu*self.iden

    def HA(self):

        return hbm/self.meff*self.kxsq + self.omega_plus*self.iden - self.mu_lattice -self.total*self.iden - (self.alphat*self.sin)**2*self.kxsq/2/self.total

    def HB(self):

        return -self.alphat*self.kx*self.cos + self.alphat**3*self.sin**2*self.cos*self.kxcu/2/self.total**2
    
    def Delta_matrix(self):


        del_nsc1 = np.full(self.nsc, self.delta)
        del_nsc2 = np.full(self.nsc, self.delta*cmath.exp(1j*self.phi))
        del_nsm = np.full(self.nsm, 0 )

        del_array = np.concatenate((del_nsc1, del_nsm, del_nsc2))

        delta_lattice = diags(del_array)

        return delta_lattice
    
    def H_p(self):

        ## particle part of Hamiltonian

        H_particle = bmat([
                [self.HA() + self.HB(), None],
                [None, self.HA()-self.HB()]
            ])
        
        return H_particle

    def H_sys(self):

        H = bmat([
                [self.HA() + self.HB(), None, None, self.Delta_matrix()],
                [None, self.HA()-self.HB(), -self.Delta_matrix(), None],
                [None,-coo_matrix.conjugate(self.Delta_matrix()), -self.HA()+self.HB(), None],
                [coo_matrix.conjugate(self.Delta_matrix()), None, None, -self.HA()-self.HB()]
            ])

        return H
    
    def Meff_eff(self):

        meff_eff_inv = 1/self.meff - self.alphat**2*self.sin**2/2/self.total/hbm

        return 1/meff_eff_inv
    
    def Mass_inversion_pt(self):

        print('valid only if alpha*kappa << Omega_minus')

        print('alpha-kappa( in meV) = ', self.alphat*self.kappa)
        print('omega_minus( in meV) = ', self.omega_minus)

        return math.pow((hbm/self.meff*2*self.omega_minus**3/self.kappa**2),0.25)


            