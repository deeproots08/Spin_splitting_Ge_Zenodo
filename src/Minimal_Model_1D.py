from config import *

###defining constants##

hbar = 1.055*10**(-34);   #in J-s
ee = 1.602*10**(-19);     #in C
hbar_mev = hbar/ee*1000   # in meV-s
me = 9.11*10**(-31);      #in kg
hbm = hbar**2*1000/2/me/ee/(1e-18) #in meV (nm)**2

import Operators2D as Op

class oneD_Minimal_Model:

    def __init__(self, n, nsm, nsc, my, ax, omega, alphat, delta=0, meff_sc=1, meff_sm = 1 , mu_sc = 0, mu_sm=0, u0=0, uc=0, lambda1=1, lambda2=1, lambda3=1, lambda4=1, lambda5=1, px=False,  epsilon=0):
        
        """
        Initializes a new instance of the oneD_Minimal_Model class.
        
        Args:
            n (int): The number of horizontal lattice points of the system.
            nsm (int): The number of horizontal lattice points of the semiconductor weak link.
            nsc (int): The number of horizontal lattice points of the superconductor on both sides.
            my (int): The number of y subbands to eb considered
            ax (float): The horizontal (x) lattice spacing.
            meff (float): Effective mass 
            alphat (float): spin orbit coupling constant
            u0 (float) : strength of interface barrier
            uc (float) : strength of central barrier
            mu_s (float): chemical potential of the superconducting region of SNS
            mu_sm (float): chemical potential of the semiconductingconducting region of SNS
            omega (float): energy in meV of the y harmonic confinement potential
            lambda1 (float) : Analog switch to turn on/off contribution of kx^3 in csoc, and kx in lsoc.Default=1(on)
            lambda2 (float) : Analog switch to turn on/off contribution of kx^2ky term in csoc, and ky term in lsoc.default=1(on)
            lambda3 (float) : Analog switch to turn on/off contribution of kxky^2 term in csoc.Default=1(on)
            lambda4 (float) : Analog switch to turn on/off contribution of ky^3 term in csoc.Default=1(on)
            lambda5 (float) : Analog switch to make <0|kysq|0> = <1|kysq|1>. Default is <0|kysq|0> != <1|kysq|1> when lamda5 = 1. 
            px (bool, optional): Flag indicating periodicity in x direction. Defaults to False.
            epsilon (float): Kx^4 prefactor to remove higher k states from dipping

        Notes:
            - The ordering of lattice points for kx, kxsq, kxcu(imported for operator class file) is from left to right, bottom to top, similar to x and y axis in 2D graph paper. 
        """
        
        self.n = n
        self.nsm = nsm
        self.nsc = nsc
        self.my = my
        self.ax = ax
        self.meff_sc = meff_sc
        self.meff_sm = meff_sm
        self.omega = omega
        self.alphat = alphat
        self.delta = delta
        self.mu_sc = mu_sc
        self.mu_sm  = mu_sm
        self.u0 = u0
        self.uc = uc
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.lambda4 = lambda4
        self.lambda5 = lambda5
        self.px = px
        self.epsilon = epsilon
        self.onedOP = Op.Operators_2d(self.n, 1, self.ax, 1, self.px, False)

    def Energy_SHO_2D(self, ney=0):

        """
        returns energy of ne th state for a SHO in meV
        Lets not include the z confinement energy- consider that as the reference level
        
        Args:
            ne (int): Energy level whose energy is required, starts from 0 
        
        """

        return self.omega*(ney+ 0.5)
    
    def Ky_iden_pr(self,p,r):

        if p==r:

            return 1
        
        else:

            return 0 
        
    def Ky_pr(self, p, r):

        """
        returns  <p|ky|r>

        Args:
            p, r (int): refers to p and r th sho wavefunction used to calculate <p|ky|r>
        
        """

        N = 1j*cmath.sqrt(self.omega*self.meff_sm/4/hbm)

        if p==r+1 :

            return N*cmath.sqrt(r+1)
        
        elif p==r-1 :

            return -N*cmath.sqrt(r)
        
        else :

            return 0    

    def Kycube_pr(self, p, r):

        """
        returns  <p|ky^3|r>

        Args:
            p, r (int): refers to p and r th sho wavefunction used to calculate <p|ky|r>

        
        """

        N = 1j*cmath.sqrt(self.omega*self.meff_sm/4/hbm)

        if p == r+1 : 

            return -N**3*(3)*(r+1)*cmath.sqrt(r+1)
        
        elif p == r-1 :

            return N**3*3*r*cmath.sqrt(r)
        
        elif p == r+3 :

            return N**3*cmath.sqrt((r+1)*(r+2)*(r+3))
        
        elif p == r-3 :

            return -N**3*cmath.sqrt((r)*(r-1)*(r-2))
        
        else :
            
            return 0 

    def Kysq_pr(self, p, r):
        """
        returns <p|ky^2|r>

        Args:
            p, r (int): refers to p and r th sho wavefunction used to calculate <p|ky|r>

        
        """

        N = 1j*cmath.sqrt(self.omega*self.meff_sm/4/hbm)

        if p == r+2 : 

            return N**2*cmath.sqrt(r+1)*cmath.sqrt(r+2)
        
        elif p == r-2 :

            return N**2*cmath.sqrt((r)*(r-1))
        
        elif p == r:

            return -N**2*(2*r+1)
        
        else :
            
            return 0 

    def chem_pot(self):

        """
        Returns a sparse 2D array of chemical potential of size (nxn) defining the chemical potential landscape in one dimension.
        """

        nsc_data = np.full(self.nsc, self.mu_sc)
        nsm_data = np.full(self.nsm, self.mu_sm)
        remaining_data = np.full(self.n - self.nsc - self.nsm, self.mu_sc)

        data = np.concatenate((nsc_data, nsm_data, remaining_data))

        M = diags(data)

        return M

    def barrier(self, on_both_junc=False):

        """
        Returns sparse array of barrier potential between first sc- sm interface normalised to value of U0(meV-nm)

        Args: 
            on_both_junc (bool) : Boolean, true if barrier is on both junction

        """

        K = self.u0/2/self.ax

        j1 = self.nsc-1

        row = np.array([j1-1, j1, j1+1])
        col = np.array([j1-1, j1, j1+1])
        data = np.array([K/2, K, K/2])

        if on_both_junc==True:

            j2 = self.nsc + self.nsm

            row = np.append(row,np.array([j2-1, j2, j2+1]))
            col = np.append(col,np.array([j2-1, j2, j2+1]))
            data = np.append(data,np.array([K/2, K, K/2]))

        M = coo_matrix((data, (row, col)), shape=(self.n, self.n))

        return M

    def centre_barrier(self):

        """
        Returns sparse array of barrier potential at the centre of josephson junction normalised to value of U0(meV-nm)

        """

        K = self.uc/2/self.ax

        j1 = (2*self.nsc+self.nsm)/2

        row = np.array([j1-1, j1, j1+1])
        col = np.array([j1-1, j1, j1+1])
        data = np.array([K/2, K, K/2])

        M = coo_matrix((data, (row, col)), shape=(self.n, self.n))

        return M

    def zeeman_artificial(self, i, j, s, sp, Ez=0):
        """
        returns  <i,s|H_zeeman|j,sp> assuming magnetic field is in z direction
        
        Args:
            i, j (int): refers to i and j th y subband sho wavefunction 
            s, sp (int){0 or 1}: refers to spin index up and down resp. 
            Ez : Zeeman field energy in meV

        Note: This function is so as to introduce spin splitting artifically. so this just creates spin splitting but doesn't necessarily include 
        all the effects of magnetic field
        
        """

        if i==j and s==sp:

            if s == 0:
                
                return Ez*self.onedOP.iden()
            
            else:

                return -Ez*self.onedOP.iden()

        else:

            return self.onedOP.zer()
        
    def spin_orbit(self, i, j, s, so='cu', kx=0):

        """
        returns  <i,s|H_SO|j,sp>
        
        Args:
            i, j (int): refers to i and j th y subband sho wavefunction 
            s, sp (int){0 or 1}: refers to spin index up and down resp. for spin orbit, sp is opposite of s for spin orbit to be non zero
            so : spin orbit coupling, 'cu' if cubic rashba, 'li' if linear rashba
        
        """

        a = self.Kysq_pr(0,0)
        b = self.Kysq_pr(1,1)

        l1 = a*self.lambda5 + (a+b)/2*(1-self.lambda5)
        l2 = b*self.lambda5 + (a+b)/2*(1-self.lambda5)

        if so == 'cu':
            #print('cu')

            k_minus_cube = self.onedOP.KxCu(kx)*self.Ky_iden_pr(i,j)*self.lambda1 - 1j*3*self.onedOP.KxSq(kx)*self.Ky_pr(i,j)*self.lambda2 - 3*self.onedOP.Kx(kx)*self.Kysq_pr(i,j)*self.lambda3 + 1j*self.Kycube_pr(i,j)*self.onedOP.iden()*self.lambda4
            k_plus_cube =  self.onedOP.KxCu(kx)*self.Ky_iden_pr(i,j)*self.lambda1 + 1j*3*self.onedOP.KxSq(kx)*self.Ky_pr(i,j)*self.lambda2 - 3*self.onedOP.Kx(kx)*self.Kysq_pr(i,j)*self.lambda3 - 1j*self.Kycube_pr(i,j)*self.onedOP.iden()*self.lambda4

            k_minus_cube_l1 = self.onedOP.KxCu(kx)*self.Ky_iden_pr(i,j)*self.lambda1 - 1j*3*self.onedOP.KxSq(kx)*self.Ky_pr(i,j)*self.lambda2 - 3*self.onedOP.Kx(kx)*l1*self.lambda3 + 1j*self.Kycube_pr(i,j)*self.onedOP.iden()*self.lambda4
            k_plus_cube_l1 = self.onedOP.KxCu(kx)*self.Ky_iden_pr(i,j)*self.lambda1 + 1j*3*self.onedOP.KxSq(kx)*self.Ky_pr(i,j)*self.lambda2 - 3*self.onedOP.Kx(kx)*l1*self.lambda3 - 1j*self.Kycube_pr(i,j)*self.onedOP.iden()*self.lambda4
            
            k_minus_cube_l2 = self.onedOP.KxCu(kx)*self.Ky_iden_pr(i,j)*self.lambda1 - 1j*3*self.onedOP.KxSq(kx)*self.Ky_pr(i,j)*self.lambda2 - 3*self.onedOP.Kx(kx)*l2*self.lambda3 + 1j*self.Kycube_pr(i,j)*self.onedOP.iden()*self.lambda4
            k_plus_cube_l2 = self.onedOP.KxCu(kx)*self.Ky_iden_pr(i,j)*self.lambda1 + 1j*3*self.onedOP.KxSq(kx)*self.Ky_pr(i,j)*self.lambda2 - 3*self.onedOP.Kx(kx)*l2*self.lambda3 - 1j*self.Kycube_pr(i,j)*self.onedOP.iden()*self.lambda4
            
            if s==0:

                if i==0 and j==0:

                    return -1j*self.alphat*k_minus_cube_l1
                
                elif i==1 and j==1:

                    return -1j*self.alphat*k_minus_cube_l2

                else:
                
                    return -1j*self.alphat*k_minus_cube
            
            else:

                if i==0 and j==0:

                    return 1j*self.alphat*k_plus_cube_l1
                
                elif i==1 and j==1:

                    return 1j*self.alphat*k_plus_cube_l2
                
                else:

                    return 1j*self.alphat*k_plus_cube

        elif so == 'li':
            #print('li')
            # Linear SOC #
            if i==j:
                ### Note in later definitions, soc coupling is evoked only when s!=s' ###
                ### Hence not coding that statement here ###
                if s ==0 : 

                    return 1j*self.alphat*(self.onedOP.Kx(kx))*self.lambda1
                
                else:

                    return -1j*self.alphat*(self.onedOP.Kx(kx))*self.lambda1
                
            else :

                return self.alphat*(self.Ky_pr(i,j))*self.onedOP.iden()*self.lambda2        
        
        else:

            return self.onedOP.zer()

    def H_is_jsp(self, i, j, s, sp, Ez =0,  so='cu', on_both_junc = False, kx=0):

        """
        returns  <i,s|H|j,sp>
        
        Args:
            i, j (int): refers to i and j th y subband sho wavefunction 
            s, sp (int){0 or 1}: refers to spin index up and down resp.
            Ez (float): Zeeman field energy in meV
            so (str): spin orbit coupling, cu if cubic rashba, li if linear rashba
        
        """
        if s==sp:

            if i==j:

                return hbm/self.meff_sm*self.onedOP.KxSq(kx) + self.Energy_SHO_2D(j)*self.onedOP.iden() - self.chem_pot() + self.barrier(on_both_junc)+ self.zeeman_artificial(i,j,s,sp,Ez)+ self.epsilon*self.onedOP.Kx_quar(kx) + self.centre_barrier()
            
            else:

                return self.onedOP.zer() 
            
        else:

            return self.spin_orbit(i,j,s,so,kx) 
                    
    def HN(self, Ez=0, so='cu', on_both_junc = False,kx=0):

        """
        returns  Particle space Hamiltonian
        
        """

        V = csr_matrix((0,2*self.my*self.n))

        for i in range(self.my):

            for s in range(2):

                H = csr_matrix((self.n,0))

                for j in range(self.my):

                    for sp in range(2):
                        #print(self.H_is_jsp(i,j,s,sp))    
                        H = hstack((H, self.H_is_jsp(i,j,s,sp,Ez, so,on_both_junc,kx)) )
                        #print('A')
                #print(np.shape(H.todense()))
                V = vstack((V,H))
                #print('B')

        return V

    def Del_array(self, phi=0):

        """
        Returns a sparse 2D array of delta of size (nxn) defining the superducting gap landscape in one dimension.
        """

        nsc_data = np.full(self.nsc, self.delta)
        nsm_data = np.full(self.nsm, 0)
        remaining_data = np.full(self.n - self.nsc - self.nsm, self.delta*cmath.exp(1j*phi))

        data = np.concatenate((nsc_data, nsm_data, remaining_data))

        M = diags(data)

        return M

    def Delta_is_jsp(self,i,j,s,sp, phi=0):

        """
        returns  <i,s|Delta|j,sp>
        
        Args:
            i, j (int): refers to i and j th y subband sho wavefunction 
            s, sp (int){0 or 1}: refers to spin index up and down resp.
        
        """
        if i==j and s!=sp:

            if s==0:

                return self.Del_array(phi)
            
            else:

                return -self.Del_array(phi)
            
        else:

            return self.onedOP.zer()

    def Delta_mat(self, phi=0):

        """
        returns <particle| Delta | hole> 
        
        """

        V = csr_matrix((0,2*self.my*self.n))

        for i in range(self.my):

            for s in range(2):

                H = csr_matrix((self.n,0))

                for j in range(self.my):

                    for sp in range(2):
                        #print(self.H_is_jsp(i,j,s,sp))    
                        H = hstack((H, self.Delta_is_jsp(i,j,s,sp,phi)) )
                        #print('A')
                #print(np.shape(H.todense()))
                V = vstack((V,H))
                #print('B')


        return V

    def H_sys(self,phi=0,Ez=0, so = 'cu', on_both_junc= False,kx=0):

        ### - kx matters only in case of periodic 

        M = bmat([
           [self.HN(Ez, so,on_both_junc,kx), self.Delta_mat(phi)],
           [-self.Delta_mat(phi).conjugate(), -self.HN(Ez,so,on_both_junc,-kx).conjugate()]
       ]) 

        return M
