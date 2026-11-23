from config import *

import Operators2D as Op


class twoD_Minimal_Model:

    def __init__(self, n, nsm, nsc, m, ax, ay, omega=0, alphat=0, delta=0, meff_sc=1, meff_sm = 1 , mu_sc = 0, mu_sm=0, dis_mean=0, dis_var=1, px=False, py=False,  epsilon=0):

        """
        Initializes a new instance of the oneD_Minimal_Model class.
        
        Args:
            n (int): The number of horizontal lattice points of the system.
            nsm (int): The number of horizontal lattice points of the semiconductor weak link.
            nsc (int): The number of horizontal lattice points of the superconductor on both sides.
            m (int): The number of vertical lattive points
            ax (float): The horizontal (x) lattice spacing.
            ay (float): The vertical (y) lattice spacing.
            meff (float): Effective mass 
            alphat (float): spin orbit coupling constant
            mu_s (float): chemical potential of the superconducting region of SNS
            mu_sm (float): chemical potential of the semiconductingconducting region of SNS
            px (bool, optional): Flag indicating periodicity in x direction. Defaults to False.
            epsilon (float): Kx^4 prefactor to remove higher k states from dipping

        Notes:
            - The ordering of lattice points for kx, kxsq, kxcu(imported for operator class file) is from left to right, bottom to top, similar to x and y axis in 2D g

        """

        self.n = n
        self.nsm = nsm
        self.nsc = nsc
        self.m = m
        self.ax = ax
        self.ay = ay
        self.omega = omega
        self.meff_sc = meff_sc
        self.meff_sm = meff_sm
        self.alphat = alphat
        self.delta = delta
        self.mu_sc = mu_sc
        self.mu_sm  = mu_sm
        self.dis_mean = dis_mean
        self.dis_var = dis_var
        self.px = px
        self.py = py
        self.epsilon = epsilon
        self.twodOP = Op.Operators_2d(self.n, self.m, self.ax, self.ay, self.px, self.py)
        self.onedOP =  Op.Operators_2d(1, self.m, 1, self.ay, self.px, self.py)

    def ysq_matrix(self):

        row = np.empty(0)
        col = np.empty(0)
        data = np.empty(0)

        shift = (((self.m//2))*self.ay)

        for j in range(self.m):

            for i in range(self.n):
            
                row = np.append(row, j*self.n + i)
                col = np.append(col, j*self.n + i)
                data = np.append(data, ((j)*self.ay-shift)**2)

        M = csr_matrix((data,(row,col)),shape=(self.m*self.n, self.m*self.n))

        return M

    def chem_pot(self):

        """
        Returns a sparse 2D array diagonal array of chemical potential of size (nmxnm) defining the chemical potential landscape in two dimension.
        """

        nsc_data = np.full(self.nsc, self.mu_sc)
        nsm_data = np.full(self.nsm, self.mu_sm)
        remaining_data = np.full(self.n - self.nsc - self.nsm, self.mu_sc)

        data_one_row = np.concatenate((nsc_data, nsm_data, remaining_data))

        data_lattice = np.tile(data_one_row, self.m)

        M = diags(data_lattice)

        return M

    def dis_pot_matrix(self, dis_mag=0):

        """
        Generate a sparse diagonal matrix of size mnXmn 
        gives onsite disorder potential sampled from gaussian for the semiconducting region only. 
        onsite disorder potential for SC region is taken as zero

        Returns:
        dis_pot_matrix (csr_matrix): Sparse diagonal matrix mnXmn

        """
        final_array = []

        for _ in range(self.m):
            # Generate random values for the non-zero entries
            values = [random.gauss(self.dis_mean, self.dis_var) for _ in range(self.nsm)]
            values = np.array(values)

            # Create the 1D array for this iteration
            array = np.concatenate((np.zeros(self.nsc), values, np.zeros(self.nsc)))

            # Append the array to the final array
            final_array = np.append(final_array, array)

        # Convert final_array to a NumPy array
        final_array = np.array(final_array, dtype=float)

        # Multiply by magnitude
        final_array *= dis_mag

        # Create a sparse diagonal matrix from the final array
        dis_pot_matrix = diags(final_array, 0, format='csr')

        return dis_pot_matrix

    def Disorder_full(self, uncorr_var, lmbda=0, seed=None):
        """
        Returns a mnXmn sparse matrix with diagonal terms giving disorder potential.

        The disorder on each site is sampled from a normal distribution with variance uncorr_var and mean zero.
        This method assumes disorder is throughout the Josephson junction including superconductors, unlike dis_pot_mat method.
        
        Args: 
            uncorr_var (float): Variance of the standard normal distribution from which disorder is sampled.
            lmbda (float): To modify the magnitude of disorder.
            seed (int, optional): Seed for the random number generator to ensure reproducibility.
        """

        # Set the random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)

        # Generate Gaussian distributed random numbers scaled by lambda
        gaussian_array = lmbda * np.random.normal(loc=0, scale=math.sqrt(uncorr_var), size=self.m*self.n)

        # Create a diagonal matrix of disorder potentials
        dis_array = diags(gaussian_array)

        return dis_array

    def charge_trap_pot(self, x_ct= 0, y_ct=0, z_ct=50, z_im=50.2, x=0, y=0, z=0):

        """
        Returns a potential due to charge trap given x, y, z co-ordinates of charge trap and x, y, z co-orfinates of points 
        
        Args: 
            x_ct (float): x co-ordinate of charge trap
            y_ct (float): y co-ordinate of charge trap
            z_ct (float): z co-ordinate of charge trap
            z_im (float): z co-ordinate of image charge
            x (float): x co-ordinate of point where potential needs to be found
            y (float): y co-ordinate of point where potential needs to be found
            z (float): z co-ordinate of point where potential needs to be found

        Note:

            Take qd reference level as z=0 and take other z's accordingly
 
        """

        er = 11.7           ### dieelctric constant of Si

        x_im = x_ct         ### x image same as x charge trap 
        y_im = y_ct         ### y image same as y charge trap

        dis_a = abs(np.sqrt((x_ct-x)**2 + (y_ct-y)**2 + (z_ct-z)**2))
        dis_b = abs(np.sqrt((x_im-x)**2 + (y_im-y)**2 + (z_im-z)**2))

        #print(dis_a, dis_b)

        return 1.13 * k0_e/er * (1/dis_a - 1/dis_b)     ### in meV units, ee = 1 
    
    def charge_pot_lattice(self, x_ct= 0, y_ct=0, z_ct=50, z_im=50.2, z_qw=0):

        """
        Returns a mnXmn sparse array where sc region has zero potential and semiconductor region has potential due to charge trap 
        
        Args: 
            x_ct (float): x co-ordinate of charge trap
            y_ct (float): y co-ordinate of charge trap
            z_ct (float): z co-ordinate of charge trap
            z_im (float): z co-ordinate of image charge
            z_qw (float): z co-ordinate of quantum well or the plane of 2DHG 

        Note:
            Take qd reference level as z=0 and take other z's accordingly

        """

        row = []
        col = []
        data = []

        for i in range(self.nsc, self.nsm+self.nsc):

            for j in range(self.m):

                x_coord =  (i)*self.ax
                y_coord =  (j)*self.ay

                row.append(i + self.n*j)    
                col.append(i + self.n*j)              
                data.append(self.charge_trap_pot(x_ct, y_ct, z_ct, z_im, x_coord, y_coord, z_qw))

        M = csr_matrix((data, (row, col)), shape=( self.m * self.n, self.m * self.n))

        return M

    def spin_orbit(self, s, so='cu', kx=0, ky=0):

        """
        returns  <s|H_SO|j> i.e a matris in xy lattice basis i.e. a nmXnm matrix
        
        Args:
            s, sp (int){0 or 1}: refers to spin index up and down resp. for spin orbit, sp is opposite of s for spin orbit to be non zero
            so : spin orbit coupling, 'cu' if cubic rashba, 'li' if linear rashba
        
        """

        if so == 'cu':
            #print('cu')
            
            if s==0:

                return self.twodOP.H_cubic_so(self.alphat, kx, ky)
            
            else:

                return self.twodOP.H_cubic_so(self.alphat, kx, ky).transpose().conjugate()

        elif so == 'li':
            #print('li')
            # Linear SOC #
            ### Note in later definitions, soc coupling is evoked only when s!=s' ###
            ### Hence not coding that statement here ###
            if s ==0 : 

                return self.twodOP.H_linear_so(self.alphat, kx, ky)
            
            else:

                return self.twodOP.H_linear_so(self.alphat, kx, ky).transpose().conjugate()
                   
        else:

            return self.twodOP.zer()

    def H_same_spin(self, x_ct= 0, y_ct=0, z_ct=50, z_im=50.2, z_qw=0, f_ct =0, dis_mag=0, uncorr_var=1, lmbda=0, seed=None, kx=0, ky=0):

        return (hbm/self.meff_sm* self.twodOP.KxSq(kx) + hbm/self.meff_sm* self.twodOP.KySq(ky) - self.chem_pot() + 1/4/hbm*self.meff_sm*self.omega**2*self.twodOP.ysq() + f_ct*self.charge_pot_lattice(x_ct, y_ct, z_ct, z_im, z_qw) +self.dis_pot_matrix(dis_mag) +self.Disorder_full(uncorr_var, lmbda, seed) )
 
    def H_s_sp(self, s, sp, so='cu', x_ct= 0, y_ct=0, z_ct=50, z_im=50.2, z_qw=0, f_ct=0, dis_mag=0, uncorr_var=1, lmbda=0, seed= None, kx=0, ky=0):

        """
        returns  <s|H|sp>
        
        Args: 
            s, sp (int){0 or 1}: refers to spin index up and down resp.
            so (str): spin orbit coupling, cu if cubic rashba, li if linear rashba
        
        """

        if s==sp:

            return self.H_same_spin(x_ct, y_ct, z_ct, z_im, z_qw, f_ct, dis_mag,uncorr_var, lmbda, seed, kx, ky)
        
        else:

            return self.spin_orbit(s, so, kx, ky)

    def H_particle(self, so = 'cu', x_ct= 0, y_ct=0, z_ct=50, z_im=50.2, z_qw=0, f_ct=0, dis_mag=0, uncorr_var=1, lmbda=0, seed=None, kx=0, ky=0):

        """
        returns  Particle space Hamiltonian
        
        """

        V = csr_matrix((0,2*self.m*self.n))

        for s in range(2):

            H = csr_matrix((self.n*self.m,0))

            for sp in range(2):

                H = hstack( ( H, self.H_s_sp(s, sp, so, x_ct, y_ct, z_ct, z_im, z_qw, f_ct, dis_mag, uncorr_var, lmbda, seed, kx, ky) ) )

            V = vstack((V,H))

        return V

    def Del_array(self, phi=0):

        """
        Returns a sparse 2D array of delta of size (nmxnm) defining the superducting gap landscape in two dimension.
        """

        nsc_data = np.full(self.nsc, self.delta)
        nsm_data = np.full(self.nsm, 0)
        remaining_data = np.full(self.n - self.nsc - self.nsm, self.delta*cmath.exp(1j*phi))

        data = np.concatenate((nsc_data, nsm_data, remaining_data))

        data_full_lattice = np.tile(data, self.m)

        M = diags(data_full_lattice)

        return M                

    def Delta_s_sp(self,s,sp, phi=0):

        """
        returns  <s|Delta|sp> mnXmn sparse matrix
        
        Args: 
            s, sp (int){0 or 1}: refers to spin index up and down resp.
        
        """
        if s!=sp:

            if s==0:

                return self.Del_array(phi)
            
            else:

                return -self.Del_array(phi)
            
        else:

            return self.twodOP.zer()

    def Delta_mat(self, phi=0):

        """
        returns <particle| Delta | hole> 
        
        """

        V = csr_matrix((0,2*self.m*self.n))

        for s in range(2):

            H = csr_matrix((self.n*self.m,0))

            for sp in range(2):
                #print(self.H_is_jsp(i,j,s,sp))    
                H = hstack( ( H, self.Delta_s_sp(s,sp,phi) ) )
                
            V = vstack((V,H))

        return V
    
    def H_sys(self,phi=0, so = 'cu', x_ct= 0, y_ct=0, z_ct=50, z_im=50.2, z_qw=0, f_ct=0, dis_mag=0, uncorr_var=1, lmbda=0, seed=None, kx=0, ky=0):

        ### -kx, -ky of hole space matrix only matters in case of periodic         

        M = bmat([
           [self.H_particle( so, x_ct, y_ct, z_ct, z_im, z_qw, f_ct, dis_mag, uncorr_var, lmbda, seed, kx, ky), self.Delta_mat(phi)],
           [-self.Delta_mat(phi).conjugate(), -self.H_particle( so, x_ct, y_ct, z_ct, z_im, z_qw, f_ct, dis_mag, uncorr_var, lmbda, seed, -kx, -ky).conjugate()]
       ]) 

        return M
    
    def Correlation_2D(self, A, i, ip, j, jp, dl=1):

        """
            Returns correlation function between point labelled by i, j on a n X m lattice
            
            Args:
                A (float) : normalizing ocnstant for correlation function
                i, j (int) : index of two poins between which correlation function needs to be calculated
                dl (float) : decay length of correlation function 
            
        """

        #x_dis = ((i%n) - (j%n))*ax   ## x distance between two points on lattice with index i,j labelled in graph way 

        #y_dis = ((i//n) - (j//n))*ay   ## y distance between two points on lattice with index i,j labelled in graph way 

        x_dis = (i -ip)*self.ax   ## x distance between two points on lattice with index i,j labelled in graph way 

        y_dis = (j -jp)*self.ay   ## y distance between two points on lattice with index i,j labelled in graph way 

        dis_sq = x_dis**2 + y_dis**2

        decay_length_sq = dl**2

        return A**2 * np.exp(-dis_sq/decay_length_sq)
    
    def variance_of_equivalent_uncorrelated_disorder(self, WV, A, dl=1):
        
        """
        Calculate the equivalent variance of uncorrelated disorder.
        
        Args:
            WV (np.ndarray): Wavefunction array of shape (n*m, 1).
            A (float): Normalizing constant for the correlation function.
            dl (float): Decay length of the correlation function.
        
        Returns:
            float: Variance of equivalent uncorrelated disorder.
        """
        WVb = WV.reshape((4, self.m, self.n))
        WVb_squared = np.abs(WVb)**2
        
        # Calculate WVb_outer using np.einsum
        WVb_outer = np.einsum('kij,lmn->kijlmn', WVb_squared, WVb_squared)
        print(WVb_outer.shape)

        # Generate the indices and calculate the correlation function
        j, i, jp, ip = np.meshgrid(np.arange(self.m), np.arange(self.n), np.arange(self.m), np.arange(self.n), indexing='ij')
        corr_function = self.Correlation_2D(A, i, ip, j, jp, dl)
        print(corr_function.shape)

        # Multiply WVb_outer to correlation function
        resulting_matrix = np.einsum('kijlmn, ijmn -> kl', WVb_outer, corr_function)

        #calculate the sum
        var_corr = np.sum(resulting_matrix)*self.ax**2*self.ay**2
        
        # Calculate the sum of squared terms separately for normalization
        norm_factor = np.sum(np.einsum('kij,lij->kl', WVb_squared, WVb_squared))*self.ax*self.ay

        # Compute the equivalent uncorrelated disorder variance
        eqv_uncorr =  var_corr / norm_factor

        return eqv_uncorr, var_corr, norm_factor

    def SHO_wvfn(self, ne, yj=0):

        """
        Return Phi__ne(yj) where yj is the jth index and ne is the quantum number of SHO wavefunction and the whole array
        """

        # define constant of SHo wavefunctions 

        kappa = self.omega/hbm*self.meff_sm
        #kappa = self.omega/hbm         ### No meff here 
        normalization_const = 1/math.sqrt(2**ne * math.factorial(ne))*(kappa/2/math.pi)**(0.25)*self.ay**(0.5)

        # Create the Hermite polynomial 
        coefficients = np.zeros(ne+1)
        coefficients[ne] = 1  # Setting the leading coefficient as per the physicist's definition
        
        H_n = Hermite(coefficients)

        shift = (((self.m//2))*self.ay) ### same shift from as the ysq_matrix 

        #print('SHIFT Y_array: ', shift )

        indices = np.arange(self.m)

        y_array = (indices ) * self.ay  - shift

        y_array_n = math.sqrt(kappa/2)*y_array

        Herm_array = H_n(y_array_n)

        #return normalization_const*np.exp(-kappa/4*y_array**2)*Herm_array, y_array

        wvfn = normalization_const*np.exp(-kappa/4*y_array**2)*Herm_array
        return wvfn[yj], wvfn      

    def U_dag(self, my):
        
        """
        Returns sparse matrix of shape (my * n, m * n)

        Args:
            my: No. of y subbands 
        """
        row = []
        col = []
        data = []        

        for i in range(self.n):
            for j in range(my):
                d, p = self.SHO_wvfn(ne=j, yj=0)
                for k in range(self.m):
                    row.append(j * self.n + i)    
                    col.append(k * self.n + i)               
                    data.append(p[k])

        M = csr_matrix((data, (row, col)), shape=(my * self.n, self.m * self.n))

        return M
    
    def U_dag_ph(self, my=1):

        M = bmat([
            [ self.U_dag(my), None],
            [None, self.U_dag(my).conjugate()],
        ])

        return M
    
    def U_num(self, my, kx=0, ky=0):

        """
        Returns U matrix using eigenvalues found from numerical model 
        """
        

        H = 1/4/hbm*self.meff_sm*self.omega**2*self.onedOP.ysq() + hbm/self.meff_sm* self.onedOP.KySq(ky) 

        E, V = eigsh(H, k=my, sigma=0, return_eigenvectors=True)

        sorted_index = np.argsort(E)

        sorted_V = V[:, sorted_index]

        sorted_E = E[sorted_index]

        #print(sorted_E)

        #print(sorted_V.shape)

        row = []
        col = []
        data = []

        for i in range(self.n):

            for j in range(self.m):

                for k in range(my):

                    row.append(i + self.n*j)    
                    col.append(i + self.n*k)              
                    data.append(sorted_V[j, k])

        M = csr_matrix((data, (row, col)), shape=( self.m * self.n, my * self.n))

        return M
    
    def U_num_ph(self, my=1):

        M = bmat([
            [ self.U_num(my), None],
            [None, self.U_num(my).conjugate()],
        ])

        return M

    def Hamiltonian_simplified(self, x_ct= 0, y_ct=0, z_ct=50, z_im=50.2, z_qw=0, f_ct=0, dis_mag=0, uncorr_var=1, lmbda=0, seed=None, kx=0, ky=0, phi=0):

        M = bmat([
           [self.H_same_spin(x_ct, y_ct, z_ct, z_im, z_qw, f_ct, dis_mag, uncorr_var, lmbda, seed, kx, ky ), self.Del_array(phi) ],
           [self.Del_array(phi).conjugate().transpose(),-self.H_same_spin(x_ct, y_ct, z_ct, z_im, z_qw, f_ct, dis_mag, uncorr_var, lmbda, seed, -kx, -ky).conjugate() ]
       ]) 

        return M

    def Ham_changed_basis(self,  x_ct= 0, y_ct=0, z_ct=50, z_im=50.2, z_qw=0, f_ct=0, dis_mag=0, uncorr_var=1, lmbda=0, seed=None, kx=0, ky=0, phi=0, my=1):

        S = self.U_num_ph(my)         ### used u num operator to change basis as it is more accurate 

        S_dag = S.conjugate().transpose()

        H = self.Hamiltonian_simplified(x_ct, y_ct, z_ct, z_im, z_qw, dis_mag, f_ct, uncorr_var, lmbda, seed, kx, ky,phi)

        return S_dag.dot(H).dot(S)
