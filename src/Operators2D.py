from config import *

class Operators_2d:

    """
    Represents a 2d operators discretized over a 2d lattice.

    Attributes:
    n (int): The number of horizontal lattice points.
    m (int): The number of vertical lattice points.
    ax (float): The horizontal (x) lattice spacing in nm.
    ay (float): The vertical (y) lattice spacing in nm.
    px (bool): Flag indicating periodicity in x direction.
    py (bool): Flag indicating periodicity in y direction.

    """
    
    def __init__(self, n, m, ax, ay, px=False, py=False):

        """
        Initializes a new instance of the Operators_2d class.
        
        Args:
            n (int): The number of horizontal lattice points.
            m (int): The number of vertical lattice points.
            ax (float): The horizontal (x) lattice spacing.
            ay (float): The vertical (y) lattice spacing.
            px (bool, optional): Flag indicating periodicity in x direction. Defaults to False.
            py (bool, optional): Flag indicating periodicity in y direction. Defaults to False.
        
        Notes:
            - The ordering of lattice points is from left to right, bottom to top, similar to a 2D graph.
        """

        
        self.n = n
        self.m = m
        self.ax = ax
        self.ay = ay
        self.px = px
        self.py = py
        self.hbar = 1.055*10**(-34);   #in J-s
        self.ee = 1.602*10**(-19);     #in C
        self.me = 9.11*10**(-31);      #in kg
        
    def iden(self):

        """
        Returns the sparse representation of the identity operator.

        Returns:
            scipy.sparse.coo_matrix: Sparse matrix representation of the identity operator.
        """


        return identity(self.m*self.n)

    def zer(self):

        """
        Returns the sparse representation of the zero operator.

        Returns:
            scipy.sparse.coo_matrix: Sparse matrix representation of the identity operator.
        """
        return csr_matrix((self.m*self.n,self.m*self.n))

    def Kx(self, kx=0):

        """
        Returns the sparse representation of the Kx operator(mnXmn matrix).

        Args:
            kx (float, optional): The value of kx. Defaults to 0.

        Returns:
            scipy.sparse.coo_matrix: Sparse matrix representation of the Kx operator.
        """

        row = []
        col = []
        data = []

        for j in range(self.m):
            for i in range(self.n):
                if i + 1 < self.n:
                    row.append(j * self.n + i)
                    col.append( j * self.n + i + 1)
                    data.append(-1j / (2 * self.ax))
                    
                if i - 1 >= 0:
                    row.append(j * self.n + i)
                    col.append( j * self.n + i - 1)
                    data.append( 1j / (2 * self.ax))

            ### To account for how boundary point connect in case of periodic operator ### 
            if self.px:
                row.append(j * self.n + self.n - 1)
                col.append(j * self.n)
                data.append(-1j / (2 * self.ax) * cmath.exp(1j * kx * self.n * self.ax))

                row.append(j * self.n)
                col.append(j * self.n + self.n - 1)
                data.append(1j / (2 * self.ax) * cmath.exp(-1j * kx * self.n * self.ax))

        M = coo_matrix((data, (row, col)), shape=(self.m * self.n, self.m * self.n))
        return M
    
    def Ky(self, ky=0):

        """
        Returns the sparse representation of the Ky operator.

        Args:
            ky (float, optional): The value of ky. Defaults to 0.

        Returns:
            scipy.sparse.coo_matrix: Sparse matrix representation of the Ky operator.
        """


        row = []
        col = []
        data = []

        for i in range(self.n):
            for j in range(self.m):
                if j + 1 < self.m:
                    row.append( j * self.n + i)
                    col.append((j + 1) * self.n + i)
                    data.append( -1j / (2 * self.ay))

                if j - 1 >= 0:
                    row.append(j * self.n + i)
                    col.append( (j - 1) * self.n + i)
                    data.append(1j / (2 * self.ay))

            ### To account for how boundary point connect in case of periodic operator ###
            if self.py:
                row.append( (self.m - 1) * self.n + i)
                col.append( 0 * self.n + i )
                data.append( -1j / (2 * self.ay) * cmath.exp(1j * ky * self.m * self.ay))

                row.append( 0 * self.n + i)
                col.append( (self.m - 1) * self.n + i)
                data.append( 1j / (2 * self.ay) * cmath.exp(-1j * ky * self.m * self.ay))

        M = coo_matrix((data, (row, col)), shape=(self.m * self.n, self.m * self.n))

        return M

    def KxSq(self, kx=0):

        """
        Returns the sparse representation of the KxSq operator.

        Args:
            kx (float, optional): The value of kx. Defaults to 0.

        Returns:
            scipy.sparse.coo_matrix: Sparse matrix representation of the KxSq operator mnXmn.
        """

        row, col, data = [], [], []

        for j in range(self.m):

            for i in range(self.n):

                current_idx = j * self.n + i

                row.append(current_idx)
                col.append(current_idx)
                data.append(2 / (self.ax**2))

                if i + 1 < self.n:
                    row.append(current_idx)
                    col.append(j * self.n + i + 1)
                    data.append(-1 / (self.ax**2))

                if i - 1 >= 0:
                    row.append(current_idx)
                    col.append(j * self.n + i - 1)
                    data.append(-1 / (self.ax**2))

            ### To account for how boundary point connect in case of periodic operator ###
            if self.px:
                row.append(j * self.n + self.n - 1)
                col.append(j * self.n + 0)
                data.append(-1 / (self.ax**2) * np.exp(1j * kx * self.n * self.ax))

                row.append(j * self.n + 0)
                col.append(j * self.n + self.n - 1)
                data.append(-1 / (self.ax**2) * np.exp(-1j * kx * self.n * self.ax))

        M = coo_matrix((data, (row, col)), shape=(self.m * self.n, self.m * self.n))
        return M

    def KySq(self, ky=0):

        """
        Returns the sparse representation of the KySq operator.

        Args:
            ky (float, optional): The value of ky. Defaults to 0.

        Returns:
            scipy.sparse.coo_matrix: Sparse matrix representation of the KySq operator.
        """

        row, col, data = [], [], []

        for i in range(self.n):

            for j in range(self.m):

                current_idx = j * self.n + i

                row.append(current_idx)
                col.append(current_idx)
                data.append(2 / (self.ay**2))

                if j + 1 < self.m:
                    row.append(current_idx)
                    col.append((j + 1) * self.n + i)
                    data.append(-1 / (self.ay**2))

                if j - 1 >= 0:
                    row.append(current_idx)
                    col.append((j - 1) * self.n + i)
                    data.append(-1 / (self.ay**2))

            ### To account for how boundary point connect in case of periodic operator ###
            if self.py:
                row.append((self.m - 1) * self.n + i)
                col.append(0 * self.n + i)
                data.append(-1 / (self.ay**2) * np.exp(1j * ky * self.m * self.ay))

                row.append(0 * self.n + i)
                col.append((self.m - 1) * self.n + i)
                data.append(-1 / (self.ay**2) * np.exp(-1j * ky * self.m * self.ay))

        M = coo_matrix((data, (row, col)), shape=(self.m * self.n, self.m * self.n))

        return M

    def KxKy(self, kx=0, ky=0):

        """
        Returns the sparse representation of the KxKy operator.

        Args:
            kx (float, optional): The value of kx. Defaults to 0.
            ky (float, optional): The value of ky. Defaults to 0.

        Returns:
            scipy.sparse.coo_matrix: Sparse matrix representation of the KxKy operator.
        """

        row = []
        col = []
        data = []

        for j in range(self.m):

            for i in range(self.n):
                
                current_idx = j * self.n + i

                if j + 1 < self.m and i + 1 < self.n:
                    row.append( current_idx)
                    col.append( (j + 1) * self.n + i + 1)
                    data.append( -1 / (4 * self.ax * self.ay))

                if j + 1 < self.m and i - 1 >= 0:
                    row.append(current_idx)
                    col.append( (j + 1) * self.n + i - 1)
                    data.append( 1 / (4 * self.ax * self.ay))

                if j - 1 >= 0 and i + 1 < self.n:
                    row.append( current_idx)
                    col.append( (j - 1) * self.n + i + 1)
                    data.append( 1 / (4 * self.ax * self.ay))

                if j - 1 >= 0 and i - 1 >= 0:
                    row.append(current_idx)
                    col.append( (j - 1) * self.n + i - 1)
                    data.append( -1 / (4 * self.ax * self.ay))

        ### To account for how boundary point connect in case of periodic operator ###
        if self.py and self.px:

            ### Boundary points on lower and upper horizontal edges of 2D lattice ###
            ### Except points at the 4 corner of lattice ###

            for i in range(self.n):

                current_idx = (self.m - 1) * self.n + i

                if i != self.n - 1:
                    row.append( current_idx)
                    col.append( 0 * self.n + i + 1)
                    data.append( -1 / (4 * self.ax * self.ay) * cmath.exp(1j * ky * self.m * self.ay))

                if i != 0:
                    row.append( current_idx)
                    col.append( 0 * self.n + i - 1)
                    data.append( 1 / (4 * self.ax * self.ay) * cmath.exp(1j * ky * self.m * self.ay))

                if i != self.n - 1:
                    row.append( 0 * self.n + i)
                    col.append( (self.m - 1) * self.n + i + 1)
                    data.append( 1 / (4 * self.ax * self.ay) * cmath.exp(-1j * ky * self.m * self.ay))

                if i != 0:
                    row.append( 0 * self.n + i)
                    col.append( (self.m - 1) * self.n + i - 1)
                    data.append( -1 / (4 * self.ax * self.ay) * cmath.exp(-1j * ky * self.m * self.ay))

            ### Boundary points on left and right vertical edges of 2D lattice ###
            ### Except points at the 4 corner of lattice ###

            for j in range(self.m):

                current_idx = j * self.n + self.n - 1

                if j != self.m - 1:
                    row.append( current_idx)
                    col.append( (j + 1) * self.n + 0)
                    data.append( -1 / (4 * self.ax * self.ay) * cmath.exp(1j * kx * self.n * self.ax))

                if j != 0:
                    row.append(current_idx)
                    col.append( (j - 1) * self.n + 0)
                    data.append( 1 / (4 * self.ax * self.ay) * cmath.exp(1j * kx * self.n * self.ax))

                if j != self.m - 1:
                    row.append( j * self.n + 0)
                    col.append( (j + 1) * self.n + self.n - 1)
                    data.append( 1 / (4 * self.ax * self.ay) * cmath.exp(-1j * kx * self.n * self.ax))

                if j != 0:
                    row.append( j * self.n + 0)
                    col.append( (j - 1) * self.n + self.n - 1)
                    data.append( -1 / (4 * self.ax * self.ay) * cmath.exp(-1j * kx * self.n * self.ax))


            ### Points at the 4 corner of lattice ###

            row.append( (self.m - 1) * self.n + self.n - 1)
            col.append(0 * self.n + 0)
            data.append( -1 / (4 * self.ax * self.ay) * cmath.exp(1j * kx * self.n * self.ax) * cmath.exp(1j * ky * self.m * self.ay))

            row.append((self.m - 1) * self.n + 0)
            col.append( 0 * self.n + self.n - 1)
            data.append( 1 / (4 * self.ax * self.ay) * cmath.exp(-1j * kx * self.n * self.ax) * cmath.exp(1j * ky * self.m * self.ay))

            row.append( 0 * self.n + self.n - 1)
            col.append( (self.m - 1) * self.n + 0)
            data.append( 1 / (4 * self.ax * self.ay) * cmath.exp(1j * kx * self.n * self.ax) * cmath.exp(-1j * ky * self.m * self.ay))

            row.append( 0 * self.n + 0)
            col.append((self.m - 1) * self.n + self.n - 1)
            data.append( -1 / (4 * self.ax * self.ay) * cmath.exp(-1j * kx * self.n * self.ax) * cmath.exp(-1j * ky * self.m * self.ay))

        M = coo_matrix((data, (row, col)), shape=(self.m * self.n, self.m * self.n))
        
        return M

    def KxCu(self, kx=0):
        """
        Returns the sparse representation of the Kx cubed operator.

        Args:
            kx (float, optional): The value of kx. Defaults to 0.

        Returns:
            scipy.sparse.coo_matrix: Sparse matrix representation of the Kx cube.

        """


        row = []
        col = []
        data = []

        for j in range(self.m):
            for i in range(self.n):
                current_idx = j * self.n + i

                if i + 1 < self.n:
                    row.append(current_idx)
                    col.append(j * self.n + i + 1)
                    data.append(-1j / (self.ax ** 3))

                if i - 1 >= 0:
                    row.append(current_idx)
                    col.append(j * self.n + i - 1)
                    data.append(1j / (self.ax ** 3))

                if i + 2 < self.n:
                    row.append(current_idx)
                    col.append(j * self.n + i + 2)
                    data.append(1j / 2 / (self.ax ** 3))

                if i - 2 >= 0:
                    row.append(current_idx)
                    col.append(j * self.n + i - 2)
                    data.append(-1j / 2 / (self.ax ** 3))

            ### To account for how boundary point connect in case of periodic operator ###

            if self.px:
                row.append(j * self.n + self.n - 1)
                col.append(j * self.n + 0)
                data.append(-1j / (self.ax ** 3) * cmath.exp(1j * kx * self.n * self.ax))

                row.append(j * self.n + 0)
                col.append(j * self.n + self.n - 1)
                data.append(1j / (self.ax ** 3) * cmath.exp(-1j * kx * self.n * self.ax))

                row.append(j * self.n + self.n - 2)
                col.append(j * self.n + 0)
                data.append(1j / 2 / (self.ax ** 3) * cmath.exp(1j * kx * (self.n) * self.ax))

                row.append(j * self.n + 0)
                col.append(j * self.n + self.n - 2)
                data.append(-1j / 2 / (self.ax ** 3) * cmath.exp(-1j * kx * (self.n) * self.ax))

                row.append(j * self.n + self.n - 1)
                col.append(j * self.n + 1)
                data.append(1j / 2 / (self.ax ** 3) * cmath.exp(1j * kx * (self.n) * self.ax))
                
                row.append(j * self.n + 1)
                col.append(j * self.n + self.n - 1)
                data.append(-1j / 2 / (self.ax ** 3) * cmath.exp(-1j * kx * (self.n) * self.ax))

        M = csr_matrix((data, (row, col)), shape=(self.m * self.n, self.m * self.n))

        return M

    def KyCu(self, ky=0):
        '''

        Returns sparse representation of mnXmn matrix for KY cubed operator

        Args:
            ky (float, optional): The value of ky. Defaults to 0.

        Returns:
            scipy.sparse.coo_matrix: Sparse matrix representation of the Ky cube.

        '''

        row = []
        col = []
        data = []

        for i in range(self.n):
            for j in range(self.m):
                current_idx = j * self.n + i

                if j + 1 < self.m:
                    row.append(current_idx)
                    col.append((j + 1) * self.n + i)
                    data.append(-1j / (self.ay ** 3))

                if j - 1 >= 0:
                    row.append(current_idx)
                    col.append((j - 1) * self.n + i)
                    data.append(1j / (self.ay ** 3))

                if j + 2 < self.m:
                    row.append(current_idx)
                    col.append((j + 2) * self.n + i)
                    data.append(1j / 2 / (self.ay ** 3))

                if j - 2 >= 0:
                    row.append(current_idx)
                    col.append((j - 2) * self.n + i)
                    data.append(-1j / 2 / (self.ay ** 3))

            ### To account for how boundary point connect in case of periodic operator ###

            if self.py:
                row.append((self.m - 1) * self.n + i)
                col.append(0 * self.n + i)
                data.append(-1j / (self.ay ** 3) * cmath.exp(1j * ky * self.m * self.ay))

                row.append(0 * self.n + i)
                col.append((self.m - 1) * self.n + i)
                data.append(1j / (self.ay ** 3) * cmath.exp(-1j * ky * self.m * self.ay))

                row.append((self.m - 2) * self.n + i)
                col.append(0 * self.n + i)
                data.append(1j / 2 / (self.ay ** 3) * cmath.exp(1j * ky * self.m * self.ay))

                row.append(0 * self.n + i)
                col.append((self.m - 2) * self.n + i)
                data.append(1j / 2 / (self.ay ** 3) * cmath.exp(-1j * ky * self.m * self.ay))

                row.append((self.m - 1) * self.n + i)
                col.append(1 * self.n + i)
                data.append(1j / 2 / (self.ay ** 3) * cmath.exp(1j * ky * self.m * self.ay))

                row.append(1 * self.n + i)
                col.append((self.m - 1) * self.n + i)
                data.append(1j / 2 / (self.ay ** 3) * cmath.exp(-1j * ky * self.m * self.ay))

        M = csr_matrix((data, (row, col)), shape=(self.m * self.n, self.m * self.n))

        return M

    def KxSqKy(self, kx=0, ky=0):
        '''

        Returns sparse representation of mnXmn matrix for KX squared KY operator

        Args:
            kx (float, optional): The value of kx. Defaults to 0.
            ky (float, optional): The value of ky. Defaults to 0.

        Returns:
            scipy.sparse.coo_matrix: Sparse matrix representation of the Kx square Ky

        '''

        row = []
        col = []
        data = []

        for j in range(self.m):

            for i in range(self.n):

                if ( j+1) < self.m and i+1 < self.n:

                    row.append( j*self.n + i)
                    col.append( (j+1)*self.n + i+1)
                    data.append( 1j/2/(self.ax*self.ax*self.ay))

                if j+1 < self.m and i-1 >= 0:

                    row.append( j*self.n + i)
                    col.append((j+1)*self.n + i-1)
                    data.append( 1j/2/(self.ax*self.ax*self.ay))

                if j-1 >=0 and i+1 < self.n:

                    row.append( j*self.n + i)
                    col.append( (j-1)*self.n + i+1)
                    data.append( -1j/2/(self.ax*self.ax*self.ay))

                if j-1 >=0 and i-1 >=0:

                    row.append( j*self.n + i)
                    col.append( (j-1)*self.n + i-1)
                    data.append( -1j/2/(self.ax*self.ax*self.ay))

                if j+1 < self.m :

                    row.append( j*self.n + i)
                    col.append( (j+1)*self.n + i)
                    data.append( -2j/2/(self.ax*self.ax*self.ay))

                if j-1 >= 0 :

                    row.append( j*self.n + i)
                    col.append( (j-1)*self.n + i)
                    data.append( +2j/2/(self.ax*self.ax*self.ay))
        
        ### To account for how boundary point connect in case of periodic operator ###
        if self.py and self.px:

            ## upper and lower horizontal(x direction) edges of the lattice without corners ### 
            for i in range(self.n):

                if i != self.n-1 :
                    
                    row.append( (self.m-1)*self.n + i)
                    col.append( 0*self.n + i+1)
                    data.append( 1j/2/(self.ax*self.ax*self.ay)*cmath.exp(1j*ky*self.m*self.ay))

                if i!= 0 :

                    row.append( (self.m-1)*self.n + i)
                    col.append( 0*self.n + i-1)
                    data.append( 1j/2/(self.ax*self.ax*self.ay)*cmath.exp(1j*ky*self.m*self.ay))

                if i != self.n-1:

                    row.append( 0*self.n + i)
                    col.append( (self.m-1)*self.n + i+1)
                    data.append( -1j/2/(self.ax*self.ax*self.ay)*cmath.exp(-1j*ky*self.m*self.ay))

                if i != 0 :

                    row.append( 0*self.n + i)
                    col.append( (self.m-1)*self.n + i-1)
                    data.append( -1j/2/(self.ax*self.ax*self.ay)*cmath.exp(-1j*ky*self.m*self.ay))

                row.append( (self.m-1)*self.n + i)
                col.append( 0*self.n + i)
                data.append( -2j/2/(self.ax*self.ax*self.ay)*cmath.exp(1j*ky*self.m*self.ay))

                row.append( 0*self.n + i)
                col.append( (self.m-1)*self.n + i)
                data.append( 2j/2/(self.ax*self.ax*self.ay)*cmath.exp(-1j*ky*self.m*self.ay))            

            ## left and right vertical(y direction) Edges of the lattice without corner ###
            for j in range(self.m):

                if j != self.m-1 :

                    row.append( (j)*self.n + self.n-1)
                    col.append( (j+1)*self.n + 0)
                    data.append( 1j/2/(self.ax*self.ax*self.ay)*cmath.exp(1j*kx*self.n*self.ax)) 

                if j != 0 :
                    
                    row.append( (j)*self.n + self.n-1)
                    col.append( (j-1)*self.n + 0)
                    data.append( -1j/2/(self.ax*self.ax*self.ay)*cmath.exp(1j*kx*self.n*self.ax))

                if j !=self.m-1 :

                    row.append( (j)*self.n + 0)
                    col.append( (j+1)*self.n + self.n-1 )
                    data.append( 1j/2/(self.ax*self.ax*self.ay)*cmath.exp(-1j*kx*self.n*self.ax))

                if j != 0 :

                    row.append( j*self.n + 0)
                    col.append( (j-1)*self.n + self.n-1)
                    data.append( -1j/2/(self.ax*self.ax*self.ay)*cmath.exp(-1j*kx*self.n*self.ax))

            ## four corners of lattice ###  

            row.append( (self.m-1)*self.n + self.n-1)
            col.append( (0*self.n + 0))
            data.append( 1j/2/(self.ax*self.ax*self.ay)*cmath.exp(1j*kx*self.n*self.ax)*cmath.exp(1j*ky*self.m*self.ay))

            row.append( (self.m-1)*self.n + 0)
            col.append( (0*self.n + self.n-1))
            data.append( 1j/2/(self.ax*self.ax*self.ay)*cmath.exp(-1j*kx*self.n*self.ax)*cmath.exp(1j*ky*self.m*self.ay))

            row.append((0)*self.n + self.n-1)
            col.append( ((self.m-1)*self.n + 0))
            data.append(-1j/2/(self.ax*self.ax*self.ay)*cmath.exp(1j*kx*self.n*self.ax)*cmath.exp(-1j*ky*self.m*self.ay))

            row.append( 0*self.n + 0)
            col.append( (self.m-1)*self.n + self.n-1)
            data.append( -1j/2/(self.ax*self.ax*self.ay)*cmath.exp(-1j*kx*self.n*self.ax)*cmath.exp(-1j*ky*self.m*self.ay))            

        M = coo_matrix((data,(row,col)),shape=(self.m*self.n, self.m*self.n))

        return M

    def KxKySq(self, kx=0, ky=0):
        '''
        Returns sparse representation of mnXmn matrix for KX KY squared operator

        Args:
            kx (float, optional): The value of kx. Defaults to 0.
            ky (float, optional): The value of ky. Defaults to 0.

        Returns:
            scipy.sparse.coo_matrix: Sparse matrix representation of the Kx ky square

        '''


        row = []
        col = []
        data = []

        for j in range(self.m):

            for i in range(self.n):

                if ( j+1) < self.m and i+1 < self.n:

                    row.append( j*self.n + i)
                    col.append( (j+1)*self.n + i+1)
                    data.append( 1j/2/(self.ax*self.ay*self.ay))

                if j+1 < self.m and i-1 >= 0:

                    row.append( j*self.n + i)
                    col.append( (j+1)*self.n + i-1)
                    data.append( -1j/2/(self.ax*self.ay*self.ay))

                if j-1 >=0 and i+1 < self.n:

                    row.append( j*self.n + i)
                    col.append( (j-1)*self.n + i+1)
                    data.append( 1j/2/(self.ax*self.ay*self.ay))

                if j-1 >=0 and i-1 >=0:

                    row.append( j*self.n + i)
                    col.append( (j-1)*self.n + i-1)
                    data.append( -1j/2/(self.ax*self.ay*self.ay))

                if i+1 < self.n :

                    row.append(j*self.n + i)
                    col.append( (j)*self.n + i+1)
                    data.append( -2j/2/(self.ax*self.ay*self.ay))

                if i-1 >= 0 :

                    row.append( j*self.n + i)
                    col.append( (j)*self.n + i-1)
                    data.append( 2j/2/(self.ax*self.ay*self.ay))
        
        ### To account for how boundary point connect in case of periodic operator ###

        if self.py and self.px :

            ## upper and lower horizontal(x direction) Edges of the lattice without corner ### 
            for i in range(self.n):

                if i != self.n-1 :
                    
                    row.append( (self.m-1)*self.n + i)
                    col.append( 0*self.n + i+1)
                    data.append( 1j/2/(self.ax*self.ay*self.ay)*cmath.exp(1j*ky*self.m*self.ay))

                if i!= 0 :

                    row.append( (self.m-1)*self.n + i)
                    col.append( 0*self.n + i-1)
                    data.append( -1j/2/(self.ax*self.ay*self.ay)*cmath.exp(1j*ky*self.m*self.ay))

                if i != self.n-1:

                    row.append( 0*self.n + i)
                    col.append( (self.m-1)*self.n + i+1)
                    data.append( 1j/2/(self.ax*self.ay*self.ay)*cmath.exp(-1j*ky*self.m*self.ay))

                if i != 0 :

                    row.append( 0*self.n + i)
                    col.append( (self.m-1)*self.n + i-1)
                    data.append( -1j/2/(self.ax*self.ay*self.ay)*cmath.exp(-1j*ky*self.m*self.ay))
            

            ## left and right(y direction)  Edges of the lattice without corner ###
            for j in range(self.m):

                if j != self.m-1 :

                    row.append( (j)*self.n + self.n-1)
                    col.append( (j+1)*self.n + 0)
                    data.append( 1j/2/(self.ax*self.ay*self.ay)*cmath.exp(1j*kx*self.n*self.ax)) 

                if j != 0 :
                    
                    row.append( (j)*self.n + self.n-1)
                    col.append( (j-1)*self.n + 0)
                    data.append( 1j/2/(self.ax*self.ay*self.ay)*cmath.exp(1j*kx*self.n*self.ax))

                if j !=self.m-1 :

                    row.append( (j)*self.n + 0)
                    col.append( (j+1)*self.n + self.n-1 )
                    data.append( -1j/2/(self.ax*self.ay*self.ay)*cmath.exp(-1j*kx*self.n*self.ax))

                if j != 0 :

                    row.append( j*self.n + 0)
                    col.append( (j-1)*self.n + self.n-1)
                    data.append( -1j/2/(self.ax*self.ay*self.ay)*cmath.exp(-1j*kx*self.n*self.ax))

                row.append( j*self.n + 0)
                col.append( j*self.n + self.n-1)
                data.append( 2j/2/(self.ax*self.ay*self.ay)*cmath.exp(-1j*kx*self.m*self.ay))

                row.append( j*self.n + self.n-1)
                col.append( j*self.n + 0)
                data.append( -2j/2/(self.ax*self.ay*self.ay)*cmath.exp(1j*kx*self.m*self.ay))  

            ## four corners of lattice ###  

            row.append( (self.m-1)*self.n + self.n-1)
            col.append( (0*self.n + 0))
            data.append( 1j/2/(self.ax*self.ay*self.ay)*cmath.exp(1j*kx*self.n*self.ax)*cmath.exp(1j*ky*self.m*self.ay))

            row.append( (self.m-1)*self.n + 0)
            col.append( (0*self.n + self.n-1))
            data.append( -1j/2/(self.ax*self.ay*self.ay)*cmath.exp(-1j*kx*self.n*self.ax)*cmath.exp(1j*ky*self.m*self.ay))

            row.append( (0)*self.n + self.n-1)
            col.append( ((self.m-1)*self.n + 0))
            data.append(1j/2/(self.ax*self.ay*self.ay)*cmath.exp(1j*kx*self.n*self.ax)*cmath.exp(-1j*ky*self.m*self.ay))

            row.append( 0*self.n + 0)
            col.append( (self.m-1)*self.n + self.n-1)
            data.append( -1j/2/(self.ax*self.ay*self.ay)*cmath.exp(-1j*kx*self.n*self.ax)*cmath.exp(-1j*ky*self.m*self.ay))            

        M = coo_matrix((data,(row,col)),shape=(self.m*self.n, self.m*self.n))

        return M

    def Kx_quar(self, kx=0, ky=0):

        """
        Returns the sparse representation of the Kx^4 operator.

        Args:
            kx (float, optional): The value of kx. Defaults to 0.

        Returns:
            scipy.sparse.coo_matrix: Sparse matrix representation of the KxSq operator mnXmn.
        """

        row, col, data = [], [], []

        for j in range(self.m):

            for i in range(self.n):

                current_idx = j * self.n + i

                row.append(current_idx)
                col.append(current_idx)
                data.append(6 / (self.ax**4))

                if i + 1 < self.n:
                    row.append(current_idx)
                    col.append(j * self.n + i + 1)
                    data.append(-4 / (self.ax**4))

                if i - 1 >= 0:
                    row.append(current_idx)
                    col.append(j * self.n + i - 1)
                    data.append(-4 / (self.ax**4))

                if i + 2 < self.n:
                    row.append(current_idx)
                    col.append(j * self.n + i + 2)
                    data.append(1 / (self.ax**4))

                if i - 2 >= 0:
                    row.append(current_idx)
                    col.append(j * self.n + i - 2)
                    data.append(1 / (self.ax**4))

            ### To account for how boundary point connect in case of periodic operator ###
            if self.px:
                row.append(j * self.n + self.n - 1)
                col.append(j * self.n + 0)
                data.append(-4 / (self.ax**4) * np.exp(1j * kx * self.n * self.ax))

                row.append(j * self.n + 0)
                col.append(j * self.n + self.n - 1)
                data.append(-4 / (self.ax**4) * np.exp(-1j * kx * self.n * self.ax))

                row.append(j * self.n + self.n - 1)
                col.append(j * self.n + 1)
                data.append(1 / (self.ax**4) * np.exp(1j * kx * self.n * self.ax))

                row.append(j * self.n + 1)
                col.append(j * self.n + self.n - 1)
                data.append(1 / (self.ax**4) * np.exp(-1j * kx * self.n * self.ax))

                row.append(j * self.n + self.n - 2)
                col.append(j * self.n + 0)
                data.append(1 / (self.ax**4) * np.exp(1j * kx * self.n * self.ax))

                row.append(j * self.n + 0)
                col.append(j * self.n + self.n - 2)
                data.append(1 / (self.ax**4) * np.exp(-1j * kx * self.n * self.ax))

                



        M = coo_matrix((data, (row, col)), shape=(self.m * self.n, self.m * self.n))
        
        return M

    def Ep_th(self, str ,kx, ky):

        """
        Returns numerical dispersion relation of the operator specified by str

        Note: This func is meant to see if the op are coded correctly

        Args:
            kx (): The value of kx(could also be an array). 
            kx (): The value of kx(could also be an array). 

        Returns:
            
        """


        if str == 'kx':

            return 1/self.ax*np.sin(kx*self.ax)

        if str == 'ky':

            return 1/self.ay*np.sin(ky*self.ay)

        if str == 'kxsq':

            return 2/self.ax**2*(1-np.cos(kx*self.ax))

        if str == 'kysq':

            return 2/self.ay**2*(1-np.cos(ky*self.ay))

        if str == 'kxky':

            return 1/self.ax/self.ay*(np.sin(kx*self.ax)*np.sin(ky*self.ay))

        if str == 'kxcu':

            return 2/self.ax**3*(np.sin(kx*self.ax)) - 1/self.ax**3*np.sin(2*kx*self.ax)

        if str == 'kycu':

            return 2/self.ay**3*(np.sin(ky*self.ay)) - 1/self.ay**3*np.sin(2*ky*self.ay)

        if str == 'kxsqky':

            return 4/2/self.ax**2/self.ay*np.sin(ky*self.ay)*(1-np.cos(kx*self.ax))

        if str == 'kxkysq':

            return 4/2/self.ay**2/self.ax*np.sin(kx*self.ax)*(1-np.cos(ky*self.ay))

        if str == 'kxquar':

            return 1/self.ax**4*(6-8*np.cos(kx*self.ax)+2*np.cos(2*kx*self.ax))
        
        raise ValueError('Invalid input string') 

    def U(self, V):

        '''
        Returns the sparse representation of the potential on the grid matrix V(mnXmn matrix).

        Args:
            V is a mxn 2D array mapping onsite potential due to gates and such

        Returns:
            scipy.sparse.coo_matrix: Sparse matrix representation of the V.

        '''

        if V.shape[0] != 0:

            V = V.flatten()

            return diags(V)
        
        pass
    
    def H_eff_mass(self, m_eff, V=np.empty((0,0)), kx=0, ky=0):

        '''
        Returns effective mass Hamiltonian for given m,n lattice 
        for given effective mass
        '''
        
        if V.shape[0] !=0:
            return hbm/m_eff*(self.KxSq(kx)+self.KySq(ky)) + self.U(V)
        
        else:

            return hbm/m_eff*(self.KxSq(kx)+self.KySq(ky))    

    def H_cubic_so(self, alpha, kx=0, ky=0):

        '''
        Returns Hamiltonian for cubic Rashba spin orbit coupling for given m,n lattice 
        for given spin orbit coupling coeff alpha and for given Electric field E
        Note that H_cubic_so function only returns <+|H_cubic_Rashba|-> component , 
        The <-|H_cubic_Rashba|+> can be found by taking complex conjugate and transpose 
        alpha is in mev nm**(3)
        '''

        return alpha*(-1j)*(self.KxCu(kx)-3j*self.KxSqKy(kx,ky)-3*self.KxKySq(kx,ky)+1j*self.KyCu(ky))
    
    def H_linear_so(self, alpha, kx=0, ky=0):
        '''
        Returns Hamiltonian for linear Rashba spin orbit coupling for given m,n lattice 
        for given spin orbit coupling coeff alpha
        Note that H_cubic_so function only returns <+|H_cubic_Rashba|-> component , 
        The <-|H_linear_Rashba|+> can be found by taking complex conjugate 
        alpha is in mev nm
        '''
        return alpha*(self.Ky(ky) + 1j*self.Kx(kx))

    def Delta_matrix(self, Delta):

        return self.iden()*Delta

    def Basis_change_y(self, ny):
        
        '''
        Method returns the Basis change operator U dag  for y direction. 
        U dag operator changes basis from lattice to subband of quantum well when used as U dag (Matrix in xy lattice basis) U
        U dag (Matrix in xy lattice basis) U gives Matrix in x anad y subband basis

        Args:
            ny(int): No. of y subbands to be considered  

        returns:
            M( sparse matrix): Matrix of size (ny)nX(ny)n endocoded in sparse format

        '''

        row = []
        col = []
        data = []

        nf = math.sqrt(2/(self.m+1))         # Normalization factor

        for k in range(ny):

            for i in range(self.n):

                for j in range(self.m):

                    row.append(k*self.n + i)
                    col.append(j*self.n + i )
                    data.append(nf*math.sin((k+1)*math.pi*(j+1)/(self.m+1)))  # k+1, j+1 as subband no. and lattice numbering starts from 1 

        M = coo_matrix((data, (row, col)), shape=(ny * self.n, self.m * self.n))
        
        return M

    def ysq(self):

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