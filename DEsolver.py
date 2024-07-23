import numpy as np
import matplotlib
import scipy.linalg as spla
import matplotlib.animation as animation

class ODEivp:
    def integrator_ord1(self, x, y0, f, phi):
        """
        This function integrates for an order 1 ODE, so f has to be a 1D array. 
        The solver can be specified using phi.
        
        Input:
            x   : 1D array.
            y0  : either a number or an array initial values.
            f   : the ODE which you need to solve, s.t. f(x,y)=y'.
            phi : the method of solving (e.g. Euler method).

        Output:
            y   : the solution to f
        """
        ordn = 1

        y = np.zeros(np.size(x))
        
        y0_arr = np.array(y0)
        len_y0 = np.size(y0_arr)
        y[:len_y0] = y0_arr
        
        for i in range(len_y0-1,np.size(x)-1):
            h = x[i+1]-x[i]
            y[i+1] = y[i] + h * phi(x, y, f, i, ordn)
        
        return y

    def integrator_ordn(self, x, y0, f, phi):
        """
        This function is the integrator function for an 
        order N differential equation (N>2). 

        Input:
            x    : 1D array
            y0   : Matrix/multidimensional array of initial values. 
                   The multidimensional array of initial values does 
                   not have to be Nxm array, where m is the number 
                   of initial values for the nth parameter. 
            f    : The ODE which has to be solved, which has to be a 
                   1D array consisting of N elements: 
                   f = np.array([y1,y2,...,yN])
            phi  : The ODE solver method.

        Output:
            y    : The solution of the ODE, which is a Nxk array with N elements.
        """
        
        ordn = np.shape(f(x,y0))[0]
        
        y = np.zeros((ordn,np.size(x)))
        len_y0_lst = []
        
        y0_dim = np.shape(y0)[0]
        for i in range(y0_dim):
            len_y0 = np.size(y0[i,:])
            len_y0_lst.append(len_y0)
            y[i,:len_y0] = y0[i,:len_y0]

        len_y0_arr = np.array(len_y0_lst)

        for i in range(np.size(x)-1):
            for n in range(ordn):
                if i<(len_y0_arr[n]-1):
                    pass
                else:
                    h = x[i+1]-x[i]
                    
                    y[n,i+1] = y[n,i] + h * phi(x, y, f, i, ordn)[n]
        
        return y

    def integrator(self, x, y0, f, phi, ordn = 1):
        """
        This function combines the order 1 and order n integrators into
        one function, which decides wether the order 1 or order n integrator
        is called for a specific problem.

        Input:
            x    : 1D array
            y0   : Matrix/multidimensional, one dimensional or scalar 
                   array of initial values. The multidimensional array 
                   of initial values does not have to be Nxm array, 
                   where m is the number of initial values for the nth 
                   parameter. The 1 dimensional array is an array of 
                   initial values. The scalar is an initial value of 
                   the ODE
            f    : The ODE which has to be solved, which has to be a 
                   1D array consisting of N elements: 
                   f = np.array([y1,y2,...,yN]) for an order n 
                   problem. Or for one dimensional problem, the ODE which 
                   you need to solve, s.t. f(x,y)=y'
            phi  : The ODE solver method.

        Output:
            y    : The solution of the ODE, which is a Nxk array with N elements.
        """          
        
        phi_func = self.phi_choose(phi)

        if ordn == 1:                           
            y = self.integrator_ord1(x, y0, f, phi_func)   
        if ordn > 1:                            
            y = self.integrator_ordn(x, y0, f, phi_func) 
        return y  

    def phi_choose(self, phi):
        """
        phi_choose converts the method from a string to a function which
        can be implemented in the code.

        Input:
            phi : string, the method used to solve the ODE.

        Output:
            phi_functions[phi]  : function which is the method used to 
                                  solve the ODE. 
        """

        phi_functions = {
            'euler': self.phi_euler,
            'collatz': self.phi_collatz,
            'heun': self.phi_heun,
            'rk4': self.phi_rk4,
            'ab3': self.phi_ab3,
            'ab4': self.phi_ab4
        }
        
        if phi in phi_functions:
            return phi_functions[phi]
        else:
            raise ValueError("Unknown method called: {}".format(phi))

    def phi_euler(self, x, y, f, i, ordn):
        """
        Euler method of solving ODEs, where Phi is just the function itself at 
        point x_i and y_i.
        This function also chooses wether to use the order one case of 
        the Euler method or to use the order n Euler method. See 
        phi_euler_ordn for specifics on the order n function.
        
        Input:
            x   : 1D array.
            y   : 1D array, same size as x.
            f   : the ODE which you need to solve, s.t. f(x,y)=y'.
            i   : integer, specifies which iteration you are on.
            ordn: integer which is >0, specifies the order of 
                  the ODE.

        Output:
            phi : the integration part which needs to be added to the previous y_i
        """

        if ordn == 1:                           
            phi = f(x[i],y[i])  
        elif ordn > 1:                            
            phi = self.phi_euler_ordn(x, y, f, i)
        else:
            raise ValueError("Uncompattible order of ODE: {}. Order has to >=1".format(ordn))
        
        return phi

    def phi_collatz(self, x, y, f, i, ordn):
        """
        Collatz method of solving ODEs. 
        This function also chooses wether to use the order one case of 
        the Collatz method or to use the order n Collatz method. See 
        phi_collatz_ordn for specifics on the order n function.
        
        Input:
            x   : 1D array.
            y   : 1D array, same size as x.
            f   : the ODE which you need to solve, s.t. f(x,y)=y'.
            i   : integer, specifies which iteration you are on.
            ordn: integer which is >0, specifies the order of 
                  the ODE.

        Output:
            phi : the integration part which needs to be added to the previous y_i
        """
        if ordn == 1:
            h = x[i+1]-x[i]
            phi = f( x[i]+h/2, y[i]+h*f(x[i],y[i])/2 )
        
        elif ordn > 1:
            phi = self.phi_collatz_ordn(x, y, f, i)
        
        else:
            raise ValueError("Uncompattible order of ODE: {}. Order has to >=1".format(ordn))

        
        return phi
    
    def phi_heun(self, x, y, f, i, ordn):
        """
        Heun method of solving ODEs.
    
        The input is:
            x   : 1D array.
            y   : 1D array, same size as x.
            f   : the ODE which you need to solve, s.t. f(x,y)=y'.
            i   : integer, specifies which iteration you are on.
            ordn: integer which is >0, specifies the order of 
                  the ODE.

        The output is:
            phi : the integration part which needs to be added to the previous y_i
        """

        if ordn == 1:
            h = x[i+1]-x[i]
            phi = 0.5 * ( f(x[i],y[i]) + f(x[i+1],y[i]+h*f(x[i],y[i])) )
        
        elif ordn > 1:
            phi = self.phi_heun_ordn(x, y, f, i)
        
        else:
            raise ValueError("Uncompattible order of ODE: {}. Order has to >=1".format(ordn))

        return phi
    
    def phi_rk4(self, x, y, f, i, ordn):
        """
        RK method of 4th order for solving ODEs.
        
        Input:
            x   : 1D array.
            y   : 1D array, same size as x.
            f   : the ODE which you need to solve, s.t. f(x,y)=y'.
            i   : integer, specifies which iteration you are on.
            ordn: integer which is >0, specifies the order of 
                  the ODE.

        Output:
            phi : the integration part which needs to be added to the previous y_i
        """
        
        if ordn == 1:
            h = x[i+1]-x[i]
        
            k1 = f(x[i],y[i])
            k2 = f(x[i]+h/2,y[i]+h*k1/2)
            k3 = f(x[i]+h/2,y[i]+h*k2/2)
            k4 = f(x[i]+h,y[i]+h*k1)

            phi = (1/6) * ( k1 + 2*k2 + 2*k3 + k4 )

        elif ordn > 1:
            phi = self.phi_rk4_ordn(x, y, f, i)
        
        else:
            raise ValueError("Uncompattible order of ODE: {}. Order has to >=1".format(ordn))


        return phi

    def phi_ab3(self, x, y, f, i, ordn):
        """
        AB3 method of solving ODEs, but as long as iteration below 3 it lets the RK4 method solve it.
        
        Input:
            x   : 1D array.
            y   : 1D array, same size as x.
            f   : the ODE which you need to solve, s.t. f(x,y)=y'.
            i   : integer, specifies which iteration you are on.
            ordn: integer which is >0, specifies the order of 
                  the ODE.

        Output:
            phi : the integration part which needs to be added to the previous y_i
        """
        if ordn == 1:
            if i < 3:
                phi = self.phi_rk4(x, y, f, i, ordn)
            else:
                phi = (1/12) * ( 23*f(x[i],y[i]) - 16*f(x[i-1],y[i-1]) + 5*f(x[i-2],y[i-2]) )

        elif ordn > 1:
            phi = self.phi_ab3_ordn(x, y, f, i)
        
        else:
            raise ValueError("Uncompattible order of ODE: {}. Order has to >=1".format(ordn))
        
        return phi

    def phi_ab4(self, x, y, f, i, ordn):
        """
        AB4 method of solving ODEs, but as long as iteration below 4 it lets the RK4 method solve it.
        
        Input:
            x   : 1D array.
            y   : 1D array, same size as x.
            f   : the ODE which you need to solve, s.t. f(x,y)=y'.
            i   : integer, specifies which iteration you are on.
            ordn: integer which is >0, specifies the order of 
                  the ODE.

        Output:
            phi : the integration part which needs to be added to the previous y_i
        """

        if ordn == 1:
            if i < 4:
                phi = self.phi_rk4(x, y, f, i, ordn)
            else:
                phi = (1/24) * ( 55*f(x[i],y[i]) - 59*f(x[i-1],y[i-1]) + 37*f(x[i-2],y[i-2]) - 9*f(x[i-3],y[i-3]) )
        
        elif ordn > 1:
            phi = self.phi_ab4_ordn(x, y, f, i)
        
        else:
            raise ValueError("Uncompattible order of ODE: {}. Order has to >=1".format(ordn))
        
        return phi

    def phi_euler_ordn(self, x, y, f, i):
        """
        Euler method of solving ODEs of order N (N>2).
        
        Input:
            x   : 1D array.
            y   : Nxk array, where k is the size of x.
            f   : the ODE which you need to solve, which is an ND 
                  vector, s.t. f = np.array([y1,y2,...,yN]).
            i   : integer, specifies which iteration you are on.

        Output:
            phi : 1xN array, the integration part which needs to 
                be added to the previous y_i. 
        """
        
        phi = f(x[i],y[:,i])
        
        return phi
    
    def phi_collatz_ordn(self, x, y, f, i):
        """
        Collatz method of solving ODEs of order N (N>2).

        Input:
            x   : 1D array.
            y   : Nxk array, where k is the size of x
            f   : the ODE which you need to solve, which is an ND
                  vector, s.t. f = np.array([y1,y2,...,yN])
            i   : integer, specifies which iteration you are on.
        
        Output: 
            phi : 1xN array, the integration part which needs to 
                  be added to the previous y_i. 
        """

        h = x[i+1]-x[i]
        phi = f( x[i]+h/2, y[:,i]+h*f(x[i],y[:,i])/2 )

        return phi

    def phi_heun_ordn(self, x, y, f, i):
        """
        Heun method of solving ODEs of order N (N>2).

        Input:
            x   :1D array.
            y   : Nxk array, where k is the size of x
            f   : the ODE which you need to solve, which is an ND
                  vector, s.t. f = np.array([y1,y2,...,yN])
            i   : integer, specifies which iteration you are on.
            ordn: integer which is >0, specifies the dimension of 
                  the ODE.
        
        Output:
            phi : 1xN array, the integration part which needs to 
                  be added to the previous y_i. 
        """

        h = x[i+1]-x[i]
        phi = 0.5 * ( f(x[i],y[:,i]) + f(x[i+1],y[:,i]+h*f(x[i],y[:,i])) )

        return phi

    def phi_rk4_ordn(self, x, y, f, i):
        """
        RK4 method of solving ODEs of order N (N>2).
        
        Input:
            x   : 1D array.
            y   : Nxk array, where k is the size of x.
            f   : the ODE which you need to solve, which is a ND 
                vector, s.t. f = np.array([y1,y2,...,yN]).
            i   : specifies which iteration you are on.

        Input:
            phi : 1xN array, the integration part which needs to 
                be added to the previous y_i. 
        """
        
        h = x[i+1]-x[i]
        
        k1 = f(x[i],y[:,i])
        k2 = f(x[i]+h/2,y[:,i]+h*k1/2)
        k3 = f(x[i]+h/2,y[:,i]+h*k2/2)
        k4 = f(x[i]+h,y[:,i]+h*k1)

        phi = (1/6) * ( k1 + 2*k2 + 2*k3 + k4 )
        return phi

    def phi_ab3_ordn(self, x, y, f, i):
        """
        AB3 method of solving ODEs of order N (N>2).
        
        Input:
            x   : 1D array.
            y   : Nxk array, where k is the size of x.
            f   : the ODE which you need to solve, which is a ND 
                vector, s.t. f = np.array([y1,y2,...,yN]).
            i   : specifies which iteration you are on.

        Output:
            phi : 1xN array, the integration part which needs to 
                be added to the previous y_i. 
                
        If the integration step is less then 3 the RK4 method 
        needs to be used, because the (i-2)th element has to exist 
        and >0. So as long as i<3, this method cannot be used and
        the RK4 method is called. 
        """
        
        if i < 3:
            phi = self.phi_rk4_ordn(x, y, f, i)
        else:
            phi = (1/12) * ( 23*f(x[i],y[:,i]) - 16*f(x[i-1],y[:,i-1]) + 5*f(x[i-2],y[:,i-2]) )
        
        return phi

    def phi_ab4_ordn(self, x, y, f, i):
        """
        AB4 method of solving ODEs of order N (N>2).
        
        Input:
            x   : 1D array.
            y   : Nxk array, where k is the size of x.
            f   : the ODE which you need to solve, which is a ND 
                vector, s.t. f = np.array([y1,y2,...,yN]).
            i   : specifies which iteration you are on.

        Output:
            phi : 1xN array, the integration part which needs to 
                be added to the previous y_i. 

        If the integration step is less then 4 the RK4 method 
        needs to be used, because the (i-3)th element has to exist 
        and >0. So as long as i<4, this method cannot be used and
        the RK4 method is called. 
        """
        
        if i < 4:
            phi = self.phi_rk4_ordn(x, y, f, i)
        else:
            phi = (1/24) * ( 55*f(x[i],y[:,i]) - 59*f(x[i-1],y[:,i-1]) + 37*f(x[i-2],y[:,i-2]) - 9*f(x[i-3],y[:,i-3]) )
        
        return phi
    
class SolvePDE:
    def waveEq(self, a, x, t, f, g):
        """
        This function solves the wave equation of the form 
            u_tt = a^2 u_xx
        where a is some scalar value. 

        Input:
            a : scalar, coefficient in wave equation
            x : array, the domain on which you want to define the PDE
            t : array, the time domain on which you want to solve the PDE
            f : function taking in arrays, boundary condition of w(x,0)
            g : function taking in arrays, boundary condition of w_t(x,0)

        Output:
            w : nxm array, which is the approximate solution of the wave equation.
        """
        
        l = x[-1]
        lt = t[-1]
        m = np.size(x)
        h = l / (m-1)
        n = np.size(t)
        k = lt / (n-1)
        lamb = a * k / h

        w = np.zeros((n, m))

        lamb_arr = np.ones(m-3)*lamb**2
        lamb_diag = np.ones(m-2)*2*(1-lamb**2)
        diag = np.diag(lamb_diag)
        off_diag_u = np.diag(lamb_arr,1)
        off_diag_b = np.diag(lamb_arr,-1)

        A = diag + off_diag_u + off_diag_b

        if isinstance(f, function):
            if isinstance(g, function):
                pass
            else:
                raise TypeError("'g' should be of type 'function'")
        else:
            raise TypeError("'f' should be of type 'function'")
            

        for j in range(n):
            if j == 0:
                w[j, 1:-1] = f(x[1:-1])
            elif j == 1:
                for i in range(1, m - 1):
                    w[j, i] = (1 - lamb**2) * f(x[i]) + lamb**2 * (f(x[i + 1]) + f(x[i - 1])) / 2 + k * g(x[i])
            else:
                w[j, 1:-1] = np.dot(A,w[j-1, 1:-1]) - w[j-2, 1:-1]

        return w
    
    def statSEQ_1d(self, x, V):
        """
        This function solves the static or time independent Schrodinger 
        equation, for some potential V on domain x.

        Input:
            x : one dimensional array, space domain of wave equation
            V : function, the potential in which the SEQ is solved
        
        Output:
            E         : one dimensional array, ordered array of energies
            psi_order : nxm array, ordered array of wave function solutions
        """
        Nx = np.size(x)
        dx = (x[-1] - x[0])/Nx

        H = (1/dx**2) * ( np.diag(2*np.ones(Nx)) - np.diag(np.ones(Nx-1),1) - np.diag(np.ones(Nx-1),-1) ) + np.diag(V(x))

        E, psi = spla.eig(H)

        order = np.argsort(E)
        E = E[order]
        psi_order = psi[:, order]

        return E,psi_order

    def dynSEQ_1d(self, x, t, V, f):
        """
        This function solves the dynamic Schrodinger equation, for some 
        potential V on time domain t and space domain x, with boundary condition f.

        Input:
            x : array, the domain on which you want to define the PDE
            t : array, the time domain on which you want to solve the PDE
            V : function, potential in which the SEQ is solved
            f : function, boundary condition of w_t(x,0)

        Output:
            w : nxm array, which is the approximate solution of the SEQ equation.
        """

        Nx = np.size(x)
        Nt = np.size(t)

        k = (t[-1] - t[0])/Nt
        h = (x[-1] - x[0])/Nx

        lamb1 = 1j*k/h**2
        lamb2 = 1j*k/2

        w = np.zeros((Nx,Nt), dtype='complex_')

        w[:,0] = f(x)

        if isinstance(V, function):
            V_pot = V(x[1:-1])
        else:
            raise TypeError("'V' should be of type 'function'")

        A = np.diag( (1+lamb1)*np.ones(Nx-2) + lamb2 * V_pot ) - np.diag( lamb1 / 2 * np.ones(Nx-3), 1) - np.diag( lamb1 / 2 * np.ones(Nx-3), -1)
        B = np.diag( (1-lamb1)*np.ones(Nx-2) - lamb2 * V_pot ) + np.diag( lamb1 / 2 * np.ones(Nx-3), 1) + np.diag( lamb1 / 2 * np.ones(Nx-3), -1)
    
        for j in range(1, Nt):
            w[1:-1, j] = np.dot(spla.inv(A),B@w[1:-1,j-1])

        return w

    def DiffusionEq_const(self, D, x, y, t, g, h_1 = None, h_2 = None, h_3 = None, h_4 = None, f = None):
        """
        This function solves the diffusion equation for constant diffusion
        coefficient D, on spacial domain (x,y) and time domain t, with boundary 
        conditions g and h_i.

        Input:
            D   : scalar, diffusion coefficient in wave equation
            x   : array, x coordinate the domain on which you want to define the PDE
            y   : array, y coordinate the domain on which you want to define the PDE
            t   : array, the time domain on which you want to solve the PDE
            g   : function or 2D array, boundary condition of u; u(0,x,y)
            h_i : function or 2D array, boundary condtion of u; i=1,2 u(t,x)
            h_i : function or 2D array, boundary condtion of u; i=3,4 u(t,y)
            f   : function or 3D array, source function for PDE

        Output:
            u : 3D array (Nt, Nx, Ny), which is the approximate solution of the 2D 
                diffusion equation with constant coefficient.
        """
        Nx = np.size(x)
        Ny = np.size(y)

        shape_x = np.shape(x)
        shape_y = np.shape(y)

        if np.size(shape_x) == 1:
            if np.size(shape_y) == 1:
                pass
            else:
                raise TypeError("'y' should be a 1D array")
        else:
            raise TypeError("'x' should be a 1D array")

        if Nx != Ny:
            raise ValueError("x and y need to have the same dimensions.")
        
        if isinstance(D, float):
            pass
        elif isinstance(D, int):
            pass
        else:
            raise TypeError("Diffusion coefficient 'D' should be of type 'float'")

        Nt = np.size(t)
        dx = (x[-1] - x[0]) / Nx
        dy = (y[-1] - y[0]) / Ny
        dt = (t[-1] - t[0]) / Nt

        lamb1 = D * dt / (2 * dx**2)
        lamb2 = D * dt / (2 * dy**2)

        if f is None:
            F_source = np.zeros((Nt,Nx,Ny))
        elif isinstance(f,function):
            F_source = f(t,x,y)
        elif isinstance(f, np.ndarray):
            shape_f = np.shape(f)
            if shape_f[0] == Nt:
                if shape_f[1] == Nx:
                    if shape_f[2] == Ny:
                        pass
                    else:
                        raise IndexError("Index structure not correct: numpy.shape(g)[2] != numpy.size(y)")
                else:
                    raise IndexError("Index structure not correct: numpy.shape(g)[1] != numpy.size(x)")
            else: 
                raise IndexError("Index structure not correct: numpy.shape(g)[0] != np.size(t)")
        else: 
            raise TypeError("'f' should be of type 'function' or 'numpy.ndarray'")

        u = np.zeros((Nt,Nx,Ny))

        if isinstance(g,function):
            u[0] = g(x,y)
        elif isinstance(g, np.ndarray):
            shape_g = np.shape(g)
            if shape_g[0] == Nx:
                if shape_g[1] == Ny:
                    pass
                else:
                    raise IndexError("Index structure not correct: numpy.shape(g)[1] != numpy.size(y)")
            else: 
                raise IndexError("Index structure not correct: numpy.shape(g)[0] != np.size(x)")
        else: 
            raise TypeError("'g' should be of type 'function' or 'numpy.ndarray'")
        
        if h_1 is None:
            h_1 = np.zeros(Nx)
        elif isinstance(h_1,function):
            h_1 = h_1(t,x)
        elif isinstance(h_1,np.ndarray):
            shape_h1 = np.shape(h_1)
            if shape_h1[0] == Nt:
                if shape_h1[1] == Nx:
                    pass
                else:
                    raise IndexError("Index structure not correct: numpy.shape(h_1)[1] != numpy.size(x)")
            else: 
                raise IndexError("Index structure not correct: numpy.shape(h_1)[0] != np.size(t)")
        else: 
            raise TypeError("'h_1' should be of type 'function' or 'numpy.ndarray'")
        
        if h_2 is None:
            h_2 = np.zeros(Nx)
        elif isinstance(h_1,function):
            h_2 = h_2(t,x)
        elif isinstance(h_2,np.ndarray):
            shape_h2 = np.shape(h_2)
            if shape_h2[0] == Nt:
                if shape_h2[1] == Nx:
                    pass
                else:
                    raise IndexError("Index structure not correct: numpy.shape(h_2)[1] != numpy.size(x)")
            else: 
                raise IndexError("Index structure not correct: numpy.shape(h_2)[0] != np.size(t)")
        else: 
            raise TypeError("'h_2' should be of type 'function' or 'numpy.ndarray'")
        
        if h_3 is None:
            h_3 = np.zeros(Nx)
        elif isinstance(h_3,function):
            h_3 = h_3(t,x)
        elif isinstance(h_3,np.ndarray):
            shape_h3 = np.shape(h_3)
            if shape_h3[0] == Nt:
                if shape_h3[1] == Ny:
                    pass
                else:
                    raise IndexError("Index structure not correct: numpy.shape(h_3)[1] != numpy.size(y)")
            else: 
                raise IndexError("Index structure not correct: numpy.shape(h_3)[0] != np.size(t)")
        else: 
            raise TypeError("'h_3' should be of type 'function' or 'numpy.ndarray'")
        
        if h_4 is None:
            h_4 = np.zeros(Nx)
        elif isinstance(h_4,function):
            h_4 = h_4(t,x)
        elif isinstance(h_4,np.ndarray):
            shape_h4 = np.shape(h_4)
            if shape_h4[0] == Nt:
                if shape_h4[1] == Ny:
                    pass
                else:
                    raise IndexError("Index structure not correct: numpy.shape(h_4)[1] != numpy.size(y)")
            else: 
                raise IndexError("Index structure not correct: numpy.shape(h_4)[0] != np.size(t)")
        else: 
            raise TypeError("'h_4' should be of type 'function' or 'numpy.ndarray'")


        A_1 = np.diag((1+2*lamb1)*np.ones(Nx)) + np.diag(-lamb1*np.ones(Nx-1), -1) + np.diag(-lamb1*np.ones(Nx-1), 1)
        A_1[0,:] = np.zeros(Nx)
        A_1[-1,:] = np.zeros(Nx)

        B_1 = np.diag((1-2*lamb1)*np.ones(Nx)) + np.diag(-lamb1*np.ones(Nx-1), -1) + np.diag(-lamb1*np.ones(Nx-1), 1)
        B_1[:,0] = np.zeros(Nx)
        B_1[:,-1] = np.zeros(Nx)

        A_2 = np.diag((1+2*lamb2)*np.ones(Nx)) + np.diag(-lamb2*np.ones(Nx-1), -1) + np.diag(-lamb2*np.ones(Nx-1), 1)
        A_2[0,:] = np.zeros(Nx)
        A_2[-1,:] = np.zeros(Nx)

        B_2 = np.diag((1-2*lamb2)*np.ones(Nx)) + np.diag(-lamb2*np.ones(Nx-1), -1) + np.diag(-lamb2*np.ones(Nx-1), 1)
        B_2[:,0] = np.zeros(Nx)
        B_2[:,-1] = np.zeros(Nx)

        A_1_inv = spla.inv(A_1)
        B_2_inv = spla.inv(B_2)

        for i in range(1,Nt):
            u[i] = (A_2@(A_1_inv@(u[i-1]@B_1)))@B_2_inv + F_source[i] / 2 + F_source[i-1] / 2
            u[i,0,:] = h_1[i,:]
            u[i,-1,:] = h_2[i,:]
            u[i,:,0] = h_3[i,:]
            u[i,:,-1] = h_4[i,:]

        return u
    
    def DiffusionEq_tdep(self, D, x, y, t, g, h_1 = None, h_2 = None, h_3 = None, h_4 = None, f = None):
        """
        This function solves the diffusion equation for time dependent diffusion 
        coefficient D, on spacial domain (x,y) and time domain t, with boundary
        conditions g and h_i.

        Input:
            D   : 1D array, time dependent diffusion coefficient in diffusion equation
            x   : array, x coordinate the domain on which you want to define the PDE
            y   : array, y coordinate the domain on which you want to define the PDE
            t   : array, the time domain on which you want to solve the PDE
            g   : function or 2D array, boundary condition of u; u(0,x,y)
            h_i : function or 2D array, boundary condtion of u; i=1,2 u(t,x)
            h_i : function or 2D array, boundary condtion of u; i=3,4 u(t,y)
            f   : function or 3D array, source function for PDE

        Output:
            u : 3D array (Nt, Nx, Ny), which is the approximate solution of the 2D 
                diffusion equation with constant coefficient.
        """
        Nx = np.size(x)
        Ny = np.size(y)

        shape_x = np.shape(x)
        shape_y = np.shape(y)

        if np.size(shape_x) == 1:
            if np.size(shape_y) == 1:
                pass
            else:
                raise TypeError("'y' should be a 1D array")
        else:
            raise TypeError("'x' should be a 1D array")

        if Nx != Ny:
            raise ValueError("x and y need to have the same dimensions.")
        
        if isinstance(D, np.ndarray):
            D_size = np.size(np.shape(D))
            if D_size == 1:
                pass
            else:
                raise IndexError("Diffusion coefficient should be one dimensional")
        elif isinstance(D, function):
            D = D(t)
            D_size = np.size(np.shape(D))
            if D_size == 1:
                pass
            else:
                raise IndexError("Diffusion coefficient should be one dimensional")
        else:
            raise TypeError("Diffusion coefficient 'D' should be a 1D array or of type 'function'")

        Nt = np.size(t)
        dx = (x[-1] - x[0]) / Nx
        dy = (y[-1] - y[0]) / Ny
        dt = (t[-1] - t[0]) / Nt

        if f is None:
            F_source = np.zeros((Nt,Nx,Ny))
        elif isinstance(f,function):
            F_source = f(t,x,y)
        elif isinstance(f, np.ndarray):
            shape_f = np.shape(f)
            if shape_f[0] == Nt:
                if shape_f[1] == Nx:
                    if shape_f[2] == Ny:
                        pass
                    else:
                        raise IndexError("Index structure not correct: numpy.shape(g)[2] != numpy.size(y)")
                else:
                    raise IndexError("Index structure not correct: numpy.shape(g)[1] != numpy.size(x)")
            else: 
                raise IndexError("Index structure not correct: numpy.shape(g)[0] != np.size(t)")
        else: 
            raise TypeError("'f' should be of type 'function' or 'numpy.ndarray'")

        u = np.zeros((Nt,Nx,Ny))

        if isinstance(g,function):
            u[0] = g(x,y)
        elif isinstance(g, np.ndarray):
            shape_g = np.shape(g)
            if shape_g[0] == Nx:
                if shape_g[1] == Ny:
                    pass
                else:
                    raise IndexError("Index structure not correct: numpy.shape(g)[1] != numpy.size(y)")
            else: 
                raise IndexError("Index structure not correct: numpy.shape(g)[0] != np.size(x)")
        else: 
            raise TypeError("'g' should be of type 'function' or 'numpy.ndarray'")
        
        if h_1 is None:
            h_1 = np.zeros(Nx)
        elif isinstance(h_1,function):
            h_1 = h_1(t,x)
        elif isinstance(h_1,np.ndarray):
            shape_h1 = np.shape(h_1)
            if shape_h1[0] == Nt:
                if shape_h1[1] == Nx:
                    pass
                else:
                    raise IndexError("Index structure not correct: numpy.shape(h_1)[1] != numpy.size(x)")
            else: 
                raise IndexError("Index structure not correct: numpy.shape(h_1)[0] != np.size(t)")
        else: 
            raise TypeError("'h_1' should be of type 'function' or 'numpy.ndarray'")
        
        if h_2 is None:
            h_2 = np.zeros(Nx)
        elif isinstance(h_1,function):
            h_2 = h_2(t,x)
        elif isinstance(h_2,np.ndarray):
            shape_h2 = np.shape(h_2)
            if shape_h2[0] == Nt:
                if shape_h2[1] == Nx:
                    pass
                else:
                    raise IndexError("Index structure not correct: numpy.shape(h_2)[1] != numpy.size(x)")
            else: 
                raise IndexError("Index structure not correct: numpy.shape(h_2)[0] != np.size(t)")
        else: 
            raise TypeError("'h_2' should be of type 'function' or 'numpy.ndarray'")
        
        if h_3 is None:
            h_3 = np.zeros(Nx)
        elif isinstance(h_3,function):
            h_3 = h_3(t,x)
        elif isinstance(h_3,np.ndarray):
            shape_h3 = np.shape(h_3)
            if shape_h3[0] == Nt:
                if shape_h3[1] == Ny:
                    pass
                else:
                    raise IndexError("Index structure not correct: numpy.shape(h_3)[1] != numpy.size(y)")
            else: 
                raise IndexError("Index structure not correct: numpy.shape(h_3)[0] != np.size(t)")
        else: 
            raise TypeError("'h_3' should be of type 'function' or 'numpy.ndarray'")
        
        if h_4 is None:
            h_4 = np.zeros(Nx)
        elif isinstance(h_4,function):
            h_4 = h_4(t,x)
        elif isinstance(h_4,np.ndarray):
            shape_h4 = np.shape(h_4)
            if shape_h4[0] == Nt:
                if shape_h4[1] == Ny:
                    pass
                else:
                    raise IndexError("Index structure not correct: numpy.shape(h_4)[1] != numpy.size(y)")
            else: 
                raise IndexError("Index structure not correct: numpy.shape(h_4)[0] != np.size(t)")
        else: 
            raise TypeError("'h_4' should be of type 'function' or 'numpy.ndarray'")

        for i in range(1,Nt):
            lamb1 = dt / (2 * dx**2)
            lamb2 = dt / (2 * dy**2)
            D_n_1_2 = (D[i] + D[i-1]) / 2

            A_1 = np.diag((1+2*D_n_1_2*lamb1)*np.ones(Nx)) + np.diag(-D_n_1_2*lamb1*np.ones(Nx-1), -1) + np.diag(-D_n_1_2*lamb1*np.ones(Nx-1), 1)
            A_1[0,:] = np.zeros(Nx)
            A_1[-1,:] = np.zeros(Nx)

            B_1 = np.diag((1-2*D[i]*lamb1)*np.ones(Nx)) + np.diag(-D[i]*lamb1*np.ones(Nx-1), -1) + np.diag(-D[i]*lamb1*np.ones(Nx-1), 1)
            B_1[:,0] = np.zeros(Nx)
            B_1[:,-1] = np.zeros(Nx)

            A_2 = np.diag((1+2*D_n_1_2*lamb2)*np.ones(Nx)) + np.diag(-D_n_1_2*lamb2*np.ones(Nx-1), -1) + np.diag(-D_n_1_2*lamb2*np.ones(Nx-1), 1)
            A_2[0,:] = np.zeros(Nx)
            A_2[-1,:] = np.zeros(Nx)

            B_2 = np.diag((1-2*D[i]*lamb2)*np.ones(Nx)) + np.diag(-D[i]*lamb2*np.ones(Nx-1), -1) + np.diag(-D[i]*lamb2*np.ones(Nx-1), 1)
            B_2[:,0] = np.zeros(Nx)
            B_2[:,-1] = np.zeros(Nx)

            A_1_inv = spla.inv(A_1)
            B_2_inv = spla.inv(B_2)

            u[i] = (A_2@(A_1_inv@(u[i-1]@B_1)))@B_2_inv + F_source[i] / 2 + F_source[i-1] / 2
            u[i,0,:] = h_1[i,:]
            u[i,-1,:] = h_2[i,:]
            u[i,:,0] = h_3[i,:]
            u[i,:,-1] = h_4[i,:]

        return u

class SolveSDE:
    def solveSDE(x):
        x = 1
        return x
    
ode_solve = ODEivp()
pde_solve = SolvePDE()