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
        Input:
            a : scalar, coefficient in wave equation
            x : array, the domain on which you want to define the PDE
            t : array, the time domain on which you want to solve the PDE
            f : function taking in arrays, boundary condition of u(x,0)
            g : function taking in arrays, boundary condition of u_t(x,0)

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
        Nx = np.size(x)
        dx = (x[-1] - x[0])/Nx

        H = (1/dx**2) * ( np.diag(2*np.ones(Nx)) - np.diag(np.ones(Nx-1),1) - np.diag(np.ones(Nx-1),-1) ) + np.diag(V(x))

        E, psi = spla.eig(H)

        order = np.argsort(E)
        E = E[order]
        psi_order = psi[:, order]

        return E,psi_order

    def dynSEQ_1d(self, x, t, V, f):
        
        Nx = np.size(x)
        Nt = np.size(t)

        k = (t[-1] - t[0])/Nt
        h = (x[-1] - x[0])/Nx

        lamb1 = 1j*k/h**2
        lamb2 = 1j*k/2

        w = np.zeros((Nx,Nt), dtype='complex_')

        w[:,0] = f(x)

        V_pot = V(x[1:-1])

        A = np.diag( (1+lamb1)*np.ones(Nx-2) + lamb2 * V_pot ) - np.diag( lamb1 / 2 * np.ones(Nx-3), 1) - np.diag( lamb1 / 2 * np.ones(Nx-3), -1)
        B = np.diag( (1-lamb1)*np.ones(Nx-2) - lamb2 * V_pot ) + np.diag( lamb1 / 2 * np.ones(Nx-3), 1) + np.diag( lamb1 / 2 * np.ones(Nx-3), -1)
    
        for j in range(1, Nt):
            w[1:-1, j] = np.dot(spla.inv(A),B@w[1:-1,j-1])

        return w
    
    def HeatEq_2d(self, a, x, t):

        return 1

class SolveSDE:
    def solveSDE(x):
        x = 1
        return x
    
ode_ivp = ODEivp()
pde_solve = SolvePDE()