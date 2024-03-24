# DEsolver: Differential Equation Library

This library contains function to solve differential equations. The type of differential equations which can be solved are:
- Ordinary differential equations: `ODEivp`,
- Partial differential equations: `SolvePDE`,
- Stochastic differential equations `SolveSDE`.

There are constraints in the type of problems that can be solved by this differential equation solver librabry. These constraints will be addressed in their associated sections

## Ordinary Differential Equations
The kind of problems that can be solved with this solver are initial value problems (ivp's).
With an order one ODE is meant an equation of the form:

$$y' = f(x,y)$$

Where $y$ is the function for which the ODE has to be solved and $f(x,y)$ is a general function of $x$ and $y$, which is equal to $y'$. An example is the ODE with a simple exponential solution:

$$y' = y$$

With an order n ODE is meant an equation of the form:

$$\mathbf{y}' = \mathbf{f}(x,\mathbf{y}).$$

Here the bold $\mathbf{y}, \mathbf{f}$ are vector. This way one is able to solve higher order ODEs, e.g. a pendulum-like problem:

$$y'' = -\sin{\left(y(x)\right)} \quad \to \quad \begin{pmatrix} y_0'(x) \\ y_1'(x) \end{pmatrix} = \begin{pmatrix} y_1(x) \\ -\sin{\left[ y_0(x) \right]} \end{pmatrix}.$$

NOTE: order $n$ form is also a system of coupled ODEs. In the rest of the text and in the code is spoken of 'order $n$' ODEs, but this may be replaced by '$n$ coupled' ODEs

### Integrator functions
The function which is called to solve the ODE is `integrator(x, y0, f, phi, ordn)`, where `x` is the interval over which the ODE is solved, `y0` are the initial values, `f` is the right hand side of the ODE, `phi` is the solving method and `ordn` is the order of the method. When `ordn` is not specified, it assumes it is a first oder ODE. First `integrator` chooses the function `phi_[method]` corresponding to the method which it is given. Because the `integrator` function does not do the integration itself, it then decides wether the problem is order one or order $n>1$. For order one ODEs the function `integrator` calls `integrator_ord1` and for order $n>1$ it calls `integrator_ordn`. The general solving principle used in both order one and order $n$ integrator function is 

$$y_{i+1} = y_i + h \Phi(x_i,y_i; h; f),$$
$$x_{i+1} = x_i + h,$$

where $h$ is the step size. For the order $n$ the $y$'s are then replaced by vectors.

The order one integrator function initializes a solution array `y`, which is the same size as the input interval `x`. Then stores the initial values, given in `y0`, in the right elements of solution array `y`. It then starts a for-loop where, every loop, it first calculates the stepsize of the next step and then uses the general solving principle to obtain the next step of the solution `y`. 

The order $n$ integrator function works similar to the order one integrator function. To solve the ODE it first has to know the order `ordn`, which it obtains from the shape of intput function `f`. After obtaining the order of the ODE, the solution array `y` is created. 
To have more freedom, the dimensions of the array `y0` do not have to be equal, e.g. `y0 = np.array([[1,1.1],[2,2.3,2.5]])` for a 2D problem. Ideally, the dimensions of `y0` are equal, but in this code the dimensions do not have to be equal.
Then in a nested for-loop the integration is done using the general solving prinviple. The first for-loop runs over the steps in the interval (shape of `x`) and the second for-loop runs over the dimensions of the ODE. The if-statement checks if at step `i` in the $n$ th solution of the ODE is an initial value or not. If it is an initial value, nothing is done and if it is not an initial value, the integration steps are done using solver method `phi`.

The following solver methods may be used:
| Solving Method | Input string |
|----------------|--------------|
| Euler method   | 'euler'      |
| Collatz method | 'collatz'    |
| Heun method    | 'heun'       |
| Runge-Kutta 4th order | 'rk4' |
|Adam-Bashforth 3 step method | 'ab3' |
|Adam-Bashforth 4 step method | 'ab4' |

All functions without `_ordn` at the end of its name, also choose wether to use the order one or order n variant of `phi`.

Example input order one ODE:
```
f = lambda x,y: x + y
x = np.linspace(0, 4, 1000)
y0 = 0.1
y = integrator(x, y0, f, 'ab4')
```
One can also define a function for the ODE instead of using `lambda`, e.g.:
```
def f(x,y):

    return y - x**2 + 1.0
```
Example input order 2 ODE:
```
def f(x,y):

    return np.array([y[1],-np.sin(y[0])])

y0 = np.array([[45*np.pi/180],[0]])
x = np.linspace(0, 10*np.pi, 1000)
y = integrator_ndim(x, y0, f, 'rk4')
```

### Euler Method
The Euler method is called by specifying `euler` in the `integrator` function. The Euler method has 

$$\Phi(x_i,y_i; h; f) = f(x_i,y_i).$$ 

Thus it returns the value of the function at point $(x_i,y_i)$. In Python code it is written like `f(x[i],y[i])`. 
In the order $n$ case, 

$$\Phi(x_i,y_i; h; f) = \mathbf{f}(x_i,\mathbf{y}_i).$$ 

Or in code `f(x[i],y[:,i])`.

### Collatz method
The Collatz method is called by specifying `collatz` in the `integrator` function. The Collatz method has 

$$\Phi(x_i,y_i; h; f) = f(x_i + h/2,y_i + h*f(x_i,y_i/2)).$$ 

In Python code it is written like `f( x[i]+h/2, y[i]+h*f(x[i],y[i])/2 )`. 
Similarly, in the order $n$ case, 

$$\Phi(x_i,y_i; h; f) = \mathbf{f}(x_i + h/2,\mathbf{y}_i + h*\mathbf{f}(x_i,\mathbf{y}_i/2)).$$ 

Or in code `f( x[i]+h/2, y[;,i]+h*f(x[i],y[;,i])/2)`.

### Heun method
The Heun method is called by specifying `heun` in the `integrator` function. The Heun method has 

$$\Phi(x_i,y_i; h; f) = \frac{1}{2} \left( f(x_i,y_i) + f(x_{i+1}, y_i + hf(x_i,y_i)) \right).$$ 

In Python code it is written like `0.5 * ( f(x[i],y[i]) + f(x[i+1],y[i]+h*f(x[i],y[i])) )`. 
Similarly, in the order $n$ case, 

$$\Phi(x_i,y_i; h; f) = \frac{1}{2} \left( \mathbf{f}(x_i,\mathbf{y}_i) + \mathbf{f}(x_{i+1}, \mathbf{y}_i + h\mathbf{f}(x_i,\mathbf{y}_i)) \right).$$ 

or in code `f( 0.5 * ( f(x[i],y[;,i]) + f(x[i+1],y[;,i]+h*f(x[i],y[;,i])) )`.

### Runge-Kutta method (4th order)
The Runge-Kutta method of 4th order is called by specifying `rk4` in the `integrator` function. The RK method of 4th order has 

$$\Phi(x_i,y_i; h; f) = \frac{1}{6} * ( k_1 + 2k_2 + 2k_3 + k_4 ),$$

where

$$k_1 = f(x_i,y_i),$$ 
$$k_2 = f(x_i + h/2, y_i + hk_1),$$
$$k_3 = f(x_i + h/2, y_i + hk_2),$$
$$k_4 = f(x_i + h, y_i + hk_1),$$

In order $n$ ODEs, $y_i \to \mathbf{y}_i$ or in code `y[i]` $\to$ `y[:,i]`. 

### Adam-Bashforth 3 step method
The Adam-Bashforth 3 step method is called by specifying `ab3` in the `integrator` function. The AB3 method has 

$$\Phi(x_i,y_i; h; f) = \frac{1}{12} \left( 23 f(x_i,y_i) - 16f(x_{i-1},y_{i-1}) + 5f(x_{i-2},y_{i-2}) \right).$$

From this form it can be seen that $\Phi$ can only be calculated when $y_j$ is known for $j \in \{i, i-1, i-2 \}$. Thus when $i<3$ the RK4 method is called and the AB3 method is only started when $i \geq 3$. 
In the order $n$ ODEs, $y_i \to \mathbf{y}_i$ or in code `y[i]` $\to$ `y[:,i]`. 

### Adam-Bashforth 4 step method
The Adam-Bashforth 4 step method is called by specifying `ab4` in the `integrator` function. The AB4 method has 

$$\Phi(x_i,y_i; h; f) = \frac{1}{24} \left( 55f(x_i,y_i) - 59*f(x_{i-1},y_{i-1}) + 37f(x_{i-2},y_{i-2}) - 9f(x_{i-3},y_{i-3}) \right).$$

From this form it can be seen that $\Phi$ can only be calculated when $y_j$ is known for $j \in \{i, i-1, i-2, i-3 \}$. Thus when $i<4$ the RK4 method is called and the AB4 method is only started when $i \geq 4$. 
In the order $n$ ODEs, $y_i \to \mathbf{y}_i$ or in code `y[i]` $\to$ `y[:,i]`. 

## Partial Differential Equations
### Wave equation
The wave equation is a PDE of the form

$$\frac{\partial^2 u}{\partial t^2} = \alpha^2 \frac{\partial^2 u}{\partial x^2} \qquad 0 \leq x \leq l, \quad 0 \leq t$$

The boundary condition for solving this are assumed to be:

$$ u(0,t)=u(l,t)=0 \quad \text{for } t > 0,$$
$$u(0,x)=f(x)$$
$$\frac{\partial}{\partial t} u(x,0)=g(x) \quad \text{for } 0 \leq x \leq l.$$

The numerical solution is obtained iteratively in time. The time is given by $t = j\Delta t$, $x = i\Delta x$ (where $i \in \{1,\ldots, m-2\} $), $\vec{w}$ is an $m-2$ dimensional vector, for different $j$ we have$:

$$\vec{w}_{j=0} = f(\vec{x}),$$
$$\vec{w}_{j=1} = (1-\lambda^2) f(x_i) + \frac{\lambda^2}{2} f(x_{i+1}) + \frac{\lambda^2}{2} f(x_{i-1}) + (\Delta t) g(x_i),$$
$$\vec{w}_{j+1} = \mathbf{A} \vec{w}_j - \vec{w}_{j-1},$$

with 

$$\mathbf{A} = \begin{pmatrix} 2(1-\lambda^2) & \lambda^2 & & 0\\
        \lambda^2 & \ddots & \ddots &  \\
        & \ddots & \ddots &  \lambda^2 \\
        0 &  & \lambda^2 & 2(1-\lambda^2)   \end{pmatrix}$$

with $\lambda = \alpha (\Delta t)/(\Delta x) $.

### Static Schrödinger Equation (1D)
One of the most famous equations of quantum mechanics is the Schrödinger equation. By calling `statSEQ_1d` you call the static Schrödinger equation (SEQ):

$$\hat{H} \Psi(x) = - \frac{\hbar^2}{2m} \frac{\partial^2}{\partial x^2} \Psi(x) + V(x) \Psi(x) = E \Psi(x)$$

In the code I have set $\hbar = 1, m = 0.5$. The SEQ can be numerically transformed to

$$-\frac{1}{\Delta x^2} ( \Psi_{n+1} - 2\Psi_n + \Psi_{n-1} ) = E_n \Psi.$$

The right-hand side represents the Hamiltonian, which can be written in matrix form:

$${H} = \frac{1}{(\Delta x)^2}\left(\begin{array}{lccc} 2 + V(x_0) & \frac{-1}{h^2} & &  0 \\ \\
        -1         & \ddots         & \ddots         &               \\ \\
                               & \ddots         & \ddots         &  -1    \\ \\
        0                      &                & -1 & 2 + V(x_m)    \end{array} \right)$$

The solution to the static SEQ are the eigenvalues ($E_n$) and eigenvectors ($\Psi$) of $H$. This can be solved for some potential $V(x)$. The static SEQ solver is called by `statSEQ_1d(x, V)`, for some one dimensional grid `x` and some potential `V`. The grid should be consist of equidistant point. The potential should either be a lambda function or a general function, e.g. for a simple parabolic potential:

```
V = lambda x: x**2
```

or

```
def V(x):
    return x**2
```
The output are an ordered array of energies and an array of solutions $\Psi$, ordered by their energies. An example of calling the function is:

```
x = np.linspace(-2,2,100, endpoint=True)
V = lambda x: x**2
E, psi = statSEQ_1d(x, V)
```

### Dynamic Schrödinger Equation (1D)
The dynamical Schrodinger (SEQ) is of the form

$$i\hbar \frac{\partial}{\partial t} \Psi(x,t) = -\frac{\hbar^2}{2m} \frac{\partial^2}{\partial x^2} \Psi(x,t) + V(x)\Psi(x,t).$$

In the code I have set $\hbar = 1$ and $m = 0.5$. Taking this to a numerical setting using the Crank-Nicolson method gives:

$$\vec{w}_{j+1} = \mathbf{A}^{-1} \mathbf{B} \vec{w}_{j}$$
$$\mathbf{A} = \begin{pmatrix} 1 + \lambda_1 + \lambda_2V(x_1) & -\frac{\lambda_1}{2} & & & 0  \\ \\
 - \frac{\lambda_1}{2} & \ddots & \ddots & & \\ \\
  & \ddots & \ddots & & - \frac{\lambda_1}{2} \\
 0 & & & - \frac{\lambda_1}{2} & 1 + \lambda_1 + \lambda_2V(x_n) \end{pmatrix}$$
$$\mathbf{B} = \begin{pmatrix} 1 - \lambda_1 - \lambda_2V(x_1) & \frac{\lambda_1}{2} & & & 0  \\ \\
 \frac{\lambda_1}{2} & \ddots & \ddots & & \\ \\
  & \ddots & \ddots & & \frac{\lambda_1}{2} \\
 0 & & & \frac{\lambda_1}{2} & 1 - \lambda_1 - \lambda_2V(x_n) \end{pmatrix}$$

with $\lambda_1 = \frac{i\Delta t}{(\Delta x)^2}$ and $\lambda_2 = \frac{i\Delta t}{2}$. 
This matrix equation is solved iteratively for all $j \in \{1, \ldots, t_N/\Delta t\}$. The output of the solver is the wave function $\Psi(x,t)$.  

An example of calling the  dynamical SEQ solver is given below:
```
V = lambda x: 0.25 * x**2.0
f = lambda x: np.exp(-x**2.0) / 2.5

x = np.linspace(-5, 5, 100, endpoint=True)
t = np.linspace(0, 20, 400, endpoint=True)

psi = dynSEQ_1d(x, t, V, f(x))
```
Here `V` is some potential, `f` is the initial wave function at $t_0$, `x` is the spacial domain on which the SEQ is solved and `t` is the time over which it is solved and evolves. Note that one may use the dynamical and static SEQ solvers together: the initial wave function at $t = 0$ can be obtained by calling `statSEQ_1d` and then can be fed to the dynamical SEQ solver `dynSEQ_1d`.

### 2D Diffusion Equation with Constant Diffusion Coefficient
The diffusion equation with constant coefficient and some source $f(\mathbf{r},t)$ is given by

$$\frac{\partial}{\partial t} \phi(\mathbf{\phi},t) = D\nabla^2 \phi(\mathbf{r},t) + f(\mathbf{r},t).$$

In two dimensions $\vec{\nabla}^2 = \frac{\partial^2}{\partial x^2} + \frac{\partial^2}{\partial y^2}$. The approximate solution of $\phi$ will be given by $u$. 
Using the Crank-Nicolson method the equation can be written as

$$\frac{u_{i,j}^{k+1} - u_{i,j}^k}{\Delta t} - \frac{1}{2} D \left( \frac{\delta_x^2}{(\Delta x)^2} + \frac{\delta_y^2}{(\Delta x)^2} \right) \left( u_{i,j}^{k+1} + u_{i,j}^k \right) - \frac{1}{2} f_{i,j}^k - \frac{1}{2} f_{i,j}^{k+1}$$

with 

$$\delta_x^2 u_{i,j}^k = u_{i+1,j}^k - 2u_{i,j}^k + u_{i-1,j}^k,$$

and similarly for $\delta_y^2$.  When using the same approach as in the dynamic Schrödinger equation, one arrives at an equation of the form $Z_1u^{k+1} + u^{k+1}Z_2 = C$, where $C$ is the time stamp $k$. The RHS is not easily solved for $u^{k+1}$, therefore some other method has to be used. The method that is used in this code to solve the diffusion equation with constant diffusion coefficient is the 'alternating-direction implicit method' (ADI). Every iteration in time is then split into two problems; one in the $x$ direction and one in the $y$. The solution first takes half a time step and solves for the $x$ direction: 

$$\frac{u_{i,j}^{k+1/2} - u_{i,j}^k}{\Delta t / 2} = D \frac{\delta_x^2 u_{i,j}^{k+1/2} + \delta_y^2 u_{i,j}^k}{(\Delta x)^2}$$

and then takes half a time step in the $y$ direction to get the final solution:

$$\frac{u_{i,j}^{k+1} - u_{i,j}^{k+1/2}}{\Delta t /2} = D \frac{\delta_x^2 u_{i,j}^{k+1/2} + \delta_y^2 u_{i,j}^{k+1}}{(\Delta y)^2}$$

Solving the above equation for $u^{k+1/2}$ and $u^{k+1}$ gives:

$$u^{k+1/2} = A_1^{-1}A_2 u^k$$
$$u^{n+1} = B_1^{-1}B_2 u^{n+1/2}$$

with 

$$A_1 = \left( \begin{array}{cccc} 1 + 2\lambda & -\lambda & & 0 \\ -\lambda & 1+2\lambda & -\lambda & \\ \\ & \ddots & \ddots & \ddots  \\  0& & -\lambda & 1+2\lambda \end{array} \right)$$ 
$$A_2 = \left( \begin{array}{cccc} 1 - 2\lambda & \lambda & & 0 \\ \lambda & 1-2\lambda & \lambda & \\ \\ & \ddots & \ddots & \ddots  \\  0& & \lambda & 1-2\lambda \end{array} \right)$$
$$B_1 = \left( \begin{array}{cccc} 1 + 2\mu & -\mu & & 0 \\ -\mu & 1-2\mu & -\mu & \\ \\ & \ddots & \ddots & \ddots  \\  0& & -\mu & 1-2\mu \end{array} \right)$$
$$B_2 = \left( \begin{array}{cccc} 1 - 2\mu & \mu & & 0 \\ \mu & 1-2\mu & \mu & \\ \\ & \ddots & \ddots & \ddots  \\  0& & \mu & 1-2\mu \end{array} \right)$$

with $\lambda = \frac{D \Delta t}{2(\Delta x)^2}$ and $\mu = \frac{D \Delta t}{2(\Delta y)^2}$, are $(m-2)\times (m-2)$ matrices. 

Then the final procedure, including some source term, may be 

$${u}^{n+1}_{1:m-1} = B_1^{-1}B_2A_1^{-1}A_2 u^n_{1:m-1} + \frac{1}{2}f^n + \frac{1}{2}f^{n+1}.$$

Where ${u}^{n+1}$ is the solution where the boundaries are 

The boundary conditions are Von Neumann type boundary conditions:

$$\frac{\partial}{\partial x} u(x_0,y,t) = 0 = \frac{\partial}{\partial x} u(x_n,y,t)$$
$$\frac{\partial}{\partial y} u(x,y_0,t) = 0 = \frac{\partial}{\partial x} u(x,y_n,t)$$
$$u(x,y,0) = g(x,y)$$

where $(x_0,y),(x_n,y),(x,y_0),(x,y_m)$ are the boundaries of the system. Note that the source term $f(t,x,y)$ needs to fulfill the boundary conditions of the system, i.e. $f(t,x,y)$ need to have stationary boundaries. 

The 2D diffusion equation with constant diffusion coefficient solver is called by calling `DiffEq_const_2d(D, x, y, t, g, f)`. The function returns the solution of the differential equation with the given parameters and boundary conditions `u`, which is an $N\times m \times m$ array. The cross sections are as follows:
- The solution of to the PDE at time $t_0 = j_0 \Delta t$: `u[j_0]` is an $m \times m$ array
- The solution of of the $x$ direction in time at some constant $y_0 = k_0 \Delta y$: `u[:,:,k_0]`
- The solution of of the $y$ direction in time at some constant $x_0 = l_0 \Delta x$: `u[:,l_0,:]`

The boundary condition function $g(x,y)$ should be given to the `DiffEq_const_2d` such that it gives a two dimensional array with the indices such that one obtains an array with structure for $x$ and $y$ `g[x,y]`. 
It is possible to not feed a source function to `DiffEq_const_2d`, then the source is taken to be zero, i.e. no source is present. If a source function is given as input, then the function should be of similar form as `u`, such that the `f[j]` returns a two dimensional array of the function at some time $t = j\Delta t$ and `f[:,x,y]` should be the positions of the $x$ and $y$ coordinates. 

$g$ and $f$ should be given to `DiffEq_const_2d` in function form, such that to get the 2d and 3d arrays, respectively, can be calculated by `DiffEq_const_2d` like `g(x,y)` and `f(t,x,y)`, e.g. by

```
def g(x,y):
    xv, yv = np.meshgrid(x, y)

    return np.exp(-(xv**2 + yv**2))

def f(t,x,y):
    tv, xv, yv = np.meshgrid(t, x, y)

    return np.exp(-tv)*np.sin(xv+yv)
```


## Stochastic Differential Equations

## Error of methods & recommandation
### ODEs
The absolute error of the numerical solution is given by 

$$\delta(x) = | \tilde{y}(x;h) - y(x) |$$

Consider the ODE 

$$y' = y - x^2 + 1.$$

This ODE has been solved and compared to its analytical solution on the interval $x \in [0,2]$:

$$y = (x+1)^2-\frac{1}{2}e^x.$$

In the figure below the absolute error of the numerical solution over the interval is shown.

<img src="https://github.com/DanielMikkers/DiffEq_library/blob/main/Error_interval.png" width="600" height="450">

In the figure below the absolute error of the numerical solution at the end of the interval ($x=2$) is shown at different stepsizes (or number of steps in the interval, which is $2/h$). 

<img src="https://github.com/DanielMikkers/DiffEq_library/blob/main/Error_step_size.png" width="600" height="450">

Combining the information of both figures it can be seen that the Adam-Bashforth 3 step method is the most optimal solving method and Euler the least optimal solving method. Generally, it is recommanded to use the Adam-Bashforth 3 step method (`'ab3'`). 
