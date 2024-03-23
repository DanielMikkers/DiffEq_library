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

### Poisson Equation Type PDEs (Elliptic)

### Heat Equation Type PDEs (Parabolic)

### Wave Equation Type PDEs (Hyperbolic)

### Hyperbolic PDEs

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

### PDEs
