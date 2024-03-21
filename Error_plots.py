import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm 

from DEsolver import ode_ivp

def ODEF(x,y):

    return y-x**2 + 1.0

method = np.array(['euler', 'collatz', 'heun', 'rk4', 'ab3', 'ab4'])
color = cm.rainbow(np.linspace(0, 1, np.size(method)))

x_int = np.arange(0,2,0.02)
Exact_1 = lambda x: (x+1)**2-0.5*np.exp(x)
y0 = 0.5

fig = plt.figure()

for i,c in enumerate(color):
    y = ode_ivp.integrator(x_int, y0, ODEF, method[i])
    delta_err_int = np.abs(y - Exact_1(x_int))
    plt.plot(x_int,delta_err_int,color=c,label="{}".format(method[i]))

plt.xlabel(r'$x$')
plt.ylabel(r'$\delta(x)$')
plt.title(r'Absolute errors $\delta(x)=|\tilde{y}(x;0.02)-y(x)|$ to ODE $dy/dx=y-x^2 + 1.0$')
plt.legend(prop={'size': 4})
plt.yscale('log')
plt.savefig('Error_interval.png', dpi=300)
plt.close()


step1 = 10
step2 = 1000
numb = step2 - step1
step = np.linspace(step1,step2,num=numb)

delta_err_step = np.zeros((np.size(method),np.size(step)))

exact_2 = lambda x: np.exp(x)

for i in range(np.size(step)):
    for j in range(np.size(method)):
        x_step = np.linspace(0,2,num=int(step[i]))
        y = ode_ivp.integrator(x_step, y0, ODEF, method[j])
        delta_err_step[j,i] = np.abs(y[-1] - Exact_1(2))

for k,c in enumerate(color):
    plt.plot(step,delta_err_step[k,:],color=c,label="{}".format(method[k]))

plt.xlabel(r'number of steps (2/h)')
plt.ylabel(r'$\delta(2;h)$')
plt.title(r'Absolute errors $\delta(2;h)=|\tilde{y}(2;h)-y(2)|$ to ODE $y=y - x^2 + 1.0$')
plt.legend(prop={'size': 4})
plt.xscale('log')
plt.yscale('log')
plt.savefig('Error_step_size.png', dpi=300)
plt.close()
