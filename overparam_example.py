from autograd import numpy as anp, grad
from scipy.optimize import minimize
from matplotlib import pyplot

N = 100
x = anp.arange(0, 10, 10/N)
y = x + anp.random.normal(size=N)

def plotpoints():
    pyplot.plot(x, y, 'r.')

def mll_fit():
    def obj(a):
        err = y - a*x
        return 0.5 * anp.dot(err, err)
    res = minimize(obj, 1.0, jac=grad(obj))
    print(res['message'])
    a = res['x'][0]
    
    print("MLL slope estimate: y =", a, "* x")
    pyplot.plot(x, y, 'r.'); pyplot.plot(x, a*x, 'b-')
    return

def overparam_sample():
    def obj(th):
        a = th[0]; xi = th[1:]
        err = y - a*x - xi
        return 0.5 * anp.dot(err, err)
    res = minimize(obj, anp.random.normal(size=N+1), jac=grad(obj), method='CG', options={'maxiter':1000})
    print(res['message'])
    a = res['x'][0]; xi = res['x'][1:]
    
    print("Overparam slope sample: y =", a, "* x")
    pyplot.plot(x, y, 'r.'); pyplot.plot(x, a*x, 'b-')
    return

def very_overparam_sample():
    def obj(th):
        a = th[0]; xi = th[1:]
        xi = anp.reshape(xi, (100, N))
        err = y - a*x - anp.sqrt(3/N)*anp.sum(xi, axis=0)
        return 0.5 * anp.dot(err, err)
    th0 = 1 - 2*anp.random.random(size=100*N+1)
    th0[0] = anp.random.normal()
    res = minimize(obj, th0, jac=grad(obj), method='CG', options={'maxiter':1000})
    print(res['message'])
    a = res['x'][0]; xi = res['x'][1:]
    
    print("Very overparam slope sample: y =", a, "* x")
    pyplot.plot(x, y, 'r.'); pyplot.plot(x, a*x, 'b-')
    return
