# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.line_search.html

import numpy as np
from scipy.optimize import line_search
from scipy.optimize import minimize

import jax

def obj_func(x):
    return (x[0])**2+(x[1])**2

def obj_grad(x):
    return [2*x[0], 2*x[1]]

# We can find alpha that satisfies strong Wolfe conditions.

start_point = np.array([1.8, 1.7])
# start_point = np.array([27.0, -11.0])
# start_point = np.array([-12.35, 2011.0])
search_gradient = np.array([-1.0, -1.0])
search_gradient = np.array([-1.8, -1.7])


jax.grad(obj_func)(start_point)  # gradient at x
gfun = jax.grad(obj_func)
hfun = jax.hessian(obj_func)
gfun(start_point)
hfun(start_point)

hinv = jax.numpy.linalg.inv(hfun(start_point))
gvals = gfun(start_point)
hinv
gvals
step = -hinv.dot(gvals)
step

line_search(obj_func, obj_grad, start_point, search_gradient)

start_point + step  # perfect


obj_grad(start_point)

# returns alpha, fc, gc, newf, oldf, newslope
ls = line_search(obj_func, obj_grad, start_point, search_gradient)
ls
# (1.0, 2, 1, 1.1300000000000001, 6.13, [1.6, 1.4])
ls[0] * search_gradient
line_search(obj_func, obj_grad, [1.6, 1.4], search_gradient)
newx = start_point + ls[0]*search_gradient
line_search(obj_func, obj_grad, newx, np.array([1.6, 1.4]))
line_search(obj_func, obj_grad, newx, search_gradient)
# returns alpha, # f calls, # g calls, new f, old f, new slope
# desired step is [-1.0, 1.0] which is 1.0 * [-1.0, 1.0]
dir(ls)

line_search(obj_func, obj_grad, start_point, np.array([-2.0, 3.0]))  # bad search dir does not converge
# step too large, converges
line_search(obj_func, obj_grad, start_point, np.array([-3.0, 3.0]))
# actual step:
np.array([-3.0, 3.0]) * line_search(obj_func, obj_grad, start_point, np.array([-3.0, 3.0]))[0]

sdir = np.array([-3.0, 3.0])
sdir = np.array([-2.0, 2.0])
sdir = np.array([-1.0, 1.0])
sdir = np.array([-0.5, 0.5])  # step too small
sdir = np.array([-1.2, 1.2])
alpha, fc, gc, newf, oldf, newslope = line_search(obj_func, obj_grad, start_point, sdir)
alpha
newslope
fc
gc
sdir * alpha



(alpha, fc, *all) = line_search(obj_func, obj_grad, start_point, search_gradient)
alpha
fc
all

def f(x):
    return x[0]**2.0 - x[0]*x[1]

# def g(x):
#     return np.array([2*x[0], 3*x[1]**2])
g = jax.grad(f)
h = jax.hessian(f)

x = np.array([7.0, -3.0])
g(x)
h(x)
minimize(f, x, method='Newton-CG', jac=g, hess=h)

hinv = jax.numpy.linalg.inv(h(x))
gvals = g(x)
hinv
gvals
step = -hinv.dot(gvals)
step



fruits = ("apple", "mango", "papaya", "pineapple", "cherry")

(green, *tropic, red) = fruits

print(green)
print(tropic)
print(red)



# scipy.optimize.line_search
# scipy.optimize.line_search(f, myfprime, xk, pk,
#   gfk=None, old_fval=None, old_old_fval=None,
#   args=(), c1=0.0001, c2=0.9, amax=None,
#   extra_condition=None, maxiter=10)
# Find alpha that satisfies strong Wolfe conditions.

# Parameters
# fcallable f(x,*args)   Objective function.

# myfprimecallable f’(x,*args)  Objective function gradient.

# xk ndarray Starting point.
# pkndarray Search direction.
# gfk ndarray, optional  Gradient value for x=xk
#   (xk being the current parameter estimate). Will be recomputed if omitted.

# old_fval float, optional Function value for x=xk. Will be recomputed if omitted.
# old_old_fval float, optional Function value for the point preceding x=xk.

# argstuple, optional Additional arguments passed to objective function.

# c1 float, optional Parameter for Armijo condition rule.
# c2 float, optional Parameter for curvature condition rule.

# amaxfloat, optional
# Maximum step size

# extra_condition callable, optional A callable of the form
#   extra_condition(alpha, x, f, g) returning a boolean. Arguments are the proposed
#   step alpha and the corresponding x, f and g values. The line search accepts
#   the value of alpha only if this callable returns True. If the callable returns
#   False for the step length, the algorithm will continue with new iterates. The callable is only called for iterates satisfying the strong Wolfe conditions.

# maxiterint, optional Maximum number of iterations to perform.

# Returns alpha float or None
#   Alpha for which x_new = x0 + alpha * pk, or
#     None if the line search algorithm did not converge.

# fcint Number of function evaluations made.

# gcint  Number of gradient evaluations made.

# new_fval float or None New function value f(x_new)=f(x0+alpha*pk), or
#    None if the line search algorithm did not converge.

# old_fval float Old function value f(x0).
# new_slope float or None The local slope along the search direction at the new value <myfprime(x_new), pk>, or None if the line search algorithm did not converge.
