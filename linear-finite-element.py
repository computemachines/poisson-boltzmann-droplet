#! /usr/bin/python

import numpy as np
import numpy.linalg as la
from matplotlib import pyplot as plt
import sys

sys.path.insert(0, 'lib')
from fem import *

F = 0.174
u_0 = 2

# distict basisfunctions along radii

plt.rc('text', usetex=True)
np.set_printoptions(precision=2, linewidth=300, threshold=10000)

def computeA(phi, F):
    """Generate forcing matrix from phi and F"""
    A = np.zeros((2*len(phi)-1, 2*len(phi)-1))
    for i in range(2*len(phi)-1):
        for j in range(2*len(phi)-1):
            if abs(i-j) <= 1:
                A[j][i] = phi[i+1].grad().inner(phi[j+1].grad()) + F*phi[i+1].inner(phi[j+1])
    return np.matrix(A)

def computeV(phi, Fu_0):
    V = np.zeros((2*len(phi)-1))
    for i in range(2*len(phi)-1):
        V[i] = Fu_0*phi[i+1].inner()
    return np.matrix(V).transpose()



def PB(h):
    h = int(h)
    if h%2 ==0:
        h = h+1

    print h
    phi = Basis(100, h)

    A = computeA(phi, F)
    V = computeV(phi, F*u_0)

    U = la.solve(A,V) - u_0
    return U

def plot(U):
    if type(U)!=list:
        U = [U]
    fig = plt.figure(figsize=(5.5, 6), dpi=300, facecolor='w', edgecolor='k')
    plt.ylim((-2, 0))
    for u in U:
        R = np.linspace(-100, 100, len(u))
        plt.plot(R,u, 'k')

    plt.title("Linear Poisson Boltzmann Solution in Droplet")
    plt.ylabel(r"$\bar\phi$")
    plt.xlabel("r (nm)")
#    plt.tight_layout()
    fig.savefig('../reports/2014-08-15/images/plot.png', dpi=300)
    
#    plt.show()

plot(PB(1000))

