#! /usr/bin/python

import numpy as np
import numpy.linalg as la
from matplotlib import pyplot as plt
import sys

sys.path.insert(0, 'lib')
from fem import *

F = 0.1
u_0 = -2

# distict basisfunctions along radii
h = 1001
phi = Basis(100, h)

np.set_printoptions(precision=2, linewidth=300, threshold=10000)

def computeA(F):
    """Generate forcing matrix from phi and F"""
    A = np.zeros((2*len(phi)-1, 2*len(phi)-1))
    for i in range(1, 2*len(phi)):
        for j in range(1, 2*len(phi)):
            if abs(i-j) <= 1:
                A[j-1][i-1] = phi[i].grad().inner(phi[j].grad()) + F*phi[i].inner(phi[j])
    return np.matrix(A)

def computeV(A, Fu_0):
    V = np.zeros((2*len(phi)-1))
    for i in range(1, 2*len(phi)):
        V[i-1] = phi[i].inner(lambda x: -Fu_0)
    return np.matrix(V).transpose()

A = computeA(F)
V = computeV(A, F*u_0)

U = la.solve(A,V) + u_0
r = np.linspace(0, 100, 10000)
R = np.linspace(-100, 100, 2*len(phi)-1)

def plot(U):
    fig, ax1 = plt.subplots()
    ax1.plot(R,U)
    plt.show()

plot(U)

