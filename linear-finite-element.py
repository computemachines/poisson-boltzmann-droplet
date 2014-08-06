#! /usr/bin/python

import numpy as np
import numpy.linalg as la
from matplotlib import pyplot as plt
import sys

sys.path.insert(0, 'lib')
from fem import *

F = 10
u_0 = 2


h = 3 # distict steps along radii
p = 2*(h-2)+1  #number or basis
d = 100/float(h)

phi = Basis(100, h)

def computeA(F):
    
    A = np.zeros((p,p))
    for i in range(-1, 2):
        print float(i)/h
        for j in range(-1, 2):
            print (i, j, h, p, len(phi))
            if abs(i-j) <= 1:
                A[j+h-2][i+h-2] = phi[i].grad().inner(phi[j].grad()) + F*phi[i].inner(phi[j])
    return np.matrix(A)

def computeV(A, Fu_0):
    V = np.zeros((p))
    for i in range(-h+2, h-1):
        V[i+h-2] = phi[i].inner(lambda x: Fu_0)
    return np.matrix(V).transpose()

A = computeA(F)
V = computeV(A, F*u_0)
#print V[h-1]
U = la.solve(A,V)
r = np.linspace(0, 100, 10000)

def plot(U):
    plt.title(str(sys.argv[1])+" elements")
    plt.plot(U)
    plt.show()

plot(U)
