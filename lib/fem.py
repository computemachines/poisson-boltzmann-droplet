#! /usr/bin/python
# -*- coding: utf-8 -*-


# -------- Usage --------
# phi = Basis()
# phi[1]            => first basis function
# phi[50](3.81)     => value of 50th basis function at r=3.81
# phi[50]([...])    => value of 50th basis function at values in array

import numpy as np
import matplotlib.pyplot as plt

class CenteredList(list):
    def __getitem__(self, index):
        return list.__getitem__(self, index+(len(self)-1)/2)

class Basis:
    # h = size of basis
    # r[i] = ith r value
    def __init__(self, *args): #Basis(r_h, h) or Basis(r)
        if type(args[0]) == np.ndarray:
            self.h = len(args[0])
            self.r = args[0]
        else:
            self.h = args[1]
            self.r = CenteredList(np.linspace(-args[0], args[0], args[1]))
        if self.h % 2 == 0:
            raise Exception("must be odd")
    def __getitem__(self, index):
        print (index, self.h)
        if index <= -(self.h-1)/2 or index >= (self.h-1)/2:
            raise Exception("Index out of bounds")
        return BasisFunction(self, index)
    def __len__(self):
        return self.h
    def __repr__(self):
        return "Basis("+str(self.r[self.h/2])+", "+str(self.h)+")"
    def __iter__(self):
        return BasisIterator(self)

class BasisIterator:
    def __init__(self, basis):
        self.basis = basis
        self.current = -(len(basis)-1)/2 +1
        print self.basis
        print self.current
    def next(self):
        print self.current
        try:
            self.basis[self.current]
        except Exception:
            raise StopIteration

        print self.current
        self.current = self.current + 1
        print self.current
        print self.basis.r
        return self.basis[self.current-1]
    
class BasisFunction:
    def __init__(self, basis, index, grad=False):
        self.basis = basis
        if type(index) != int:
            raise Exception("Basis must be integer indexed")
        self.index = index
        self.isGrad = grad

    def __repr__(self):
        return "BasisFunction("+str(self.basis)+", "+str(self.index)+", "+str(self.isGrad)+")"
    def __call__(self, r):
        if type(r) == np.ndarray:
            return np.array([self(r[i]) for i in range(len(r))])

        basis = self.basis
        i = self.index

        if basis.r[i-1] <= r and r < basis.r[i]:
            if self.isGrad:
                return 1/float(basis.r[i]-basis.r[i-1])
            return r/(basis.r[i]-basis.r[i-1]) + basis.r[i-1]/(basis.r[i-1]-basis.r[i])
        if basis.r[i] <= r and r < basis.r[i+1]:
            if self.isGrad:
                return -1/float(basis.r[i+1]-basis.r[i])
            return -r/(basis.r[i+1]-basis.r[i]) + basis.r[i+1]/(basis.r[i+1]-basis.r[i])
        return 0

    def inner(self, function, N=10 ): # N is subdivisions per basis support
        R = np.linspace(self.basis.r[self.index-1], 
                        self.basis.r[self.index+1], N)
        F1 = self(R)
        F2 = function(R)
        return np.sum(F1*F2*R**2)*(self.basis.r[self.index+1]-self.basis.r[self.index-1])/len(R)

    def grad(self):
        return BasisFunction(self.basis, self.index, True)

    
    
