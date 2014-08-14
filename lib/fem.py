#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Finite Element utilities for radially symmetric functions.
Author: Tyler Parker (tparker@umass.edu), UMASS Amherst
"""

import numpy as np
import matplotlib.pyplot as plt

class Basis:
    """Collection of BasisFunctions spanning function in region.

    Args:
      r_h (float): radius of region
      h (int): number of basis functions along a radius
    
    Attributes:
      h (int): number of basis functions along a radius
      r_h (float): radius of region
      r (CenteredList): list of ith radii, with i in range -h <= i <= h

    Operators:
      basis[i] (BasisFunction): returns the ith BasisFunction, with i in 
        range -h+1 <= i <= h-1
      len(basis) (int): h, number of nonnegative indexed BasisFunctions
    """

    def __init__(self, r_h, h):
        if h % 2 == 0:
            raise Exception("must be odd")

        self.h = h
        self.r = np.linspace(-r_h, r_h, 2*h+1)

    def __getitem__(self, index):
        if not index in range(1, 2*self.h):
            raise Exception("Index out of bounds")
        return BasisFunction(self, index)

    def __len__(self):
        return self.h

    def __iter__(self):
        return BasisIterator(self)

class BasisIterator:
    """Iterates over all BasisFunctions in basis"""
    def __init__(self, basis):
        self.basis = basis
        self.current = -len(basis)+1

    def next(self):
        try:
            self.basis[self.current]
        except Exception:
            raise StopIteration
        self.current = self.current + 1
        return self.basis[self.current-1]

class BasisFunction:
    """Basis element in piecwise linear function space.

    Args:
      basis (Basis): the containing basis
      index (int): index in basis, BasisFunctions with same basis, index, 
        isGrad, are indistinguishable.
      grad (bool): If True: self is member of gradient of basis
        If False: self is a member of basis

    Attributes:
      basis (Basis): the containing basis
      index (int): index in basis, BasisFunctions with same basis, index, 
        isGrad, are indistinguishable.
      isGrad (bool): If True: self is member of gradient of basis. If 
        False: self is a member of basis

    Operators:
      basisFunction(r): evaluate basisFunction at r. see __call__(self, r)
    """

    def __init__(self, basis, index, grad=False):
        self.basis = basis
        self.index = index
        self.isGrad = grad

    def __call__(self, r):
        """Compute function for specified r values.
        Args:
          r (float): r value to compute function
          r (numpy.ndarray): r value to compute function
        
        Returns:
          (float) : Value of BasisFunction at specified r value
          (numpy.ndarray) : Value of BasisFunction at specified r values
        """

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

    def inner(self, function=None):
        """Compute inner product of BasisFunction with function.
        Args:
          function (BasisFunction): Inner product between self and function.
            if None then assume function(x)==1

        Returns:
          (float): int_0^basis.r_h self(r)*function(r) r**2 dr
        """
        r = self.basis.r
        i = self.index
        if function == None:
            return -(r[i-1]**2+r[i]**2+r[i]*r[i+1]+r[i+1]**2+r[i-1]*(r[i]+r[i+1]))*(r[i-1]-r[i+1])/12
        if type(function) == BasisFunction:
            if function.index == self.index:
                return -(r[i-1]-r[i+1])*(r[i-1]**2+3*r[i]**2+2*r[i]*r[i+1]+r[i+1]**2+r[i-1]*(2*r[i]+r[i+1]))/30
            if abs(function.index - self.index) == 1:
                pass

        R, dr = np.linspace(self.basis.r[self.index-1], 
                            self.basis.r[self.index+1], N, retstep=True)
        F1 = self(R)
        F2 = function(R)
        return np.sum(F1*F2*R**2)*(self.basis.r[self.index+1]-self.basis.r[self.index-1])/N/dr

    def grad(self):
        """Returns gradient of self"""
        return GradBasisFunction(self)
    

class GradBasisFunction(BasisFunction):
    pass
