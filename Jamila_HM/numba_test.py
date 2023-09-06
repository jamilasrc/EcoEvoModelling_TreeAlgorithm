# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 14:13:54 2023

@author: jamil
"""

from numba import jit
import numpy as np
import time

def timer(func):
    t11 = time.time()
    func(x)
    t21 = time.time()
    return t21-t11


def standard(a):
    trace = 0.0
    for i in range(a.shape[0]):   
        trace += np.tanh(a[i, i]) 
    return a + trace    

@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def go_fast(a): # Function is compiled to machine code when called the first time
    trace = 0.0
    for i in range(a.shape[0]):   # Numba likes loops
        trace += np.tanh(a[i, i]) # Numba likes NumPy functions
    return a + trace              # Numba likes NumPy broadcasting

x = np.arange(10000 ** 2).reshape(10000, 10000)

tot_time_s = 0
for i in range(1000):
   tot_time_s += timer(standard)

tot_time_f = 0 
for i in range(1000):
    tot_time_f += timer(go_fast)


