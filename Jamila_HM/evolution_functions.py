# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 16:18:41 2023

@author: jamil
"""

import numpy as np
import random
import random as rand
from random import random
from numpy import random

import pandas as pd
import scipy
from scipy.sparse import csr_matrix
from scipy.integrate import solve_ivp

import time
from time import sleep

from tqdm import tnrange, tqdm_notebook
from tqdm.notebook import trange, tqdm

# from seaborn import axes_style
# import seaborn as sns

import copy
import pickle

################################################

def mutation_freq(mut_prob, n, size):
    '''
    A function which gives the number of mutants in a sub-population based on binomial distribution
    Input: 
        mut_prob - probability of success of a particular event occuring, here success is mutation 
        n - the size of sub-population, or number of cells in each subpopulation. The number of independent trial
            Make sure the selected 'n' are more than 0
        size - the number of repeats of an event. For mutation, choose size = len(n) going over once
               For example, size = 10 means flipping a coin 10 times to get number of successes of 'heads'      
    Output: 
        The number of success for a given number of trials, the number of mutants in a sub-population (1d array)     
    '''
    num_mutants = np.random.binomial(n = n, p = mut_prob,size=size) # binomial distribution for subspecies
    return num_mutants

def mutation_distribution():             
# Author: Alex Yuan
    return np.round(np.random.normal(),2)

def null_distribution():             
    return -1

def mutation_distribution2(mutants,s_pos,s_neg):
    '''
    The distribution from Li et.al paper (from Dunham lab haploid data) for selecting mutation value for parameter fp
    Inputs:
        mutants - the number of mutants in subpopulation, selected using binomial distribution. Type: 1d array
        s_pos - positive mutation effect value, enhancing mutation. Type: int
        s_neg - negative mutation effect value, diminishing mutation. Type: int 
    Output:
        Mutation values i.e the changes in the parameters of the mutants.
    '''
    u2 = np.random.random(mutants)
    lim = s_neg/(s_pos + s_neg)
    m = np.zeros(mutants)
    
    index = np.where([u2[i]<=lim for i in range(mutants)])[0]
    m[index] = [s_neg*np.log(u2[index[i]]*(s_pos+s_neg)/s_neg) for i in range(len(index))]

    index = np.where([u2[i]>lim for i in range((mutants))])[0]
    m[index] = [-s_pos*np.log((1-u2[index[i]])*(s_pos+s_neg)/s_pos) for i in range(len(index))]
    m = np.array([-1 if m[i]<-1 else m[i] for i in range(len(m))])
    
    return m 

