# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 10:39:36 2023

@author: jamil
"""

######################## Example skeleton for new algorithm relying on sparse matrices #####################################


import numpy as np
from random import random
import random as rand
from numpy import random
import pandas as pd
import matplotlib.pyplot as plt

import scipy
from scipy.sparse import csr_matrix
from scipy.integrate import solve_ivp
from scipy import io 

import copy
from time import sleep
import pickle
import os

###### NOTE: THIS IS A ROUGH SKELETON. A LOT OF CODE WILL BE INCORRECT. THIS IS JUST TO GIVE A ROUGH IDEA.


"""
        I THINK TO DO WHAT I WANT TO DO (UPDATE A VARIABLE ITERATIVELY USING A FUNCTION), I NEED TO MAKE A CLASS AND CALL MAGIC METHODS.
        ALTERNATIVELY, I COULD RETURN A DUPLICATE OBJECT OF LIST_SPARSE (UPD_LIST_SPARSE), BUT THIS IS MORE MEMORY INTENSIVE.
"""

# species population dynamics, chemical production and species evolution over a single time step.
# FUNCTION CALL. upd_list_sparse, upd_chemicals, ... = community_dynamics(upd_list_sparse,etc...)
# NOTE: WE WILL NEED ANOTHER FUNCTION TO INITIALLY CONSTRUCT SPARSE MATRICES FROM ANCESTORAL SPECIES.
def community_dynamics(list_sparse,param_dict,chemicals,t,...):
    """

    Parameters
    ----------
    list_sparse : TYPE list of sparse matrices, TYPE scipy.sparse.csr_matrix
        DESCRIPTION. List of sparse matrices, one matrix per species. Contains information on species sub-population
            parameter change from ancestor, biomass, etc.
    param_dict : TYPE dictionary
        DESCRIPTION. Key =  name of parameter, Value 1 = ancestoral parameter value, 
            Value 2 = col number associated with parameter in sparse matrix.
            (This is to match Indra's model a bit better. Might change.)
    chemicals : TYPE numpy array
        DESCRIPTION. Concentration/relevent unit of each chemical in a No_chemicals x 1 dim numpy array. 
        (This might change, need to see how chemicals are stored and associated with relevent producing species in Indra's model.)
    t : TYPE float64?
        DESCRIPTION. Time step for community population dynamics calculations.
    ... : TYPE ?
        DESCRIPTION. Any other required function arguments. (e.g. might need info on which parameters are undergoing mutation)

    Returns
    -------
    list_sparse : TYPE list of sparse matrices, TYPE scipy.sparse.csr_matrix
        DESCRIPTION. Updated list_sparse after dynamics calculations and evolution.
    upd_chemicals : TYPE numpy array
        DESCRIPTION. Updated chemical species after dynamics calculations. 
    ... : TYPE ?
        DESCRIPTION. Any other required function outputs.
    """
    
    for i in range(length(list_sparse)): # index list of sparse matrices 
        
        # Extract matrix for single species
        sprse = list_sparse[i] # extract sparse matrix of species i 
        dnse = dense(sprse) # convert matrix to dense form. TYPE : np.matrix. FUNCTION THE SAME AS INDRA'S
        
        # Dynamics Calculations (Population dynamics and chemicals)
        biomass0 = biomass(dnse[:,param_dict['biomass'][1]]) # extract and/or calculate biomass from biomass col in dense matrix. CHANGE FROM INDRA'S.
        
        biomass_tdeltat, upd_chemicals = dynamics(dnse,param_dict,chemicals,biomass0,t,...) # some function containing scipy.solve_ivp to
        #   calculate species dynamics and chemical production over time t. SHOULD BE THE SAME AS INDRA'S MODEL.
        dnse[:,param_dict['biomass'][1]] = biomass_tdeltat # update dense matrix
        
        # Mutation
        new_mutant = mutation(dnse,param_dict,t) # allow species population to undergo mutation. Returns np.array matching dense matrix
        #   with new subpopulation. CHANGE FROM INDRA'S.
        dnse = np.vstack((dnse,new_mutant)) # Update dense matrix. THIS ISN'T RIGHT, IT NEEDS TO BE INSERTED IN THE CORRECT BRANCH. I THINK THIS 
        #   SHOULD BE EASY, JUST INSERT ROW AFTER WHATEVER SUBPOPULATION MUTATES. ADD TO MUTATION FUNCTION?
        
        # Updating list_sparse
        sprse_upd = sparse(dnse) # convert dense back to sparse matrix. CHANGE FROM INDRA'S.
        
        list_sparse[i] = sprse_upd # update list_sparse
    
    # IDEALLY I'D PREFER NOT TO DO THIS.
    return list_sparse, upd_chemicals, ...

# NEW FUNCTION FOR CONVERTING DENSE MATRIX TO SPARSE
def sparse(dnse):
    
    # initialise matrix
    tobe_sparse = np.empty(dnse.shape)
    
    tobe_sparse[0,:-1] = dnse[0,:-1] # ancestor row kept the same as the dense matrix
    tobe_sparse[:,-1] = dnse[:,-1] # biomasses stay the same as the dense matrix
    
    # convert into 0s form - reverse cumsum. USE NUMBA
    #@njit
    for i in range(dnse.shape[0])[1:]:
        
        for_sparse[i,:-1] = dnse[i,:-1] - dnse[(i-1),:-1]
    
    sprse_mat = csr_matrix(for_sprse) # convert to scipy.sparse_csr_matrix
    
    return sprse_mat

