# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 16:18:18 2023

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

from evolution_functions import *

################################################


def rand_mut_param(min_param, max_param, mutation_distribution):
    '''
    Selects a random parameter and its corresponding random mutation on the selected parameter
    Input:
        min_param - minimum parameter value, marker 
        max_param - maximum parameter value, marker
        mutation_distribution - type of distribution used to select mutation change
    Output:
        Two values: randomly selected parameter (integer) and the mutation for parameter (float). Type: (int, float)
    '''
    select_parameter = rand.randrange(min_param, max_param)
    mutation_change = mutation_distribution # mutation range from the 'change' column
    
    return select_parameter, mutation_change


def ancestor_min(table,pointer,number):
    
    '''
    A function which takes away 1 cell from selected ancestral subpopulation
    Input:
        table - the table with the ancestor population information
        pointer - the location of row of ancestor in the table, the index of ancestor row
        number - how many deaths occuring at a given instance 
    Output:
        Updated table with 1 cell taken away from ancestor 
    '''
    table1 = copy.deepcopy(table)
    table1.at[pointer,'Number'] = table1.at[pointer,'Number'] - number 
    
    return table1

def add_subpop(table, pointer, select_parameter, mutation_change, number):
    '''
    A function which generates a new sub-population with 1 cell resulting from mutation and its closeout line  
    Input: 
        table - the original table with all the ancestral rows. Type: pandas table [Parameter,Change,Number,Length]
        pointer - the location of ancestor sub-population. Index of row where mutation occured. Type: int
        select_parameter - random selected parameter. Type: int
        mutation_change - random change due to mutation. Type: float
        number - the number of cells in the subpopulation. Type: int 
    Output: 
        Table with two rows, where 1st row is the new sub-population with 1 cell and its closeout line
        Inserts below the ancestor, followed by the remaining subpopulation
    '''
    
    new_subpop = pd.DataFrame({'Parameter':select_parameter,'Change':mutation_change,
                 'Number': number,'Length':table['Length'][pointer]}, index = [0])        
    closeout = pd.DataFrame({'Parameter': select_parameter, 'Change': - mutation_change,
                         'Number': 0,'Length':0}, index = [0]) 
    table1 = pd.concat([new_subpop,closeout]).reset_index(drop = True)
    table_final = pd.concat([table.iloc[:pointer+1],table1,table.iloc[pointer+1:]]).reset_index(drop = True)

    return table_final


def mutation(table,min_param,max_param,mutation_p,s_neg,s_pos):
    '''
  A function which adds mutation to a random parameter of a population
  Input: 
      table - table with ancestor. Type: pandas table [Parameter,Change,Number,Length]
      min_param - the 1s parameter index. Type: int 
      max_param - the last parameter index, total number of parameters. Type: int
      mutation_p - mutation probability. Type: float 
  Return:
      A table with added mutations to random parameters. Branching from ancestor to different subpopulation
    '''
    num_mutation = fastbinorv(np.array(table['Number']), mutation_p)
    update_index = np.arange(len(table))
    for i in range(len(num_mutation)):
        for each in range(num_mutation[i]):
            parameter, change = rand_mut_param(min_param,max_param,mutation_distribution2(1,s_neg,s_pos))
            if change != 0:
                table = ancestor_min(table, update_index[i],1)
                table = add_subpop(table,update_index[i],parameter,change,1)
                update_index[i+1:] +=2     
    return table;


### Sparse and dense matrix ###
def initial_condition(max_param):
    
    '''
    Determines initial condition values for each parameter, selects ancestral values for each parameter randomly.
    This method draws from a uniform random distribution. For example used to select random affinity for chemical absorption rate etc. 
    Input:
        max_param - maximum number of parameter, total parameters. Type: int
    Output:
        Initial values of parameters of ancestor at t=0. Type: 1d array 
    ''' 
    return np.random.uniform(size = max_param)  

def sparse(table,min_param,max_param,initial_condition):
    
    '''
    Function which converts table into the sparse csr_matrix form
    Input:
        table - table to be stored as sparse csr_matrix form. Type: pandas table [Parameter,Change,Number,Length]
        min_param - the first parameter index. Type: int (typically 0).
        max_param - the maximum parameter value, the total number of parameters. Type: int.
        initial - initial conditions with initial values for all parameters, max_param number. Type: 1d array.
    Output:
        A csr_matrix form which stores the content of the table  
    '''
    col = np.concatenate([np.arange(0,max_param),table['Parameter']]) # columns are the parameter values
    row = np.concatenate([np.int64(np.zeros(max_param)), np.arange(len(table))]) # index of the rows in the table
    data = np.concatenate([initial_condition, table['Change']]) # the changes from the table usually after mutation
    
    return csr_matrix((data,(row,col)))

def number_cells(table):
    
    '''
    Gets the number of cells as a seperate column from the table where each row represents number in each subpopulation.
    Input:
        table - the table which contains the numbers column. Type: pandas table [Parameter,Change,Number,Length].
    Output: 
        Column which has all the number of cells in each subpopulation.
    '''
    number = np.zeros((len(table),1))
    for i in range(len(table)):
        number[i] = table['Number'][i]
    return number


def dense(sparse):
    '''
    Takes in a csr_matrix format to get the dense form. Calculates the actual parameter values for each subpopulation 
    Input: 
        dense - A sparse csr_matrix format which contains initial parameter values and changes from mutation. Type: csr matrix 
    Output: 
        The total dense matrix with all the values of the parameters for all rows (subpopulation). 
    '''
    return np.cumsum(sparse.toarray(), axis=0)

def sparse_dense(table,max_param,ancestor_row):
    '''
    Combining the sparse to dense operation as one step to simplify later parts.
    Input: 
        table - table with updated population sizes and rows. Type: pandas table [Parameter,Change,Number,Length].
        max_param - the total number of parameters, based on number of chemicals. Type: int
                    where the columns of the matrix represent all the parameters: K, r0, alpha, beta, rho_plus, rho_minus
        ancestor_row - initial value of each parameter for the ancestral population. Type: 1d array
                       Single row with the total number of parameters column.
    Output: 
        The dense matrix with updated population sizes after mutation.
    '''
    # create sparse matrix from the table with (ancestor + 1 round of mutation)
    Z = sparse(table, 0,max_param,ancestor_row) # creating a sparse form from the resulting mutation 
    X = dense(Z) # total dense matrix for species 1 after 1 round of mutation
    return X

def biomass(table):
    '''
    Calculates the biomass of species, which is the number of cells * cell length. 
    Input: 
        table - table which contains the species and subpopulations and their population size and length of cells
    Output: 
        The biomass value for a species and its subpopulation 
    '''
    biomass = table['Number'] * table['Length'] 
    return biomass


def label_maker(biomass):
    '''
    Determines how many subpopulation there are and gives a unique label for all distrinct subpopulation
    Input: 
        biomass - a column which contains the biomass of species, biomass = number cells * length. Type: 1d array
    Output:
        The number of the subpopulations in a table where this number represents each distinct subpopulation. Used to make label in plot
    '''
    biomass_label =[]
    for i in range(len(biomass)):
        if biomass[i] != 0:
            biomass_label.append(biomass[i])
    return len(biomass_label)

### Growth of cell ###
def initial_size(number):
    '''
    Function that generates the initial size of cell randomly. A random Length is generated for each row i.e subpopulation.
    Input:
        number - the column with number of each cells in each subpopulation from a single ancestor. Type: 1d array
    Return:
        A random initial Length for subpopulation. Initial size is generated if there are cells i.e not closeout lines.
    '''
    size0 = np.zeros((len(number),1)) # setting up an empty column to contain initial length of cells in table
    for row in range(len(number)):
        if number[row] !=0:
            random_size = np.round(abs(np.random.normal(1.5,0.5,1)),2) # distribution around 1.5, with s.d 0.5
            size0[row] = random_size
        else:
            size0[row] = 0  
    return size0

def get_ncols(params, param_names): # run once 
    
    '''
    To get the number of columns for each parameter in a dictionary.
    Input: 
        params - the parameter dictionary, including the array of values for each component. Type: dictionary
        param_names - the name of the parameters. Type: list 
    Return: 
        Gets the number of columns for each parameter.
    '''
    return np.array([params[_].shape[1] for _ in param_names]);

def cut_species(s,table):
    '''
    A function which gives cut points i.e cut points to seperate each ancestor and its corresponding subpopulation 
    Input:
        s - the updated species biomass from dynamics calculation. Type: 1d array
        table - the updated table after mutation with species and their subpopulation. Type: pandas table [Parameter,Change,Number,Length].
    Return:
        Ancestor and its subpopulation cut at the correct points. 
        If M species have M1 and M2 subpopulation and species H zero, cut points are [2,0]
    '''
    s = np.array(s).flatten() # 1d array of biomass
    cuts = [table[i].shape[0] for i in range(len(table))] # how many subpopulation branching from each ancestor
    s_all = []
    upd = np.arange(len(cuts)) # the ancestor index, if there are two ancestors, [0,1] first and second ancestor 0 and 1
    for n in range(len(cuts)):
        m = s[upd[n]:upd[n]+cuts[n]]
        upd[n+1:]+=cuts[n]-1
        s_all.append(m)
    return s_all

def number_cell_update(table,new_number):
    '''
    Gives a table with updated  number of cells for each subpopulation
    Input:
        table - a table with subpopulation information. Type: pandas table [Parameter,Change,Number,Length]
        new_number - the new value of the number of cells of each subpopulation. Type: list 
    Output:
        The table with the updated number of cells.
    '''
    table1 = copy.deepcopy(table)
    table1.loc[:,'Number'] = new_number

    return table1

# might be better in evolution_functions.py
def normcond(N,p):
    ### Author: Caroline ###
    p[p == 0] = 1e-9;  # avoid division errors
    return np.logical_and((N > 9 * p / (1-p)), (N > 9 * (1-p) / p));

# might be better in evolution_functions.py
def fastbinorv(N,p):
    N = N.astype(np.int32)
    #p = p.astype(np.int32)
    ### Author: Caroline ###
    if (np.size(N) <= 0):
        #print("error: N must not be empty");
        return;
        
    if (np.ndim(N) > 1 and np.shape(N)[1] > 1):
        print("error: N must be a one dimensional array");
        return;
        
    thresh_b2p = 30; # threshold for binomial to poisson dist
    thresh_p2n = 800; # threshold for poisson to normal dist
    
    result = np.zeros(np.shape(N));
    
    if (np.isscalar(p)):
        p = np.ones(np.shape(N)) * p;
        
    if (N.min() > thresh_p2n and normcond(N,p).all()):
        result = np.round(np.random.normal(N * p, np.sqrt(N * p * (1 - p)))).astype(int);
    else:
        bino_ind = np.nonzero(np.logical_and(N > 0, N < thresh_b2p));
        pois_ind = np.nonzero(np.logical_and(N >= thresh_b2p, 
                np.logical_or(N < thresh_p2n, np.logical_not(normcond(N,p)))));
        norm_ind = np.nonzero(np.logical_and(N >= thresh_p2n, normcond(N,p)));
        
        if (np.size(bino_ind) > 0):
            result[bino_ind] = np.random.binomial(N[bino_ind],p[bino_ind]);
            
        result[pois_ind] = np.random.poisson(N[pois_ind] * p[pois_ind]);
        result[norm_ind] = np.round(np.random.normal(N[norm_ind] * p[norm_ind],
              np.sqrt(N[norm_ind] * p[norm_ind] * (1 - p[norm_ind])))).astype(int);
        
    result[result < 0] = result[result < 0] * 0;
    result[result > N] = result[result > N] * 0 + N[result > N];
    
    return result.astype(int);

# might be better placed in simulations, but I need to have some sort of mutation step in the tree algorithm
def mutation_HM_step(table,params_HM,matrix_form,mut_param,min_param,max_param):
    
    '''
  A function which adds mutation to an ancestor. The only parameter allowed to mutate if fp, cost function.
  There is a chance of mutation occuring for each subpopulation equally, depending on Number of mutants i.e "Number"
  Input: 
      table - the starting table with ancestor, with columns: parameter, change and number of cells 
      params_HM - dictionary with the original parameters of the ancestors i.e H and M, more specifically, fp
      i - the index to represent which species we are handling, if H - 0, M - 1, typically inside list comprehension
      min_param - the 1s parameter index
      max_param - the last parameter index, total number of parameters
      mutation_p - mutation probability
  Return:
      A table with added mutations to random parameters. Branching from ancestor to different subpopulation
    '''

    s_neg = mut_param['s_neg']
    s_pos = mut_param['s_pos'] 
    mutation_p = mut_param['mutation_p'] 
    fp_upper = 1.0

    num_mutation = fastbinorv(np.array(table['Number']), mutation_p)
    update_index = np.arange(len(table))
    for i in range(len(num_mutation)):
        for each in range(num_mutation[i]):
            row = matrix_form[update_index[i]]
            fp = row[0]
            parameter, change = rand_mut_param(min_param,max_param,mutation_distribution2(1,s_neg,s_pos))
            change = ((change + 1) * fp)  
            change[change > fp_upper] = fp_upper # Upper bound for the maximum allowed value for parameter fp
            change = change - fp # To compensate for cumsum() takeaway the ancestor fp
            if change != 0:
                table = ancestor_min(table, update_index[i],1)
                table = add_subpop(table,update_index[i],parameter,change,1)
                update_index[i+1:] +=2   
    return table;
