# import matplotlib.pyplot as plt
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

Olive = '#CAFF70'
Blue = '#2CBDFE'
Sea = '#1E90FF'
Green = '#47DBCD'
Baby_Pink = '#F3A0F2'
Magenta = '#EE1289'
Purple = '#BF3EFF'
Violet = '#7D26CD'
Lilac = '#8470FF'
Dark_blue = '#0000CD'
Green_tea = '#66CDAA'
Pink = '#FF69B4'
Hot_pink = '#EE00EE'
Barbie = '#EE30A7'
Dream = '#00EE76'
Turquoise = '#00E5EE'
Berry = '#D02090'
Sky = '#87CEFF'

updated_colour = [Blue, Baby_Pink, Green,Sea, Olive, Purple, Violet,Dark_blue,Magenta,Lilac,Green_tea,Pink,
                 Hot_pink,Barbie,Dream,Turquoise,Berry,Sky]
# plt.rcParams['axes.prop_cycle'] = plt.cycler(color = updated_colour)

### Mutation functions ###
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
     
### Growth cell updated ###
def growth_(s, table):
    '''
  This function implements growth on subpopulation depending on updated biomass after dynamics, chemical mediated interaction.
  If the size = 2, the number of cells in sub-population will multiply by 2 and new size/2, else the length and number will remain. 
  The new updated sub-population will have new size = size/2
  Input:
         s - the updated biomass after the dynamics calculations, i.e solving coupled ODE's. Type: 1d array.
         table - table after dynamics. Type: pandas table [Parameter,Change,Number,Length].
  Return:
        The updated table with size ('Length') and the number ('Number') of cells for each subpopulation in the table.         
    '''
    number_cell = table.loc[:,'Number']
    length_cell = table.loc[:,'Length']
    biomass = np.reshape(s,(len(number_cell),1))
    new_number = []
    new_size = []   
    for row in range(len(table)): 
        if number_cell[row] == 0:
            new_number.append(number_cell[row]) # else the number of cells remain the same
            new_size.append(length_cell[row]) # old size remains if subpopulation i.e row is extinct (Number=0). To distinguish opener.
        else:
            size = biomass[row][0]/number_cell[row]
            if size >= 2:
                new_number.append(number_cell[row]*2)      
                size = size/2 
                new_size.append(size)
            else:
                new_number.append(number_cell[row]) # else the number of cells remain the same
                new_size.append(size) # old size remains if less than 2
            
    new_number = np.reshape(new_number,(len(new_number),1)) # updated number of cells in a column
    new_size = np.reshape(new_size,(len(new_size),1)) # updated size of each cell after growth
    
    table1 = copy.deepcopy(table)
    table1.loc[:,'Number'] = new_number
    table1.loc[:,'Length'] = new_size
    table_final = table1.fillna(0)
    
    return table_final


### Dynamical rate of change for growth rate and chemical concentration of cell Niehaus et.al ###

### Dynamics of cell ###
def dSCdt(SC, num_spec, r0, K, alpha, beta, rho_plus, rho_minus):
    """
    Author: Alex 
    Parameters:

    SC (array): an array of species and chemical abundances in which species
        are listed before chemicals
    num_spec (int): number of species
    r0 (2d numpy.array): num_spec x 1 array of intrinsic growth rates
    K (2d numpy.array): num_spec x num_chem array of K values
    alpha (2d numpy.array): num_chem x num_spec array of consumption constants
    beta (2d numpy.array): num_chem x num_spec array of production constants
    rho_plus (2d numpy.array): num_spec x num_chem array of positive influences
    rho_minus (2d numpy.array): num_spec x num_chem array of negative influences

    Returns:
    An array of rates of change in which species are listed before chemicals
    """
    S = np.reshape(SC[:num_spec], [num_spec,1])
    C = np.reshape(SC[num_spec:], [len(SC) - num_spec, 1])
    # compute K_star
    K_star = K + C.T
    # compute K_dd
    K_dd = rho_plus * np.reciprocal(K_star)
    # compute lambda
    Lambda = np.matmul(K_dd - rho_minus, C)
    # compute dS/dt
    S_prime = (r0 + Lambda) * S
    # compute K_dag
    C_broadcasted = np.zeros_like(K.T) + C
    K_dag = np.reciprocal(C_broadcasted + K.T) * C_broadcasted
    # compute dC/dt
    C_prime = np.matmul(beta.T - (alpha.T * K_dag), S)
    
    SC_prime = np.vstack((S_prime, C_prime))
    
    return SC_prime

def split_rho(rho):
    """
    Parameters:
    rho (2d numpy.array): rho matrix

    Returns: (rho_plus, rho_minus)

    rho_plus (2d numpy.array): a matrix whose nonzero elements are the positive
        elements of rho
    rho_minus (2d numpy.array): a matrix whose nonzero elements are the negative
        elements of rho
    """
    rho_plus = rho * (rho > 0)
    rho_minus = -rho * (rho < 0)
    return (rho_plus, rho_minus)

def sc_prime_std(t, sc, params):
    """
    defines the derivative in a format more friendly to scipy.integrate.ode
    """
    result = dSCdt(sc,
                 params["num_spec"],
                 params["r0"],
                 params["K"],
                 params["alpha"],
                 params["beta"],
                 params["rho_plus"],
                 params["rho_minus"])
    return np.reshape(result, result.size).tolist()

   
def dynamics(dydt,y0,t,timestep):

    '''
  A function which solves ODE for a timespan 't' and for initial conditions 'y0'
  Input: 
    dydt - the function to be solved by numerical integrator, which has only two variables 't' and 'y0'
    y0 - initial conditions for t=0 to start solving the ODE from somewhere
    t - the duration or timespan to solve the ODE i.e integration range: {t0,t} t0-start time, t-final time
  Output:
    The array of updated dynamical values for species density 'S' and chemical concentration 'C'
    '''
    y = solve_ivp(fun=dydt,
                     t_span=[0, t],
                     y0=y0,
                     t_eval=np.linspace(0,t,int(timestep)))    
    return y 


def dynamics_exact(dydt,y0,t):

    '''
  A function which solves ODE for a timespan 't' and for initial conditions 'y0'. 
  Particularly, for using exact integration method to solve for all dynamical variables i.e all biomass and chemicals
  Input: 
    dydt - the function to be solved by numerical integrator, which has only two variables 't' and 'y0'
    y0 - initial conditions for t=0 to start solving the ODE from somewhere
    t - the duration or timespan to solve the ODE i.e integration range: {t0,t} t0-start time, t-final time
  Output:
    The array of updated dynamical values for species density 'S' and chemical concentration 'C'
    '''

    y = solve_ivp(fun=dydt,
                     t_span=[0, t],
                     y0=y0,rtol=10**-6,atol=10**-10)  
    return y


### Repacking the dense form to seperate parameters and back ###
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

def params_dict_to_matrix(params, param_names): # run once to establish the table

    '''
      Inputs all the arrays in a dictionary and input it as a single large array ordered one after another
      In this case, put all parameters K, r0, alpha, beta, rho_plus and rho_minus into one matrix with columns
      equal to the size of all the columns of individual parameters.
      Input: 
        params - the parameter names in a dictionary, and the input for each element in the dictionary
                 these are the parameter values which can be of form n_s x n_c or vice versa
        param_names - the names of all the parameters in the dictionary
      Output: 
        A single large array containing the same number of columns as the sum of individual parameter columns
    '''
    result = []
    for spec in range(params["num_spec"]):
        spec_params = []
        for param_name in param_names:
            spec_params.append(params[param_name][spec,:])
        result.append(np.concatenate(spec_params))
    return np.array(result);

def params_matrix_to_dict(params_as_matrix, param_names, ncols): # run each time you need dynamics
    '''
      Splits the large matrix or n-dimensional array into their original specific parameter matricies
      Input: 
        params_as_matrix - a single large matrix or n-dimensional array containing all the parameters
        param_names - the names of all the parameters in the dictionary
        n_cols - the number of columns for each individual parameter
      Output: 
        A dictionary which has all the parameter values in their respective small matricie 
    '''
    params = {}
    cutpoints = np.zeros(ncols.size+1).astype(int) 
    cutpoints[1:] = np.cumsum(ncols)
    for i in range(len(param_names)):
        params["num_spec"] = params_as_matrix.shape[0]
        begin_col = cutpoints[i]
        finish_col = cutpoints[i+1]
        param_name = param_names[i]
        param = params_as_matrix[:, begin_col : finish_col]
        params[param_name] = param
   
    return params;

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

def cell_cycle(params1,c0,table1,table2,s_neg,s_pos,t,timestep):
    '''
    Version 1: Large function which does the process initial trial
    Only restricted to 2-species interaction
    '''
    param_names = [i for i in params1 if i != 'num_spec'] # list of all names of the parameters in dictionary
    ncols1 = get_ncols(params1,param_names) # getting the number of columns for each parameter, K, rho etc.
    params_as_matrix = params_dict_to_matrix(params1,param_names) # combining all the columns into one big matrix
    n_c = params1['K'].shape[1]
    n_s = params1["num_spec"]

    max_param = params_as_matrix.shape[1] # numer of columns, all parameters i.e dynamical quantities K, alpha, r0...

    initial_ancestor1 = params_as_matrix[0] # the ancestor parameter values, alpha, K, r0... for species 1 
    initial_ancestor2 = params_as_matrix[1] # the ancestor parameter values, alpha, K, r0... for species 2
    
    mutation1  = mutation(table1,0,max_param,0.001,s_neg,s_pos) # adding mutation to species 1
    mutation2 = mutation(table2,0,max_param,0.001,s_neg,s_pos) # adding mutation to species 2
    
    X1 = sparse_dense(mutation1,max_param,initial_ancestor1)
    X2 = sparse_dense(mutation2,max_param,initial_ancestor2)
    
    # Converts table to biomass 
    biomass1 = biomass(mutation1)
    biomass2 = biomass(mutation2)
    
    biomass_tot = np.concatenate((biomass1,biomass2))
    X_tot = np.concatenate((X1,X2))
    
    dynamic_tot = params_matrix_to_dict(X_tot,param_names,ncols1) 
    
    def dydt(t, sc): 
        '''
   Changing format to be solved by numerical integration. The format needs the ODE to have 2 variables
   Input:
      t - time span to solve the ODE
      sc - initial condition values for all the species and chemicals (array of length n_s + n_c)
    Output: 
      The correct equation format to be solved in solve_ivp, where input is time and initial condition
      '''
        return sc_prime_std(t, sc, dynamic_tot)

    s_label = ['s{}'.format(i+1) for i in range(label_maker(biomass_tot))]
    c_label = ['c{}'.format(i+1) for i in range(n_c)] # this is list comprehension, 

    y0 = np.concatenate((biomass_tot,c0)) # initial conditions for dynamics calculations
    
    # sovle the ODE to get dynamics 
    y = dynamics(dydt,y0,t,timestep);
    t_axis = y.t
    # plot the results to see ODE in action for each timestep 
    # plt.figure()
    # plt.semilogy(t_axis,y.y.T[:,:len(biomass_tot)], '.-',alpha=0.5)
    # plt.xlabel('Timestep')
    # plt.ylabel('Change in biomass')
    # plt.legend(s_label, loc = 'best')
    # plt.show()

    # plt.figure()
    # plt.semilogy(t_axis,y.y.T[:,-n_c:],'.-',alpha=0.5)
    # plt.xlabel('Timestep')
    # plt.ylabel('Change in chemical concentration')
    # plt.legend(c_label, loc = 'best')
    # plt.show()
    
    # The results from dynamics calculations
    result = y.y.T[-1:,:]
    s1 = y.y.T[-1:,:len(mutation1)] 
    s2 = y.y.T[-1:,len(mutation1):-n_c]
    c = y.y.T[-1:,-n_c:]
    c = c[0]
    
    # Implement growth after dynamics, using updated biomass 
    growth1 = growth_(s1,mutation1)
    growth2 = growth_(s2,mutation2)
    # later combine into a single growth
    
    return c, growth1, growth2;


def cell_cycle_list(params1,c0,table,mut_prob,s_neg,s_pos,t,timestep):
    '''
    The complete cell cycle of cell division, mutation and dynamics updated
    Input:
        params1 - the initial conditions for parameters, number of species, K, rho_plus, rho_minus, r0, alpha, beta
        c0 - the initial chemical concentration, later will be updated by dynamics calculation
        table - initial starting species composition: types of species, the number of cells, the length of cells
        t - the time for maturation of cells
        timestep - how many steps to divide the time 't'. If t = 10, timestep = 500, step size is dt = 0.02 (dt = t/timestep)
    Output:
        The updated chemical concentration after species interaction, and the table with updated biomass (length and number)  
    '''
    n_s = params1["num_spec"]
    n_c = params1['K'].shape[1]
    
    param_names = [i for i in params1 if i != 'num_spec'] # list of all names of the parameters in dictionary
    ncols1 = get_ncols(params1,param_names) # getting the number of columns for each parameter, K, rho etc.
    params_as_matrix = params_dict_to_matrix(params1,param_names) # combining all the columns into one big matrix

    max_param = params_as_matrix.shape[1] # numer of columns i.e parameters columns K, r0, alpha, beta, rho_plus, rho_minus
    
    mutation0 = [mutation(table[i],0,max_param-1,mut_prob,s_neg,s_pos) for i in range(len(table))] # adding mutation 

    X = np.concatenate([sparse_dense(mutation0[i],max_param,params_as_matrix[i]) for i in range(n_s)]) # table to matrix form

    biomass0 = np.concatenate([biomass(mutation0[i]) for i in range(n_s)]) # compute biomass = cell number x cell length

    dynamic_tot0 = params_matrix_to_dict(X, param_names, ncols1) # convert big matrix to small matricies for dynamics calculation

    def dydt(t, sc): 
        '''
    Changing format to be solved by numerical integration. The format needs the ODE to have 2 variables
    Input:
      t - time span to solve the ODE
      sc - initial condition values for all the species and chemicals (array of length n_s + n_c)
    Output: 
      The correct equation format to be solved in solve_ivp, where input is time and initial condition
      '''
        return sc_prime_std(t, sc, dynamic_tot0)

    s_label = ['s{}'.format(i+1) for i in range(label_maker(biomass0))]
    c_label = ['c{}'.format(i+1) for i in range(n_c)]

    y0 = np.concatenate((biomass0, c0)) # initial conditions for dynamics calculations

    # sovle the ODE to get dynamics 
    y = dynamics(dydt,y0,t,timestep)
    t_axis = y.t
    
    # plot the results to see ODE in action for each timestep 
    # plt.figure()
    # plt.semilogy(t_axis,y.y.T[:,:len(biomass0)], '.-',alpha=0.5)
    # plt.xlabel('Timestep')
    # plt.ylabel('Change in biomass')
    # plt.legend(s_label, loc = 'best')
    # plt.show()

    # plt.figure()
    # plt.semilogy(t_axis,y.y.T[:,-n_c:],'.-',alpha=0.5)
    # plt.xlabel('Timestep')
    # plt.ylabel('Change in chemical concentration')
    # plt.legend(c_label, loc = 'best')
    # plt.show()

    s = y.y.T[-1:,:-n_c]
    c = y.y.T[-1:,-n_c:]
    c = c[0]
    
    s_cut = cut_species(s,mutation0)
    upd_table = [growth_(s_cut[i],mutation0[i]) for i in range(len(mutation0))]
    return c, upd_table


### Using trapezoidal integration ###

### The functions for chemical state update ###
def dCdt(C, S, K, alpha, beta):
    '''
    ODE representing the rate of change of chemical concentration dC/dt
    Input:
        S - 1d array of species biomass
        C - 1d array of chemical concentration 
        K - the saturation constant, 2d array (num species x num chemicals)
        alpha - rate of release of chemicals, 2d array (num chemicals x num species)
        beta - rate of consumption of chemical, 2d array (num chemicals x num species)
    Return:
        A 1d array of rate of change of chemical concentration
    '''
    S = np.reshape(S, [len(S),1])
    C = np.reshape(C, [len(C), 1])
    # compute K_dag
    C_broadcasted = np.zeros_like(K.T) + C
    K_dag = np.reciprocal(C_broadcasted + K.T) * C_broadcasted
    # compute dC/dt
    C_prime = np.matmul(beta.T - (alpha.T * K_dag), S)    
    return C_prime

def c_prime_std(t, c, params):
    '''
    Defines the derivative in a format more friendly to scipy.integrate.ode
    Input:
        t - the time duration of cell cycle
        c - chemical concentration, 1d array
        params - a dictionary including the updated biomass in dictionary and parameters 
    Output:
        The correct format of rate of change of chemical concentration for integration
    '''
    result = dCdt(c,
                 params['s'],
                 params["K"],
                 params["alpha"],
                 params["beta"])
    return np.reshape(result, result.size).tolist()

### The functions for biomass update ###
def growth_rate_species(C, r0, K, rho_plus, rho_minus):
    '''
    The funciton which calculates the species growth rate based on chemical concentration
    Input:
        C - the array of chemical concentration
        r0 - the array of net growth rate with no chemicals for each species
        K - the saturation constant, 2d-array (num spec x num chem)
        rho_plus - the positive influence of chemicals on growth rate, 2d array (num spec x num chem)
        rho_minus - the negative influence of chemicals on growth rate, 2d array (num spec x num chem)
    Output:
        A 1d array of species growth rate
    '''
    C = np.reshape(C, [len(C), 1])
    # compute K_star
    K_star = K + C.T
    # compute K_dd
    K_dd = rho_plus * np.reciprocal(K_star)
    # compute lambda
    Lambda = np.matmul(K_dd - rho_minus, C)
    # compute growth rate
    growth_rate = (r0 + Lambda)
    
    return growth_rate

def length_update(table,growth_int_exp):
    '''
    Gives the final length of the cell
    Input:
        table - table with initial length of the cells
        growth_int_exp - the growth rate obtained from trapezoid integration
    Output:
        The table with updated length of cells after calculating the biomass dynamics
    '''
    table1 = copy.deepcopy(table)
    table1.loc[:,'Length'] = table1.loc[:,'Length'] * growth_int_exp
    return table1


def cell_divide(table):
    '''
      This function will implement growth depending on length of the cell
      If the length = 2, the number of cells in subpopulation will divide, increasing population size by 2
      The updated subpopulation will have new legth half the original length 
      Input:
          table - table with the updated length of subpopulations after trapezoidal integration
      Return:
          The table with updated cell length ('Length') and the number ('Number') of cells in table          
    '''
    length = np.reshape(np.array(table.loc[:,'Length']),(len(table),1))
    number_cells = np.reshape(np.array(table.loc[:,'Number']),(len(table),1))

    new_number = []
    new_length = []   
    for i in range(len(length)): 
        if length[i]>=2:
            new_number.append(number_cells[i]*2)      
            length[i] = length[i]/2
            new_length.append(length[i])
        else:
            new_number.append(number_cells[i])   
            new_length.append(length[i])
            
    new_number = np.reshape(new_number, (len(new_number),1))
    new_length = np.reshape(new_length, (len(new_length),1))
    
    table1 = copy.deepcopy(table)
    table1.loc[:,'Number'] = new_number
    table1.loc[:,'Length'] = new_length
    table_final = table1.fillna(0)
    return table_final

def update_species_trapezoid(c,t,timestep,params1,table):
    '''
    Updates the biomass of species using trapezoid integration, and chemical concentraion from typical integration
    Input:
        c - the updated chemical concentration form integration, which has shape (timestep, n_c) 
        t - the time for integration of chemicals (integer)
        timestep - the steps of integration, how many points are cut to reach final time 't' (integer)
        params1 - the dictionary with the parameters, specifically r0, K, rho_plus, rho_minus (dictionary)
        table - the table with the updated biomass, usually after mutation to update cell 'Length' 
    Output:
        The table with the updated biomass, which include updated 'number' of cells and 'length' of cells 
    '''
    dt = t/timestep
    growth_species = []
    # Must update this method, instead of looping, think about alternative
    growth_species = np.array([(growth_rate_species(c[i],params1['r0'], params1['K'], 
                        params1['rho_plus'], params1['rho_minus'])).flatten() for i in range(timestep)])
    growth_integrated = scipy.integrate.trapezoid(growth_species, x=None, dx=dt, axis=0) # axis=0 implies time axis
    growth_integrated_exp = np.exp(growth_integrated)

    cut_points = cut_species(growth_integrated_exp,table) # how many subpopulation branching from each species
    length_final = ([length_update(table[i],cut_points[i]) for i in range(len(cut_points))])
    divided_cell = ([cell_divide(length_final[i]) for i in range(len(length_final))])

    return divided_cell

def cell_cycle_list_trapezoid(params1,c0,table,mut_prob,s_neg,s_pos,t,timestep):

    '''
    A complete cell cycle, state update. Initial starting condition at cell division, mutation, dynamics....
    This is the method which uses trapezoidal integration from Li et.al paper
    Input:
        params1 - the initial parameters: number of species, K, rho_plus, rho_minus, r0, alpha, beta, biomass
        c0 - the initial chemical concentration, later will be updated by dynamics calculation
        table - initial starting species composition: types of species, the number of cells, the length of cells
        t - the time for maturation of cells
        timestep - how many steps to reach 't'. If t = 10, timestep = 500, step size is dt = 0.02 (dt = t/timestep)
    Output:
        The updated chemical concentration after species interaction, and table with updated biomass
    '''
    n_c = params1['K'].shape[1]
    n_s = params1["num_spec"]

    param_names = [i for i in params1 if i != 'num_spec'] # list of all names of the parameters in dictionary
    ncols1 = get_ncols(params1,param_names) # getting the number of columns for each parameter, K, rho etc.
    params_as_matrix = params_dict_to_matrix(params1,param_names) # combining all the columns into one big matrix

    max_param = params_as_matrix.shape[1] # numer of columns, all parameters i.e dynamical quantities K, alpha, r0...

    mutation0 = [mutation(table[i],0,max_param-1,mut_prob,s_neg,s_pos) for i in range(len(table))]

    biomass_update = np.concatenate([biomass(mutation0[i]) for i in range(len(mutation0))]) # compute initial biomass  
    biomass_update = np.reshape(biomass_update,(len(biomass_update),1))

    X = np.concatenate([sparse_dense(mutation0[i],max_param,params_as_matrix[i]) for i in range(n_s)])
    dynamic_tot0 = params_matrix_to_dict(X, param_names, ncols1)

    dynamic_tot0['s'] = biomass_update # adding updated biomass to dictionary

    def dcdt(t, c): 
        '''
    Changing format to be solved by numerical integration. The format needs the ODE to have 2 variables
    Input:
      t - time, duration for 1 cycle of cell growth
      c - an array of chemical concentration 
    Output: 
      The correct format to be solved in solve_ivp. Input requires two variables, time and initial condition.
      '''
        return c_prime_std(t, c0, dynamic_tot0)

    c_label = ['c{}'.format(i+1) for i in range(n_c)]

    # sovle the ODE to get dynamics 
    y = dynamics(dcdt,c0,t,timestep)
    t_axis = y.t

    # plot the results to see ODE in action for each timestep 
    # plt.figure()
    # plt.semilogy(t_axis, y.y.T[:,-n_c:], '.-',alpha=0.5)
    # plt.xlabel('Timestep')
    # plt.ylabel('Change in chemical concentration')
    # plt.legend(c_label, loc = 'best')
    # plt.show()

    c1 = y.y.T # chemical state update, all points in timestep

    divided_cell = update_species_trapezoid(c1,t,timestep,dynamic_tot0,mutation0) # updated biomass
    c = c1[-1] # updated chemical state, the point in timestep, required for next loop
    
    return c, divided_cell 

### Dynamics of the HM-community ###

def HM_list(t,SC,params_HM):
    '''
    The growth rate of H and M species 
    Input:
        params_HM - all the parameters used to compute the growth rate, K_HM, alpha_HM, beta_HM, g_max, fp, death
        SC - an array of species biomass followed by chemicals
        t - the time taken for a single cycle 
    Output:
        The growth rate of species H followed by M
    '''
       
    n_s = params_HM['num_spec']
    S = np.reshape(SC[:n_s],[n_s,1])  
    R = SC[-3]
    B = SC[-2]
    P = SC[-1]
    
    g_max = params_HM['g_max']
    g_Hmax = g_max[:,:1]
    g_Mmax = g_max[:,1:2]
    
    K_HM = params_HM['K']
    alpha_HM = params_HM['alpha']
    beta_HM = params_HM['beta']
    fp = params_HM['fp']
    
    death = params_HM['death']
    death_H = death[:,:1]
    death_M = death[:,1:2]
    
    K_HM[K_HM < 1e-9] = 1e-9;
    
    # H species
    K_HR = K_HM[:,:1]
    g_H = np.divide(R,(R + K_HR))
    growth_H = g_H

    # M species 
    R_M = (np.divide(R, K_HM[:,0:1])) # first column always R
    B_M = (np.divide(B, K_HM[:,1:2])) # second column always B

    # To get rid of dividing by zero
    if R_M.all() == 0:
        g_coeff = np.zeros(len(R_M)).reshape(len(R_M),1)
    else:
        g_coeff = np.divide(np.multiply(R_M, B_M),(R_M + B_M))

    g_M = np.multiply(g_coeff,(np.divide(1, R_M + 1) + np.divide(1, B_M + 1)))
    growth_M = np.multiply((1 - fp), g_M)

    c_R = alpha_HM[:,0:1]
    c_B = alpha_HM[:,1:2]

    # To include stochastic death process instead of deterministic, as seen in the previous line
    dHdt = np.multiply((growth_H*g_Hmax),S) 
    dMdt = np.multiply((growth_M*g_Mmax),S)
    dSdt = dHdt + dMdt
    dSdt = dSdt.flatten()

    # Chemicals rate of change 
    dRdt = -sum(np.multiply(np.multiply((g_H*g_Hmax), c_R), S)) \
           -sum(np.multiply(np.multiply((g_M*g_Mmax), c_R), S));
    dBdt = sum(np.multiply((g_H*g_Hmax), S)) \
          -sum(np.multiply(np.multiply((g_M*g_Mmax), S), c_B));
    dPdt = sum(np.multiply(np.multiply((g_M*g_Mmax), fp), S));
    
    dCdt = np.array([dRdt[0],dBdt[0],dPdt[0]]) # The rate of change chemicals with convention R, B, P
    
    dSCdt = np.concatenate((dSdt,dCdt)) # The change in biomass followed by rate of change of chemicals

    return dSCdt

def death_cell(table,death_rate,timestep):
    '''
    Cell death: For each cell, a random number is chosen with lim[0,1]
    If the random value is less than death_rate*dt, then this cell dies
    Input:
        table_HM -  table which has the Number of cells in each population, and their subpopulation
        timestep - the gridsize or step size of a single step to reach full cycle length
        death_rate - accourding to the species i.e H or M 
    Output: The updated table with death of cells

    '''
    table1 = copy.deepcopy(table)
    number_ = number_cells(table1)
    for n in range(len(table1)):
        death_prob = np.random.rand(int(number_[n]))
        for i in death_prob:
            if i < death_rate*timestep:
                table1.loc[n:n,'Number'] = table1.loc[n:n+1,'Number'] - 1  
    return table1

### Fast binomial method to avoid assigning random value to each cell ###
def normcond(N,p):
    ### Author: Caroline ###
    p[p == 0] = 1e-9;  # avoid division errors
    return np.logical_and((N > 9 * p / (1-p)), (N > 9 * (1-p) / p));

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

def death_cell_fast(table,death_rate,timestep):
    '''
    Initially, the number of cells that will die are picked out using a fast binomial distribution
    This avoids assigning random numbers to each cell, using fast binomial method 
    Cell death: For each cell, a random number is chosen with lim[0,1]
    If the random value is less than death_rate*dt, then this cell dies
    Input:
        table_HM - table with subpopulation. Type: pandas table [Parameter,Change,Number,Length]
        timestep - the length of the timestep at the end, i.e time at the end of the cycle. Type: float
        death_rate - accourding to the species i.e H or M. Type: float
    Output: The updated table with death of cells.
    '''
    table1 = copy.deepcopy(table)
    num_death = fastbinorv(np.array(table1['Number']),death_rate*timestep)
    table1['Number'] = table1['Number'] - num_death
    
    return table1

def death_table(params1,table,t):
    '''
    Function which implements death accourding the species unique death rates 
    Input: 
        params1 - the parameter list which contains the rate of death. Type: dictinoary
        table - the table with species. Type: pandas table 
        t - the length of the timestep at the end, i.e time at the end of the cycle. Type: float
    Output:
        The updated table with species after stochastic death.
    '''
    
    death = params1['death']
    death_HM = [death[:,i][i] for i in range(len(death))]
    deathHM = [death_cell_fast(table[i],death_HM[i], t) for i in range(len(table))]

    return deathHM

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

def cell_cycle_list_HM(params1,c0,table,mut_param,t):
    '''
    The complete cell cycle of cell division, mutation and calculating the dynamics. Where mutation occurs for each timestep. 
    Cells will mutate regardless cell division taking place for each timestep "t".
    Input:
        params1 - the initial conditions for parameters, number of species, K, rho_plus, rho_minus, r0, alpha, beta
        c0 - the initial chemical concentration, later will be updated by dynamics calculation
        table - initial starting species composition: types of species, the number of cells, the length of cells
        t - the time for maturation of cells
    Output:
        The updated chemical concentration after species interaction, and the table with updated biomass (length and number)  
    '''
    n_c = len(c0) 
    
    param_names = [i for i in params1 if i != 'num_spec'] # list of all names of the parameters in dictionary
    ncols1 = get_ncols(params1,param_names) # getting the number of columns for each parameter, K, rho etc.
    params_as_matrix = params_dict_to_matrix(params1,param_names) # combining all the columns into one big matrix
    
    max_param = params_as_matrix.shape[1] # numer of columns i.e parameters columns K, r0, alpha, beta, rho_plus, rho_minus
    
    matrix_form = [sparse_dense(table[i],max_param,params_as_matrix[i]) for i in range(len(table))] # table to matrix form
    mutation0 = [mutation_HM_step(table[i],params1,matrix_form[i],mut_param,0,1) for i in range(len(table))] # adding mutation 

    X = np.concatenate([sparse_dense(mutation0[i],max_param,params_as_matrix[i]) for i in range(len(mutation0))]) # table to matrix form
    biomass0 = np.concatenate([biomass(mutation0[i]) for i in range(len(mutation0))]) # compute biomass = cell number x cell length
   
    biomass_update = np.reshape(biomass0,(len(biomass0),1))
    
    dynamic_tot0 = params_matrix_to_dict(X, param_names, ncols1) # convert big matrix to small matricies for dynamics calculation
    dynamic_tot0['s'] = biomass_update # adding updated biomass to dictionary
    
    def dydt(t, sc): 
        '''
    Changing format to be solved by numerical integration. The format needs the ODE to have 2 variables
    Input:
      t - time span to solve the ODE
      sc - initial condition values for all the species and chemicals (array of length n_s + n_c)
    Output: 
      The correct equation format to be solved in solve_ivp, where input is time and initial condition
      '''
        return HM_list(t,sc,dynamic_tot0) 
    
    y0 = np.concatenate((biomass0, c0)) # initial conditions for dynamics calculations

    # sovle the ODE to get dynamics 
    y = dynamics_exact(dydt, y0, t)
    
    s = y.y.T[-1:,:-n_c][0] # Biomass
    c = y.y.T[-1:,-n_c:][0] # Chemicals

    s_cut = cut_species(s,mutation0)
    upd_table = [growth_(s_cut[i],mutation0[i]) for i in range(len(mutation0))]
    final_table =  death_table(params1,upd_table,t)

    return c, final_table, s 

### Modified functions: mutation only occurs if the cells in the subpop has divided (previously, mutation occured for each timestep).

def growth__(s, table):
    '''
  This function implements growth on subpopulation depending on updated biomass after dynamics, chemical mediated interaction.
  If the size = 2, the number of cells in sub-population will multiply by 2 and new size/2, else the length and number will remain. 
  The new updated sub-population will have new size = size/2. Additionally, the divided row index.
  Input:
         s - the updated biomass after the dynamics calculations, i.e solving coupled ODE's. Type: 1d array.
         table - table after dynamics. Type: pandas table [Parameter,Change,Number,Length].
  Return:
        The updated table with size ('Length') and the number ('Number') of cells for each subpopulation in the table. Divided row index.        
    '''
    number_cell = table.loc[:,'Number']
    length_cell = table.loc[:,'Length']
    biomass = np.reshape(s,(len(number_cell),1))
    new_number = []
    new_size = []  
    inds =  []
    for row in range(len(table)): 
        if number_cell[row] == 0:
            new_number.append(number_cell[row]) # else the number of cells remain the same
            new_size.append(length_cell[row]) # old size remains if subpopulation i.e row is extinct (Number=0). To distinguish opener.
        else:
            size = biomass[row][0]/number_cell[row]
            if size >= 2:
                new_number.append(number_cell[row]*2)      
                size = size/2 
                new_size.append(size)
                inds.append(table.loc[row:row].index.tolist())
            else:
                new_number.append(number_cell[row]) # else the number of cells remain the same
                new_size.append(size) # old size remains if less than 2
            
    new_number = np.reshape(new_number,(len(new_number),1)) # updated number of cells in a column
    new_size = np.reshape(new_size,(len(new_size),1)) # updated size of each cell after growth
    
    table1 = copy.deepcopy(table)
    table1['Number'] = new_number
    table1['Length'] = new_size
    table_final = table1.fillna(0)

    return table_final, inds


def mutation_HM(table,matrix_form,ind,mut_param,min_param,max_param):
    '''
  A function which adds mutation to an ancestor. The only parameter allowed to mutate if fp, cost function.
  There is a chance of mutation occuring for each subpopulation equally, depending on Number of mutants i.e "Number"
  Input: 
      table - the starting table with ancestor, with columns: parameter, change and number of cells 
      matrix_form - the dense matrix form of the parameters. Ensure the matrix is not concatenated each species must be seperate array
      ind - the index to represent the rows or subpop which has divided
      min_param - the 1s parameter index
      max_param - the last parameter index, total number of parameters
  Return:
      A table with added mutations to random parameters, where the divided subpop is allowed to mutate.
      Branching from ancestor to different subpopulation
    '''
    s_neg = mut_param['s_neg']
    s_pos = mut_param['s_pos'] 
    mutation_p = mut_param['mutation_p'] 

    # If there was no cell division and no row/subpop will mutate
    for d in reversed(ind):
        if d == []:
            pass
        else:
            row = matrix_form[d]
            fp_upper = 1.0
            fp = row[0][0]
            num_mutation = fastbinorv(np.array(table.loc[d]['Number']), mutation_p)
            update_index = np.array(table.loc[d[0]:].index.tolist())
            for i in range(len(num_mutation)):
                for each in range(num_mutation[i]):
                    parameter, change = rand_mut_param(min_param,max_param,mutation_distribution2(1,s_neg,s_pos))
                    change = ((change + 1) * fp)  
                    change[change > fp_upper] = fp_upper # Upper bound for the maximum allowed value for parameter fp
                    change = change - fp # To compensate for cumsum() takeaway the ancestor fp
                    if change != 0:
                        table = ancestor_min(table, update_index[i],1)
                        table = add_subpop(table,update_index[i],parameter,change,1)
                        update_index[i+1:] +=2 
    return table



def mutation_HM_null(table,matrix_form,ind,mut_param,min_param,max_param):
    '''
  A function which adds mutation to an ancestor. The only parameter allowed to mutate if fp, cost function.
  There is a chance of mutation occuring for each subpopulation equally, depending on Number of mutants i.e "Number"
  Input: 
      table - the starting table with ancestor, with columns: parameter, change and number of cells 
      matrix_form - the dense matrix form of the parameters. Ensure the matrix is not concatenated each species must be seperate array
      ind - the index to represent the rows or subpop which has divided
      min_param - the 1s parameter index
      max_param - the last parameter index, total number of parameters
  Return:
      A table with added mutations to random parameters, where the divided subpop is allowed to mutate.
      Branching from ancestor to different subpopulation
    '''
    s_neg = mut_param['s_neg']
    s_pos = mut_param['s_pos'] 
    mutation_p = mut_param['mutation_p'] 
    null_prob = mut_param['null_prob'] 

    # If there was no cell division and no row/subpop will mutate
    for d in reversed(ind):
        if d == []:
            pass
        else:
            fp_upper = 1.0
            row = matrix_form[d]
            fp = row[0][0]
            # print('subpop',d)
            num_mutation = fastbinorv(np.array(table.loc[d]['Number']),mutation_p)
            # print('total mut',num_mutation)
            null_mut = fastbinorv(num_mutation,null_prob) # the number of total mutants with null mutation 
            active_mut = num_mutation - null_mut
            # print('active', active_mut)
            update_index = np.array(table.loc[d[0]:].index.tolist())
            for i in range(len(active_mut)):
                for each in range(active_mut[i]):
                    parameter, change = rand_mut_param(0,1,mutation_distribution2(1,s_neg,s_pos))
                    # print('fp ancestor row {}'.format(d),fp)
                    change = ((change + 1) * fp)  
                    # print('the value of the new mutation after cumsum() active',change)
                    change[change > fp_upper] = fp_upper # Upper bound for the maximum allowed value for parameter fp
                    change = change - fp # To compensate for cumsum() takeaway the ancestor fp
                    if change != 0:
                        # print('final fp',change)
                        table = ancestor_min(table, update_index[i],1)
                        table = add_subpop(table,update_index[i],parameter,change,1)
                        update_index[i+1:] +=2 
            if fp != 0:
                if null_mut != 0:
                    # print('null',null_mut)
                    parameter, change = rand_mut_param(0,1,null_distribution())
                    # print('fp ancestor row {}'.format(d),fp)
                    change = ((change + 1) * fp) 
                    # print('the value of the new mutation after cumsum() null',change)
                    change = change - fp # To compensate for cumsum() takeaway the ancestor fp
                    # print('fp final',change)
                    table = ancestor_min(table,d[0],null_mut)
                    table =  add_subpop(table,d[0], parameter, change,null_mut)
    return table
        

def cell_cycle_HM(table,params1,c0,mut_param,t):
    '''
    The complete cell cycle in the steps: dynamcs, death, cell division, and mutation. Where mutation occurs only if cells have divided.
    Input:
        table - initial starting species composition: types of species, the number of cells, the length of cells
        Type: pandas table ['Parameter','Change','Number','Length']
        params1 - the initial conditions for parameters, number of species, K, rho_plus, rho_minus, r0, alpha, beta
        c0 - the initial chemical concentration, later will be updated by dynamics calculation      
        mut_param - the parameters of mutation [s_neg,s_pos,mutation_p,null_prob]. Type: dictionary
        t - the time for maturation of cells
    Output:
        The updated chemical concentration after species interaction, and the table with updated biomass ('Length' and 'Number')  
    '''
    param_names = [i for i in params1 if i != 'num_spec'] # list of all names of the parameters in dictionary
    ncols1 = get_ncols(params1,param_names) # getting the number of columns for each parameter, K, rho etc.
    params_as_matrix = params_dict_to_matrix(params1,param_names) # combining all the columns into one big matrix

    max_param = params_as_matrix.shape[1] # numer of columns i.e parameters columns K, r0, alpha, beta, rho_plus, rho_minus

    X = np.concatenate([sparse_dense(table[i],max_param,params_as_matrix[i]) for i in range(len(table))]) # table to matrix form

    biomass0 = np.concatenate([biomass(table[i]) for i in range(len(table))]) # compute biomass = cell number x cell length
    biomass_update = np.reshape(biomass0,(len(biomass0),1))

    dynamic_tot0 = params_matrix_to_dict(X, param_names, ncols1) # convert big matrix to small matricies for dynamics calculation
    dynamic_tot0['s'] = biomass_update

    y0 = np.concatenate((biomass0, c0)) # initial conditions for dynamics calculations

    def dydt(t, sc): 
        '''
    Changing format to be solved by numerical integration. The format needs the ODE to have 2 variables
    Input:
        t - time span to solve the ODE
        sc - initial condition values for all the species and chemicals (array of length n_s + n_c)
    Output: 
        The correct equation format to be solved in solve_ivp, where input is time and initial condition
        '''
        return HM_list(t,sc,dynamic_tot0) 

    y = dynamics_exact(dydt, y0, t)
    n_c = len(c0) 
    s = y.y.T[-1:,:-n_c][0] # Biomass final
    c = y.y.T[-1:,-n_c:][0] # Chemicals final

    # Stochastic death at the end of cycle 
    table_upd = death_table(params1,table,t)

    s_cut = cut_species(s,table_upd) # cutting combined species array to distinguish biomass of each species
    division = [growth__(s_cut[i],table_upd[i]) for i in range(len(table_upd))]

    ind = [division[i][1] for i in range(len(division))] # the indexes of the divided cells for all species 
    table_divided = [division[i][0] for i in range(len(division))] # the table after cell division accourding to length
    matrix_form = [sparse_dense(table_divided[i],max_param,params_as_matrix[i]) for i in range(len(table_divided))] # to extract param
    new_table = [mutation_HM_null(table_divided[i],matrix_form[i],ind[i],mut_param,0,1) for i in range(len(table_divided))]

    return new_table,c,s


### These functions chooses Adult communities and reproduces them to Newborn communities ###
def sort_wells(list_,loc):
    '''
    A function which will sort wells accourding to the community function. For the HM community, this is Product
    Input:
        list_ - a list which has the simulation results of the wells/test-tubes. Type: [length, biomass, Product]
        loc - the index of where the community function is. In this case, it is the location of "Product"
    Output:
        The list sorted, where top performing communities are in the top i.e sorted by the highest Product on top
    '''
    new_list = copy.deepcopy(list_)
    new_list.sort(key = lambda x: x[loc], reverse = True) # This sorts the list of saved arrays acourding to "Product"
    
    return new_list


# Make this a function 
def reproduce(S,BM_target,max_newborn):
    '''
    A function which reproduces the list of indices which are the Newborn partitions based on selection.
    Input:
        S - biomass of the chosen Adult community
        BM_target - target biomass inside each partition 
        wells_filled - starting with 0, the partitions will be filled with cells
        max_newborn - the number of Newborn partitions from each chosen Adult, based on selection strategy
    Output:
        The list of partitions to be filled with Newborn communities selected to mature for next cycle 
    '''
    wells_filled = 0 # start with zer0 filled wells 
    
    tot_biom = sum(S) # the total biomass of the community of a single chosen Adult well
    nD = np.floor(tot_biom/BM_target) # Choosing the number or dilution factor to get the desired target biomass 
    target_indices = np.arange(wells_filled, wells_filled + nD) # starting with 0 well to the maximum possible wells 
    length_current = len(target_indices) # How many wells does the current chosen Adult community reproduce
    # the maximum should be maximum newborns per adult community 
    if length_current > max_newborn:
        target_indices = np.arange(wells_filled, wells_filled + max_newborn)
    return target_indices;


def fill_partition(adult_chosen,BM_target):
    '''
    The function which fills the partitions with on average 100 biomass (need to divide by dilution factor)
    Each partition has a probability of 1% receiving cells from the chosen Adult communities
    It can be viewed as cell sorting, i.e chosen Adult communities are sorted to Newborn communities
    Input:
        adult_chosen - the saved list containing results with table as 1st entry of the chosen Adult community
        wells - the number of wells or partitions to fill with Newborns from one chosen Adult community
        BM_target - target biomass, number of cells to be input in each well for the next round
    Output:
        The multinomial distribution of the chosen adult community aiming tov reach the target biomass (BM_target)
        Where each subpopulation is distributed to the specified number of partitions (wells)
    '''
    # The number of cells in each subpopulation of the chosen adults to reproduce
    
    number_ = [number_cells(adult_chosen[0][i]) for i in range(len(adult_chosen[0]))] # number of cells in subpop
    S = adult_chosen[1] # biomass of chosen Adult community
    tot_biom = sum(S) # total biomass of the chosen Adult community, i.e sum of all subpop
    nD = np.floor(tot_biom/BM_target) #
    filled_well_newborn = []
    for n in range(len(number_)): 
        well_prob = np.ones(int(nD))/nD
        fill_well_newborn = [np.random.multinomial(number_[n][i],well_prob) for i in range(len(number_[n]))]
        filled_well_newborn.append(fill_well_newborn)
    return filled_well_newborn

def pipette(results, top_adults, BM_target, max_newborn):
    '''
    The function sorts Adult communities accourding to Product.
    Selects highest performing Adults, then reproduces them into Newborn communities via seeding.
    Input:
        results - a list with the simulation results of the wells/test-tubes. Type: [length, biomass, Product]
        top_adults - the integer of how many Adult communities will be reproduced. For top 5, we choose 5 Adults.
        BM_target - target biomass inside each partition, number of cells in each well for the next round
        max_newborn - the number of Newborn partitions, based on selection strategy
    Output:
        The Newborn communities i.e tables for the next cycle.
    '''
    
    highest_product_communities = sort_wells(results,2) # where 2 represents the Products 
    best = highest_product_communities[:top_adults]
    
    filled_wells_all = []
    for i in range(len(best)):
        filling_wells = fill_partition(best[i],BM_target)
        filled_wells_all.append(filling_wells)
     
    # Now using the new number of cells in each partition, update the table for each partition  
    all_newborn = []
    for i in range(len(filled_wells_all)):
        S = best[i][1] # the biomass of chosen Adult community 1
        target_indices = reproduce(S,BM_target,max_newborn) # 1d array: empty partitions to fill with Adults
        distribute_cells = filled_wells_all[i]
        table = best[i][0]
        lol = []
        for p in target_indices:
            subpop = []
            for n in range(len(table)):
                row = []
                for i in range(len(table[n])):
                    new = number_cell_update(table[n].iloc[i:i+1],distribute_cells[n][i:i+1][0][p])
                    row.append(new)
                row = pd.concat(row)
                subpop.append(row)
            subpop = [delete_rows(subpop[i]) for i in range(len(subpop))]
            lol.append(subpop)
        all_newborn.append(lol)
    flattened = [_ for lists in all_newborn for _ in lists]
   # print(flattened)

    return flattened

def mature(params_HM,c0,table_HM,mut_param,step,t):
    '''
    Function which simulates a maturation cycle of a single Newborn community, i.e one well
    Input:
        params_HM - the parameters list. Type: a dictionary
        c0 - initial chemical concentration. Type: list [Resource, Byproduct, Product]
        table_HM - initial starting table with Newborns. Type: table with columns: Parameter, Change, Number, Length
        mut_param - the parameters of mutation [s_neg,s_pos,mutation_p,null_prob]. Type: dictionary
        step - the step size to solve dynamics. Type: int
        t - maturation time taken for Newborns to become Adult. Type: int
    Output:
        Adult communities produced from Newborns after a single maturation cycle. Type 
    '''
    loop = int(t/step)
    for i in range(loop):
        if i == 0:
            new_table,c,s = cell_cycle_HM(table_HM,params_HM,c0,mut_param,step)
            new_table = [delete_rows(new_table[i]) for i in range(len(new_table))]
        else:
            new_table,c,s = cell_cycle_HM(new_table,params_HM,c,mut_param,step)
            new_table = [delete_rows(new_table[i]) for i in range(len(new_table))]
    results = [new_table,s,c[-1]]
    #print(results)

    return results

def community_selection_(C, t, step,mut_param,params_HM,c0,BM_target,Newborns,top_adults,max_newborn,parallel,file_name):
    '''
    This function simulates consecutive cycles using the Newborns produced from chosen Adult communities.
    The newbowns are produced by pipetting. Here, top 5 is used as the selection strategy
    Input:
        C - Number of cycles
        t - Total maturation time 
        step - the timstep to reach total time "t". Type: floa
        mut_param - the parameters of mutation [s_neg,s_pos,mutation_p,null_prob]. Type: dictionary
        params_HM - the parameters input in a dictionary. Type: a dictionary
        BM_target - Target biomass of each well in each cycle
        Newborns - the table which contains the Newborn species, with each species as a table in list. Type: list 
        top_adults - the integer of how many Adult communities will be reproduced. For top 5, we choose 5 Adults.  
        max_newborn - Each Adult community will contribute to this number of wells  
        parallel - specify to run the simulation on local computer (False) or perform simulation on cluster (True). Type: Boolean
        file_name - the name of the file in string form. Type: string 
    Output: 
        The final Adult community after "C" number of cycles
    '''
    for j in range(C):
        results_all1 = []
        if (parallel):
            from mpi4py.futures import MPIPoolExecutor;
            from itertools import repeat;
            with MPIPoolExecutor() as executor:
                Adults = executor.map(mature,repeat(params_HM),repeat(c0), Newborns,
                                                        repeat(mut_param),repeat(step),repeat(t))
            results_all1 = [i for i in Adults]
            Newborns = pipette(results_all1, top_adults, BM_target, max_newborn)
            result_save(f'{file_name}', j, results_all1)
        else:
            for n in range(len(Newborns)): 
                Adults = mature(params_HM,c0,Newborns[n],mut_param,step,t)
                results_all1.append(Adults)
            Newborns = pipette(results_all1, top_adults, BM_target, max_newborn)
            result_save(f'{file_name}', j, results_all1)

    return results_all1

### Community selection for using mpi4py ### 


def community_(C,t, step,mut_param,params_HM,c0,Newborns,parallel,file_name):
    '''
    This function simulates consecutive cycles using the Newborns produced from chosen Adult communities.
    The newbowns are produced by pipetting. Here, top 5 is used as the selection strategy
    Input:
        C - the number of the cycle, for example running 10th cycle. Type: int
        t - Total maturation time 
        step - the timstep to reach total time "t". Type: float
        mut_param - the parameters of mutation [s_neg,s_pos,mutation_p,null_prob]. Type: dictionary
        params_HM - the parameters input in a dictionary. Type: a dictionary
        c0 - the initial concentration of chemicals. Type: 1d array
        Newborns - the table which contains the Newborn species, with each species as a table in list. Type: list of pandas table
        parallel - specify to run the simulation on local computer (False) or perform simulation on cluster (True). Type: Boolean
        file_name - the name of the file in string form. Type: string 
    Output: 
        The final Adult community after "C" number of cycles
    '''
    if __name__ == '__main__': 
        if (parallel):
            from mpi4py.futures import MPIPoolExecutor;
            from itertools import repeat;
            with MPIPoolExecutor() as executor:
                Adults = executor.map(mature,repeat(params_HM),repeat(c0), Newborns,
                                                        repeat(mut_param),repeat(step),repeat(t), unordered=True)
            adult_all = [i for i in Adults]
        else:
            adult_all = [0.] * len(Newborns)
            for i in range(len(Newborns)): 
                adult_all[i] = mature(params_HM,c0,Newborns[i],mut_param,step,t)
        results_all1 = [[0.]] * len(Newborns)
        for i in range(len(Newborns)):
            results_all1[i] = adult_all[i]
        result_save(f'{file_name}', C, results_all1)
        
        return results_all1 


def community_selection(C,t,step,mut_param,params_HM,c0,BM_target,Newborns,top_adults_num,max_newborn,parallel,file_name):
    '''
    This function simulates consecutive cycles using the Newborns produced from chosen Adult communities.
    The newbowns are produced by pipetting. Here, top 5 is used as the selection strategy
    Input:
        C - Number of cycles
        t - Total maturation time 
        step - the timstep to reach total time "t". Type: floa
        mut_param - the parameters of mutation [s_neg,s_pos,mutation_p,null_prob]. Type: dictionary
        params_HM - the parameters input in a dictionary. Type: a dictionary
        c0 - the initial concentration of chemicals. Type: 1d array
        BM_target - Target biomass of each well in each cycle
        Newborns - the table which contains the Newborn species, with each species as a table in list. Type: list of pandas table
        top_adults - the integer of how many Adult communities will be reproduced. For top 5, we choose 5 Adults.  
        max_newborn - Each Adult community will contribute to this number of wells  
        parallel - specify to run the simulation on local computer (False) or perform simulation on cluster (True). Type: Boolean
        file_name - the name of the file in string form. Type: string 
    Output: 
        The final Adult community after "C" number of cycles
    '''
    for j in range(C):
        results_all1 = community_(j, t, step,mut_param,params_HM,c0,Newborns,parallel,file_name)
        Newborns = pipette(results_all1, top_adults_num, BM_target, max_newborn)
    return Newborns 

### End of community selection for using mpi4py ###


def cut_extras(table_HM):
    '''
    Many subpopulations with zero Number of cells which are saved after pipetting i.e multinomial distribution
    Input:
        table_HM - the table after pipetting, these are Newborns 
    Output:
        The updated table with the subpopulations with zero entries removed
    '''
    # checks if above and below rows have the same entry and deletes it
    
    new_table = table_HM[table_HM['Number'].ne(table_HM['Number'].shift())] 
    new_table = new_table.reset_index(drop = True)
    
    return new_table


def remove_nest(list_):
    '''
    Function to remove nested list inside a list. For example [[p],[m]] to [p, m]. Even [[[p],[m]], [[n],[l]]] to [p,m,n,l]
    Input:
        list_ - the list which has nested listed inside the list
    Output:
        A list which only has one outer list, no nested lists inside this list.
        For output: list(remove_nest(list_)). The list() suggests to put all elements in list_ inside a list i.e [list_[i],list_[j]...]
    '''
    if isinstance(list_,list):
        for _ in list_:
            yield from remove_nest(_)
    else:
        yield list_
    return list_

### Functions to save the results from the simulation as a pickle file and load the files after they exist ###
def result_save(name, i, results):
    '''
    A function which saves the results from simulation. For example Adult data, which contains [table, biomass, Product]
    Input:
        name - the designated name of the file. The input should be in terms of string, i.e 'name'
        i - the cycle index. In the form of integer 
        results - the simulation results i.e the variable which contains all Adult from a single cycle
    Output:
        A folder with the name "name" which contains the results from the simulation
    '''
    
    file_ = open(f'{name}_{i}.pck', 'wb') # wb means opening a new file in the folder of the simulation
    pickle.dump(results, file_)
    file_.close()


def open_file(name_file, i):
    '''
    Function which opens a file saved using pickle.
    Input:
        name_file - the name of the file with the simulation results. The input must be a string 'anythin'
        i - index of the file to open, i.e which cycle index. Type: int 
    Output:
        The simulation results decoded from the saved pickle file. Typically Adult data [table, biomass, Product] after each cycle.
    '''

    with open(f'{name_file}_{i}.pck','rb') as _:
        opening = pickle.load(_)
    return opening 

### Finish of saving pcikle files and loading files ###


def divide_zero(s,N):
    '''
   A function to get rid of divide zero errors. Here specifically, get rid of error to calculate length when population size N = 0. 
    Input:
        s - the numerator, this is biomass in the case of calculating length
        N - the denominator, this will be the number of cells 
    Output:
        The divided result with no divide zero error. Function must be used for absolutely necessary case as this gets rid of error.
    '''
    # Similarly to the below, [N and s/N or 0] works too
    return s/N if N else 0


### Functions to remove extinct subpopulations ###

def find_opener(table):
    '''
    This will find an opener row. An extinct subpopulation with Number=0.
    Input:
        table - Pandas table where rows are subpopulation and columns are [Parameter,Change,Number,Length]
    Output:
        The index of the extinct opener rows, i.e the extinct subpopulations
    '''
    table = table.loc[1:] # to ensure ancestor will not be chosen to be removed, ancestors will remain even if extinct.
    index_open = table.loc[(table['Number'] == 0) & (table['Length'] != 0)] # Extinct subpopulation
    index_ = index_open.index.tolist() # Location of the extinct subpop 
    
    return index_

def find_closeout(table, opener):
    '''
    This will find an closeout lines for the extinct population. A closeout line with Number=Length=0
    Input:
        table - Pandas table where rows are subpopulation and columns are [Parameter,Change,Number,Length]
        opener - the location of the opener row, i.e subpopulation row number 
    Output:
        The index of the extinct subpopulation closeout, the first instance where the condition is met
    ''' 
    change = table.loc[opener]['Change']
    table_opener = table.loc[opener:] # to ensure we are not going back prior the creation of the extinct subpop
    closeout = np.where((table_opener['Change'] == -change) & (table_opener['Length'] == 0) & (table_opener['Number'] == 0))[0][0]
    closeout_index = opener + closeout # index of the closeout line 
    
    return closeout_index

def dead_branch(table,opener,closeout):

    '''
    A dead row with zero cells. This could also be closeout line. However, this aims to cut branches not opener followed by closeout.
    Input:
        table - A pandas table with the subpopulations
        opener - index of the opener row, where subpop are dead
        closeout - index of the corresponding closeout of the opener
    Output: 
    True if the inside of subpop are all dead
    ''' 
    if opener != closeout-1:
        return (table.loc[opener:closeout]['Number'] == 0).all()

def delete_rows(table):
  '''
  A function which gets rid of extinct subpopulation specifically for the memory efficient architechture
  Input:
    table - table after cycle. Type: pandas table [Parameter,Change,Number,Length]
  Output:
    The updated table without extinct subpopulations. 
  '''

  indexes = []
  opener = find_opener(table) # Searching for extinct subpopulation rows, where Number=0
  for s in opener:
      closing = find_closeout(table,s)  # the closeout line corresponding to extinct subpop s
      # Looking for dead branchm where between opener row and closeout row, there are extinct subpopulations
      if dead_branch(table,s,closing) == True:
          index_drop = np.arange(s,closing+1).tolist()
          indexes.append(index_drop)
      # if immediatly below, get rid of closeout and dead subpop
      if s == closing-1:
          index_drop = [s, closing]
          indexes.append(index_drop)
  remove = (list(remove_nest(indexes))) # set() records unique indicie once. Conversly, twice indicie does not break code
  remove_row = table.drop(index = remove) 
  update_index = remove_row.reset_index(drop = True)

  return update_index


### Author :Alex Yuan ###
def found_closer(table, row, current_opener_values):
  """returns True if `row` is the closer of a dead branch

  Args:
    table (Pandas Dataframe)
    row (integer): current row
    current_opener_values (float): value of opener row of a possibly dead branch, or -np.inf
  """
  if table.iloc[row]['Number'] != 0:
    return False # not a closeout line
  if table.iloc[row]['Length'] != 0:
    return False # not a closeout line
  if len(current_opener_values) < 1:
    return False # there is no opener
  if table.iloc[row]['Change'] != -current_opener_values[-1]:
    return False # does not correspond to the opener line
  return True

def set_new_opener(table, row):
  """returns True if `row` is possibly the opener of a dead branch

  Args:
    table (Pandas Dataframe)
    row (integer): current row
  """
  if table.iloc[row]['Number'] != 0:
    return False # not dead
  if table.iloc[row]['Length'] == 0:
    return False # not an opener line
  return True

def remove_branch(table):
    if table.shape[0] == 1:
        return table
    else:
        dead_branches = []
        current_opener_values = []
        current_opener_rows = []
        for i in range(table.shape[0]): # loop over table
            if found_closer(table, i, current_opener_values):
                # we have found the closeout line of a dead branch
                dead_branches.append({'opener' : current_opener_rows[-1], 'closer' : i}) # add to list of dead branches
                current_opener_values = current_opener_values[:-1] # remove latest opener
                current_opener_rows = current_opener_rows[:-1] # remove latest opener
            if table.iloc[i]['Number'] != 0 and table.iloc[i]['Length'] != 0:
                # the current branch is not dead
                current_opener_values = [] # remove all openers
                current_opener_rows = [] # remove all openers
            if set_new_opener(table, i):
                # we are possibly at the opener row of a dead branch
                current_opener_values.append(table.iloc[i]['Change']) # set opener value
                current_opener_rows.append(i) # set opener row

        # collect the rows marked for removal
        idx_for_removal = []
        for branch in dead_branches:
            idx_for_removal.append(np.arange(branch['opener'], branch['closer']+1))
        idx_for_removal = np.concatenate(idx_for_removal)

        # remove the dead branches
        table_clean = table.drop(idx_for_removal)
        table_final = table_clean.reset_index(drop = True)
    return table_final


def find_max_index(list_,order):
    '''
    Function which finds the index of the largest element in a list
    Input:
        list_ - the list to look for the largest value
        order - the index of the sorted list. Where order=0 means the largest element in list_, 1 is second largest.
    Output:
        The index of the location of the chosen element in the array. Mostly to find the index of the largest element.
    '''
    return [i for i, id in enumerate(list_) if id==(list(reversed(np.sort(list_))))[order]] 
