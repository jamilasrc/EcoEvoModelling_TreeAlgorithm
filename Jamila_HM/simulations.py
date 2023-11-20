# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 16:18:56 2023

@author: jamil
"""

from tree_algorithm import *
from evolution_functions import *

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

