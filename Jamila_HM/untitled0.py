# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 16:44:27 2023

@author: jamil
"""

def cell_cycle_HM_chunks(sprs_mat_arry,params1,c0,mut_param,t,chunk):
    
    '''
    The complete cell cycle in the steps: dynamcs, death, cell division, and mutation. Where mutation occurs only if cells have divided.
    Input:
        sprs_mat_dict - dictionary of sparsity matrices for each species
        Type: pandas table ['Parameter','Change','Number','Length']
        params1 - the initial conditions for parameters, number of species, K, rho_plus, rho_minus, r0, alpha, beta
        c0 - the initial chemical concentration, later will be updated by dynamics calculation      
        mut_param - the parameters of mutation [s_neg,s_pos,mutation_p,null_prob]. Type: dictionary
        t - the time for maturation of cells
        chunk - the size of each chunk of data to be converted from sparse to dense. Aka, the no. of species sparse matrices
            to be converted. 
    Output:
        The updated chemical concentration after species interaction, and the table with updated biomass ('Length' and 'Number')  
    '''
    param_names = [i for i in params1 if i != 'num_spec'] # list of all names of the parameters in dictionary
    ncols1 = get_ncols(params1,param_names) # getting the number of columns for each parameter, K, rho etc.
    params_as_matrix = params_dict_to_matrix(params1,param_names) # combining all the columns into one big matrix
    #print(f'{params_as_matrix=}')

    max_param = params_as_matrix.shape[1] # numer of columns i.e parameters columns K, r0, alpha, beta, rho_plus, rho_minus
    
    # Issue - sparse matrices might not contain biomass
    
    # Chunking algorithm goes here
    
    # add for loop to separate sprs_mat_arry into chunks
    
    biomass0_chunk = np.array([])
    dynamics0_chunk = np.array([])
    
    # extract per-species sparsity matrices, do appropriate non-vectorised calculations here
    for mat_i in sprs_mat_arry:
        
        s_mat = sprs_mat_arry[mat_i] # extract sparse matrix
        
        d_mat = dense(s_mat) # convert matrix to dense form 
        dynamics0_chunk = np.vstack((dynamics0_chunk,d_mat))
        
        # Mutation step (befoew or after dynamics0_chunk)
        
        
        # Calculate biomass
        #biomass0 = biomass(d_mat) # compute biomass = cell number x cell length
        biomass0 = d_mat[-1] # if we ignore cell length, as this is not included in the sparse matrix
        biomass0_chunk = np.concatenate((biomass0_chunk,biomass0))
        
        
        
    # do vectorised calculations here 

    biomass_update = np.reshape(biomass0_chunk,(len(biomass0_chunk),1))   
    
    dynamic_tot0 = params_matrix_to_dict(d_mat, param_names, ncols1) # convert big matrix to small matricies for dynamics calculation
    dynamic_tot0['s'] = biomass_update
        

    
        rtrn_to_s = sparse() # reconvert dense matrix to sparse form, might need to do some extra thinking
        # update dictionary
        
        sprs_mat_arry[mat_i] = rtrn_to_s
    
    
    #breakpoint()
    #print('table=',table)
     
    #X1 = np.concatenate(sparse_dense(table,max_param,params_as_matrix))
    #X = np.concatenate([sparse_dense(table[i],max_param,params_as_matrix[i]) for i in range(len(table))]) # table to matrix form

    #biomass0 = np.concatenate([biomass(table[i]) for i in range(len(table))]) # compute biomass = cell number x cell length
    #biomass_update = np.reshape(biomass0,(len(biomass0),1))

    #dynamic_tot0 = params_matrix_to_dict(X, param_names, ncols1) # convert big matrix to small matricies for dynamics calculation
    #dynamic_tot0['s'] = biomass_update
    #print(f'{dynamic_tot0=}')

    #y0 = np.concatenate((biomass0, c0)) # initial conditions for dynamics calculations

    #def dydt(t, sc): 
        '''
    Changing format to be solved by numerical integration. The format needs the ODE to have 2 variables
    Input:
        t - time span to solve the ODE
        sc - initial cSondition values for all the species and chemicals (array of length n_s + n_c)
    Output: 
        The correct equation format to be solved in solve_ivp, where input is time and initial condition
        '''
   #     return HM_list(t,sc,dynamic_tot0)
        #return HM_list_numba(t,sc,dynamic_tot0) 

    #y = dynamics_exact(dydt, y0, t)
    #n_c = len(c0) 
    #s = y.y.T[-1:,:-n_c][0] # Biomass final
    #c = y.y.T[-1:,-n_c:][0] # Chemicals final

    # Stochastic death at the end of cycle 
    #table_upd = death_table(params1,table,t)
    #breakpoint()
    

    #s_cut = cut_species(s,table_upd) # cutting combined species array to distinguish biomass of each species
    #division = [growth__(s_cut[i],table_upd[i]) for i in range(len(table_upd))]

    #ind = [division[i][1] for i in range(len(division))] # the indexes of the divided cells for all species 
    #table_divided = [division[i][0] for i in range(len(division))] # the table after cell division accourding to length
    #matrix_form = [sparse_dense(table_divided[i],max_param,params_as_matrix[i]) for i in range(len(table_divided))] # to extract param
    #print(f'{matrix_form=}')
    #new_table = [mutation_HM_null(table_divided[i],matrix_form[i],ind[i],mut_param,0,1) for i in range(len(table_divided))]

    #return new_table,c,s



