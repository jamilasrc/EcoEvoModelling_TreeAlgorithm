#from trial import *
import numpy as np
import pandas as pd

def biomass_init(table):
    '''
    Calculates the biomass of species, which is the number of cells * cell length. 
    Input: 
        table - table which contains the species and subpopulations and their population size and length of cells
    Output: 
        The biomass value for a species and its subpopulation 
    '''
    biomass = table['Number'] * table['Length'] 
    # This function is bad because it requires the user to name variables as number and length. The same goes 
    #   for lots of the other functions (any that use dataframes or dictionaries). 
    #   If this is avoidable, we should change this (and data structures where necessary, hopefully it shouldn't matter).
    return biomass

# Saturation concentration for H and M species
K_MR = 1/3
K_MB = (1/3)*100
K_HR = 1/5
K_HM = np.array([[K_HR,0,0],[K_MR,K_MB,0]])

# Maximal growth rate for H and M species 
g_Hmax_val = 0.3
g_Mmax_val = 0.7
g_max = np.array([[g_Hmax_val,0],[0,g_Mmax_val]])

# Initially starting with constant death rate
death_H = g_Hmax_val*5e-3
death_M = g_Mmax_val*5e-3 
death =  np.array([[death_H,0],[0,death_M]])

# Consumption rate for H and M species analogous to alpha 
c_RM = 10**-4
c_RH = 10**-4
c_BM = 1/3
alpha_HM = np.array([[c_RH,c_RM],[0,c_BM],[0,0]])
alpha_HM = alpha_HM.T # for dynamics solver

# Production rate for H and M species analogous to beta
r_PM = 1
r_BH = 1
beta_HM = np.array([[0,0],[r_BH,0],[0,r_PM]])
beta_HM = beta_HM.T

# Cost function, that is allowed to mutate
fp = np.array([[0],[0.13]]) # the cost function for optimal HM community 

# Initial starting biomass of Helper and Manifacturer species in 1 well (Length = 1 is newborn, Length = 2 is mature adult)
table_H = pd.DataFrame([[1, 0, 40, 1]], columns= ['Parameter', 'Change', 'Number','Length'])
table_M = pd.DataFrame([[1, 0, 60, 1]], columns= ['Parameter', 'Change', 'Number','Length'])
table_HM = [table_H,table_M]

n_s = len(table_HM) # total number of species in the system
s = np.array([biomass_init(table_HM[i]) for i in range(n_s)]) # calculating biomass from table with H and M species

# Dictionary which contains all the parameters in HM community. 
params_HM = {}
params_HM['num_spec'] = n_s
params_HM['fp'] = fp
params_HM['K'] = K_HM
params_HM['alpha'] = alpha_HM
params_HM['beta'] = beta_HM
params_HM['g_max'] = g_max
params_HM['death'] = death
params_HM['s'] = s

# Initial chemical concentrations with the convention [Resource, Byproduct, Product]
R0 = 1
B0 = 0
P0 = 0
c0 = np.array([R0,B0,P0]) 

t = 17 # total time of a single maturation cycle

step = 0.05 # the timestep to reach final maturation time T=17
#t = 2*step

# Mutation parameters 
s_neg = 0.067 # mutation values for positive, ehancing 
s_pos = 0.05 # mutation values for negative, diminishing 
# mutation_p = 1e-2 # rate of mutation occuring during single cycle C=1
mutation_p = 2 * 10**-3; # rate of mutation occuring during single cycle C=1
null_prob = 0.5

mut_param = {}
mut_param['s_neg'] = s_neg
mut_param['s_pos'] = s_pos
mut_param['mutation_p'] = mutation_p
mut_param['null_prob'] = null_prob



# C = 5 # number of cycles of maturation
# wells = 10 # the total number of wells or partitions for the next cycle, i.e how many newborn communities required
BM_target = 100 # the target biomass for each well
max_newborn = 10 # each Adult community will contribute to this number of wells  
top_adults_num = 10 # the number of top performing adults to reproduce newborns from
# Newborns = [table_HM for i in range(wells)] # starting conditions, newborns 

#(open_file('date_0',0))

