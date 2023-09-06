from time import sleep
import numpy as np
from os import path
import pickle
import sys
from os import remove



#for pickle files seperate, put something in the loader for pickle
def load_files_when_they_exist(files, total_wait_time=1200, loader=np.loadtxt, **kwargs):
    """Wait until files exist, and then load the files

    Arguments:
        * files (list): List of filepaths
        * total_wait_time (int): total wait time in seconds
        * loader (function): loader function. Options include np.loadtxt and np.load
        * kwargs: keyword arguments passed to the loader
    Returns:
        * list of file data in order of `files`
    """
    for i in range(10*int(total_wait_time)): 
        try:
            result = [loader(fname, **kwargs) for fname in files]
            assert np.all([len(_) > 0 for _ in result])
            return result
        except Exception:
            sleep(0.1)
    raise Exception(f'One or more files failed to appear after {total_wait_time} seconds.\nFile path set:\n{files}')



def result_save_(name, results):
    '''
    A function which saves the results from simulation. For example Adult data, which contains [table, biomass, Product]
    Input:
        name - the designated name of the file. The input should be in terms of string, i.e 'name'
        i - the cycle index. In the form of integer 
        results - the simulation results i.e the variable which contains all Adult from a single cycle
    Output:
        A folder with the name "name" which contains the results from the simulation
    '''
    
    file_ = open(f'{name}.pck', 'wb') # wb means opening a new file in the folder of the simulation
    pickle.dump(results, file_)
    file_.close()


def open_file_(name_file):
    '''
    Function which opens a file saved using pickle.
    Input:
        name_file - the name of the file with the simulation results. The input must be a string 'anythin'
        i - index of the file to open, i.e which cycle index. Type: int 
    Output:
        The simulation results decoded from the saved pickle file. Typically Adult data [table, biomass, Product] after each cycle.
    '''

    with open(f'{name_file}.pck','rb') as _:
        opening = pickle.load(_)
    return opening 
