from trial import *
from parameters import *
from utility_array_job import *


np.random.seed(100)
n_cycles = int(sys.argv[2]) # NEW OR CHANGED LINE argv[2] is 36 num_cycle
id = int(sys.argv[1])-1 # subtract one to convert from one-indexing to zero-indexing

t1 = time.time()
for n in range(n_cycles):
    if n==0:
        Initial_newborn = result_save_(f'adults_data/Newborn_{id}_{n}',table_HM)# initial Newborn data already given
        Newborns = load_files_when_they_exist([f'adults_data/Newborn_{id}_{n}'], loader=open_file_)[0]
        Adults = mature(params_HM,c0,Newborns,mut_param,step,t)
        result_save_(f'adults_data/adult_{id}_{n}', Adults)
    else:
        Newborns = load_files_when_they_exist([f'adults_data/Newborn_{id}_{n}'], loader=open_file_)[0]
        Adults = mature(params_HM,c0,Newborns,mut_param,step,t)
        result_save_(f'adults_data/adult_{id}_{n}', Adults)
t2 = time.time()
print(f'Cycle {n_cycles} time:', t2-t1)


