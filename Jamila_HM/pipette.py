from trial import *
from parameters import *
from utility_array_job import *


n_wells = int(sys.argv[1]) - 1 #initially they come as strings (var type)
n_cycles = int(sys.argv[2])

t1 = time.time()
for n in range(n_cycles):
    load_adults = [f'adults_data/adult_{id}_{n}' for id in range(n_wells)]
    adult_data = load_files_when_they_exist(load_adults, loader=open_file_)
    result_save_(f'saved_files/all_adults_{n}',adult_data)
    Newborns = pipette(adult_data,top_adults_num,BM_target,max_newborn)
    for id, well in enumerate(Newborns):
        result_save_(f'adults_data/Newborn_{id}_{n+1}', well) #saving Newborn data for each well
    for id in range(n_wells):
        remove(f'adults_data/adult_{id}_{n}.pck')
        remove(f'adults_data/Newborn_{id}_{n}.pck')

t2 = time.time()
print(f'Cycle {n} time pipette:',t2-t1)