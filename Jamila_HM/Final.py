from trial import *
from parameters import *

C = 1 # number of cycles of maturation
wells = 100 # the total number of wells or partitions for the next cycle, i.e how many newborn communities required
Newborns = [table_HM for i in range(wells)] # starting conditions, newborns 



random.seed(32) 
t11 = time.time()
test_loop = community_selection_(C, t, step, mut_param,params_HM,c0, BM_target,Newborns, top_adults_num, max_newborn,False,f'date_{0}')
t21 = time.time()
time_new_fast = (t21-t11)
print('Local runtime:',time_new_fast)


# if __name__ == '__main__':
#     random.seed(0)
#     t1 = time.time()
#     test_parallel = community_selection(C,t,step,mut_param,params_HM,c0,BM_target,Newborns,top_adults_num,max_newborn,True,time.time())
#     t2 = time.time()
#     print('Cluster runtime C={}:'.format(C), t2-t1)
