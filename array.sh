#$ -l tmem=4G
#$ -l h_vmem=4G
#$ -l h_rt=48:0:0
#$ -S /bin/bash
#$ -j y
#$ -N adults_data
#$ -t 1-101
#$ -o ~/artificial_selection/

### PARAMETER SETTINGS ###
num_jobs_total=101 # no spaces allowed in these parameter designations!
num_cycles=2000

### WAIT FOR RESOURCES ###
cd ~/artificial_selection/
hostname
echo Waiting for resources...
date

num_jobs_running=$( qstat | grep $JOB_ID | grep ' r ' | wc -l )
while [ $num_jobs_running -lt $num_jobs_total ]
do
    sleep 10
    num_jobs_running=$( qstat | grep $JOB_ID | grep ' r ' | wc -l )
done

### RESOURCES ALLOCATED; RUN THE SIMULATION ###
echo Resources allocated! Running the simulation...
date

if [ $SGE_TASK_ID -lt $num_jobs_total ]
then
    python3 wells.py $SGE_TASK_ID $num_cycles # all jobs except for the last are roses
else
    python3 pipette.py $num_jobs_total $num_cycles # the last job is reserved for the gardener
fi
echo Finished!
date
