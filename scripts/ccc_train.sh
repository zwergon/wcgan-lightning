 #!/bin/bash
#MSUB -r gmnist # JOB NAME
#MSUB -N 1 #Number of nodes
#MSUB -c 32
#MSUB -T 259200 # Wall time in seconds
#MSUB -E --gres=gpu:4
#MSUB -o gmnist_%I.out # Output file
#MSUB -e gmnist_%I.err # Error file
#MSUB -q a100
#MSUB -m store
#MSUB -Q long

module load gnu/8 mpi/openmpi/4 flavor/python3/cuda python3/3.10.6

#set -x

cd $CCCSTOREDIR/repositories/wcgan-lightning

. $CCCSTOREDIR/torch_env/bin/activate

export LD_LIBRARY_PATH=/ccc/products2/cudnn-8.6.0/Rhel_8__x86_64/system/default/lib:$LD_LIBRARY_PATH

ccc_mprun  python ${CCCSTOREDIR}/repositories/wcgan-lightning/scripts/train.py
